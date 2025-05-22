# Standard library imports
import os
import tempfile
import time
from io import BytesIO
from typing import List, Dict, Any, Iterator

# Third-party imports
import cv2
import torch
from PIL import Image as PILImage
from transformers import DFineForObjectDetection, AutoImageProcessor

# Clarifai imports
from clarifai.runners.models.model_class import ModelClass
from clarifai.runners.models.model_builder import ModelBuilder
from clarifai.runners.utils.data_types import Concept, Image, Video, Region
from clarifai.utils.logging import logger
from ultralytics import YOLO

def preprocess_image(image_bytes: bytes) -> PILImage:
    """Convert image bytes into RGB format suitable for model processing.

    Args:
        image_bytes: Raw image data in bytes format

    Returns:
        PIL Image object in RGB format ready for model input
    """
    return PILImage.open(BytesIO(image_bytes)).convert("RGB")


def video_to_frames(video_bytes: bytes) -> Iterator[bytes]:
    """Convert video bytes to frames.

    Args:
        video_bytes: Raw video data in bytes

    Yields:
        JPEG encoded frame data as bytes
    """
    with tempfile.NamedTemporaryFile(delete=False, suffix=".mp4") as temp_video_file:
        temp_video_file.write(video_bytes)
        temp_video_path = temp_video_file.name
        logger.info(f"temp_video_path: {temp_video_path}")

        video = cv2.VideoCapture(temp_video_path)
        logger.info(f"video opened: {video.isOpened()}")
        
        while video.isOpened():
            ret, frame = video.read()
            if not ret:
                break
            frame_bytes = cv2.imencode('.jpg', frame)[1].tobytes()
            yield frame_bytes
            
        video.release()
        os.unlink(temp_video_path)


class MyRunner(ModelClass):
    """A custom runner for DETR object detection model that processes images and videos"""

    def load_model(self):
        """Load the model here."""
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        logger.info(f"Running on device: {self.device}")
        from huggingface_hub import hf_hub_download
        
        checkpoint_path = "yolo11l.pt"
        checkpoint_path = hf_hub_download(
            repo_id="Ultralytics/YOLO11", filename=checkpoint_path)
        # model_path = os.path.dirname(os.path.dirname(__file__))
        # builder = ModelBuilder(model_path, download_validation_only=True)
        # checkpoint_path = builder.download_checkpoints(stage="runtime")
        
        self.model = YOLO(checkpoint_path)
        self.model.eval()
        logger.info("Done loading!")

    def post_process_region(self, results:list, threshold=0.5):
        regions = []
        for result in results:
            xyxyn = result.boxes.xyxyn.cpu().tolist()  # normalized
            names = [result.names[cls.item()] for cls in result.boxes.cls.int()]  # class name of each box
            confs = result.boxes.conf.cpu().float()  # confidence score of each box
            regions_per_input = []
            for (xyxy, name, conf) in zip(xyxyn, names, confs):
                conf = conf.item()
                logger.info(f"{xyxy=},{name=},{conf=}")
                if conf < threshold:
                    continue
                region = Region(
                    box=xyxy,
                    concepts=[Concept(
                        name=name,
                        value=conf
                    )]
                )
                regions_per_input.append(region)
            regions.append(regions_per_input)
            
        return regions
    
    @ModelClass.method 
    def predict(self, image: Image, conf_threshold:float=0.5) -> List[Region]:
        """Process a single image and return detected objects."""
        image_bytes = image.bytes
        image = preprocess_image(image_bytes)
        logger.info(f"Recieved image: {image}")
        results = self.model(image)  # predict on an image
        return self.post_process_region(results, threshold=conf_threshold)[0]

    @ModelClass.method
    def generate(self, video: Video, conf_threshold:float=0.5) -> Iterator[List[Region]]:
        """Process video frames and yield detected objects for each frame."""
        video_bytes = video.bytes
        frame_generator = video_to_frames(video_bytes)
        for frame in frame_generator:
            image = preprocess_image(frame)
            results = self.model(image)  # predict on an image
            return self.post_process_region(results, threshold=conf_threshold)[0]

    @ModelClass.method
    def stream_image(self, image_stream: Iterator[Image], conf_threshold:float=0.5) -> Iterator[List[Region]]:
        """Stream process image inputs."""
        logger.info("Starting stream processing for images")
        for image in image_stream:
            start_time = time.time()
            result = self.predict(image, conf_threshold=conf_threshold)
            yield result
            logger.info(f"Processing time: {time.time() - start_time:.3f}s")

    @ModelClass.method
    def stream_video(self, video_stream: Iterator[Video], conf_threshold:float=0.5) -> Iterator[List[Region]]:
        """Stream process video inputs."""
        logger.info("Starting stream processing for videos")
        for video in video_stream:
            start_time = time.time()
            for frame_result in self.generate(video, conf_threshold=conf_threshold):
                yield frame_result
            logger.info(f"Processing time: {time.time() - start_time:.3f}s")
        
    def test(self):
        """Test the model functionality."""
        import requests  # Import moved here as it's only used for testing
        
        # Test configuration
        TEST_URLS = {
            "images": [
                "https://samples.clarifai.com/metro-north.jpg",
                "https://samples.clarifai.com/dog.tiff"
            ],
            "video": "https://samples.clarifai.com/beer.mp4"
        }

        def get_test_data(url):
            return Image(bytes=requests.get(url).content)

        def get_test_video():
            return Video(bytes=requests.get(TEST_URLS["video"]).content)

        def run_test(name, test_fn):
            logger.info(f"\nTesting {name}...")
            try:
                test_fn()
                logger.info(f"{name} test completed successfully")
            except Exception as e:
                logger.error(f"Error in {name} test: {e}")

        # Test predict
        def test_predict():
            result = self.predict(get_test_data(TEST_URLS["images"][0]))
            logger.info(f"Predict result: {result}")

        # Test generate
        def test_generate():
            for detections in self.generate(get_test_video()):
                logger.info(f"First frame detections: {detections}")
                break

        # Test stream
        def test_stream():
            # Split into two separate test functions for clarity
            def test_stream_image():
                images = [get_test_data(url) for url in TEST_URLS["images"]]
                for result in self.stream_image(iter(images)):
                    logger.info(f"Image stream result: {result}")

            def test_stream_video():
                for result in self.stream_video(iter([get_test_video()])):
                    logger.info(f"Video stream result: {result}")
                    break  # Just test first frame

            logger.info("\nTesting image streaming...")
            test_stream_image()
            logger.info("\nTesting video streaming...")
            test_stream_video()

        # Run all tests
        for test_name, test_fn in [
            ("predict", test_predict),
            ("generate", test_generate),
            ("stream", test_stream)
        ]:
            run_test(test_name, test_fn)