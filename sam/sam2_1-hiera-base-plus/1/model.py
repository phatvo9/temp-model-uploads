import base64
from collections import defaultdict
import os
import sys
import uuid
ROOT = os.path.dirname(__file__)
sys.path.append(ROOT)

import tempfile
import time
from typing import Dict, Iterator, List, Union

from clarifai.runners.models.model_builder import ModelBuilder
from clarifai.runners.models.model_class import ModelClass
from clarifai.utils.logging import logger
from clarifai.runners.utils.data_types import (Image, Video, Audio, Region, Frame, Concept)
from PIL import Image as PILImage

import torch
import numpy as np

from sam2.sam2_video_predictor import SAM2VideoPredictor
from sam2.sam2_image_predictor import SAM2ImagePredictor
from sam2.automatic_mask_generator import SAM2AutomaticMaskGenerator


class MyRunner(ModelClass):
  """A custom runner that adds "Hello World" to the end of the text and replaces the domain of the
  image URL as an example.
  """

  def load_model(self):
    """Load the model here."""
    torch.autocast(device_type="cuda", dtype=torch.bfloat16).__enter__()
    device_major = torch.cuda.get_device_properties(0).major
    if device_major >= 8:
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    self.model_id = "facebook/sam2.1-hiera-base-plus"
    
    self.image_predictor = SAM2ImagePredictor.from_pretrained(
      self.model_id, 
      hydra_overrides_extra=[
        "++compile_image_encoder=True"
      ]
    )
    self.video_predictor = SAM2VideoPredictor.from_pretrained(
      self.model_id, 
      #vos_optimized=True,
      hydra_overrides_extra=[
        "++compile_image_encoder=True"
      ],
      device=device
    )
    self.automask_generator = SAM2AutomaticMaskGenerator.from_pretrained(
      model_id=self.model_id, 
      output_mode="binary_mask", 
      multimask_output=False
    )
    
    logger.info("----------------------")
    logger.info(
        "# Model is warming up. This could take few minutes depending on kind of GPU.")
    st_warmup = time.perf_counter()
    n_warmup = 0
    for _ in range(n_warmup):
      self._warmup_video_predictor()
    st_warmup = time.perf_counter() - st_warmup
    logger.info(
        f"## Model {self.model_id} takes {round(st_warmup, 3)} sec for {n_warmup} warmup runs.")
    logger.info("----------------------")
  
  def _warmup_video_predictor(self):
    
    points = np.array([[210, 350]], dtype=np.float32)
    labels = np.array([1], np.int32)
    inference_state = self.video_predictor.init_state(video_path=os.path.join(ROOT, "data/bedroom.mp4"))
    _ = self.video_predictor.add_new_points_or_box(
        inference_state=inference_state,
        frame_idx=150,
        obj_id=1,
        points=points,
        labels=labels,
    )
    with torch.autocast("cuda", torch.bfloat16):
      with torch.inference_mode():
        for (
            out_frame_idx,
            out_obj_ids,
            out_mask_logits,
        ) in self.video_predictor.propagate_in_video(inference_state, start_frame_idx=150):
          pass
    
  
  def prase_regions(self, frames: List[Frame], w = None, h = None):
    input_objects = {}
    denorm = True if isinstance(w, int) and isinstance(h, int) and (w and h) else False
    for frame in frames:
      frame_idx = frame.proto.frame_info.index
      for region in frame.regions:
        _x, _y = region.proto.region_info.point.col, region.proto.region_info.point.row
        points = (_x, _y) if sum((_x, _y)) else None
        obj_id = region.proto.track_id
        box = region.box if sum(region.box) else None
        input_mask = region.proto.region_info.mask.image.base64
  
        if points:
          point_x, point_y = points
          if denorm:
            point_x = point_x * w
            point_y = point_y * h
          if len(region.concepts):
            try:
              label = int(region.concepts[0].value)
            except:
              try:
                label = int(region.concepts[0].name)
              except:
                raise ValueError(f"Concept.name or value must be either 1 (positive) or 0 (negative), got {region.concepts[0]}")
          else:
            raise ValueError(f"No concept found for the input point, region={region}")
          
          current_point_obj = input_objects.get(obj_id, dict(
              points=[],
              obj_id=obj_id,
              labels=[],
              frame_idx=frame_idx,
              box=None
            ))
          current_point_obj["points"].append([point_x, point_y])
          current_point_obj["labels"].append(label)
          input_objects[obj_id] = current_point_obj
        elif box:
          if denorm:
            box = self.denormalize_box(box, w, h)
          input_objects[obj_id] = dict(
              points=None,
              obj_id=obj_id,
              labels=None,
              frame_idx=frame_idx,
              box=box
            )
        elif input_mask:
          numpy_mask = Image(bytes=input_mask).to_numpy()
          assert len(numpy_mask.shape ) == 2, ValueError(f"input mask must be gray scale image e.i 2 channels, got {len(numpy_mask.shape)} channels")
          input_objects[obj_id] = dict(
              points=None,
              obj_id=obj_id,
              labels=None,
              frame_idx=frame_idx,
              box=None,
              mask=numpy_mask
            )
        else:
          raise ValueError(f"Expected one of mask, box or point in region, but got region={region}")

    return list(input_objects.values())
  
  def denormalize_box(self, box, w, h):
    x1, y1, x2, y2 = box
    return x1*w, y1*h, x2*w, y2*h
  
  def denormalize_dict_inputs(self, inputs, w, h):
    box = inputs.get("box")
    new_box = []
    if isinstance(box, list) or isinstance(box, tuple):
      if not (isinstance(box[0], list) or isinstance(box[0], tuple)):
        box = [box]
      for each_box in box:
        x1,y1,x2,y2 = each_box
        new_box.append([x1*w, y1*h, x2*w, y2*h])
    points = inputs.get("points")
    new_points = []
    if points:
      for (x,y) in points:
        new_points.append([x*w, y*h])
    
    inputs.update(dict(box=new_box, points=new_points))
    return inputs
    
  def parse_regions_for_image(self, regions: List[Region], w=None, h=None):
    points = []
    labels = []
    boxes = []
    logger.info((w, h))
    denorm = True if w and h else False
    for region in regions:
      point_x, point_y = region.proto.region_info.point.col, region.proto.region_info.point.row
      box = region.box if sum(region.box) else None
      label = None
      if region.concepts:
        try:
          label = int(region.concepts[0].value)
        except:
          label = int(region.concepts[0].name)
        finally:
          labels.append(label)
        
      if box:
        if denorm:
          boxes.append(self.denormalize_box(box, w, h))
        else:
          boxes.append(box)
      if label is not None:
        if denorm:
          points.append([point_x*w, point_y*h])
        else: 
          points.append([point_x, point_y])
    
    return dict(
      box=boxes, 
      labels=labels if labels else None,
      points=points
      )
      
  
  @ModelClass.method
  def predict(
          self, image: Image, 
          regions: List[Region] = None, 
          dict_inputs: Dict = None, 
          round_mask: bool = False, 
          multimask_output:bool=False,
          denormalize_coord:bool = True,
  ) -> List[Region]:
    
    logger.info("# ----------------- Create Mask --------------- #")
    torch.cuda.empty_cache()
    st_time = time.perf_counter()
    
    assert regions is not None or dict_inputs is not None, ValueError(
        f"Required one of 'regions(List[Region])' or 'dict_inputs(Dict)'")
    image = np.array(image.to_pil().convert("RGB"))
    h, w = image.shape[:2]
    if denormalize_coord:
      logger.info("Denorm coords")
      if regions: 
        regions = self.parse_regions_for_image(regions, w=w, h=h) 
      else:
        regions = self.denormalize_dict_inputs(dict_inputs, w=w, h=h)
    else:
      regions = self.parse_regions_for_image(regions) if regions else dict_inputs
    logger.info(f"Input regions: {regions}")
    self.image_predictor.set_image(image)
    masks, scores, logits = self.image_predictor.predict(
      point_coords=np.array(regions.get("points"), np.float32) if regions.get("points") else None,
      point_labels=np.array(regions.get("labels"), np.int32) if regions.get("labels") else None,
      box=np.array(regions.get("box"), np.float32) if regions.get("box") else None,
      multimask_output=multimask_output
    )
    outputs = []
    for (mask, score) in zip(masks, scores):
      if len(mask.shape) > 2:
        mask = np.squeeze(mask, axis=0)
      mask = mask*255
      if round_mask:
        mask[mask >= 128] = 255
        mask[mask < 128] = 0
      mask = mask.astype("uint8")
      out_region = Region()
      out_region.proto.id = uuid.uuid4().hex
      out_region.proto.region_info.mask.image.CopyFrom(Image.from_pil(PILImage.fromarray(mask)).to_proto())
      out_region.concepts = [
        Concept(
          value=score[0] if len(score.shape) else score,
          name="1",
        )]
      outputs.append(out_region)
    
    mask_time = time.perf_counter() - st_time
    logger.info(f"Finished in {round(mask_time, 3)} seconds")
    
    torch.cuda.empty_cache()
    
    return outputs
      
      
  @ModelClass.method
  def generate(
    self, video: Video, 
    frames: List[Frame] = None, 
    list_dict_inputs: List[Dict] = None,
    denormalize_coord:bool = True,
  ) -> Iterator[Frame]:
    logger.info("# ----------- Start tracking --------------- #")
    
    torch.cuda.empty_cache()
    
    add_new_points_args = []
    add_new_mask_args = []
    assert frames is not None or list_dict_inputs is not None, ValueError(
        f"Required one of 'frames(List[Frame])' or 'list_dict_inputs(List[Dict])'")
    
    
    with tempfile.NamedTemporaryFile(delete=False, suffix=".mp4") as tmp:
      tmp.write(video.bytes)
      tmp_video_path = tmp.name
      
    try:
      logger.info("## Init state")
      inference_state = self.video_predictor.init_state(video_path=tmp_video_path)
    except Exception as e:
      raise RuntimeError(e)
    finally:
      if os.path.exists(tmp_video_path):
        os.remove(tmp_video_path)
    
    w, h = inference_state["video_width"], inference_state["video_height"]
    
    if denormalize_coord:
      if frames:
        regions = self.prase_regions(
        frames, w, h) 
      else:
        regions = [self.denormalize_dict_inputs(each, w=w, h=h) for each in list_dict_inputs]
    else:
      regions = self.prase_regions(
        frames) if frames else list_dict_inputs
    
    logger.info("## Adding objects")
    for region in regions:
      obj_id = region.get("obj_id")
      if obj_id is not None:
        frame_idx = region.get("frame_idx", 0)
        mask = region.get("mask", None)
        if isinstance(mask, str):
          mask = Image(bytes=base64.b64decode(mask)).to_numpy()
        if isinstance(mask, np.ndarray):
          logger.info(f"* Mask: id={obj_id}, frame_idx={frame_idx}, mask={mask.shape}")
          add_new_mask_args.append(dict(
            obj_id=obj_id,
            mask=mask,
            frame_idx=frame_idx,
          ))
        else:
          logger.info(f"Point or Box: {region}")
          add_new_points_args.append(
            dict(
              points=np.array(region.get("points", []), dtype=np.float32) if region.get("points") else None,
              box=np.array(region.get("box", []), dtype=np.float32) if region.get("box") else None,
              obj_id=obj_id,
              labels=np.array(region.get("labels"), np.int32) if region.get("labels") is not None else None,
              frame_idx=frame_idx,
            )
          )
        logger.info(f"    added")
    
    # add new points or box

    for each_point_arg in add_new_points_args:
      _, out_obj_ids, out_mask_logits = self.video_predictor.add_new_points_or_box(
          inference_state=inference_state,
          **each_point_arg
      )
    for each_mask_arg in add_new_mask_args:
      _, out_obj_ids, out_mask_logits = self.video_predictor.add_new_mask(
          inference_state=inference_state,
          **each_mask_arg
      )

    logger.info("## Perform tracking..")
    with torch.autocast("cuda", torch.bfloat16):
      with torch.inference_mode():
        for (
            out_frame_idx,
            out_obj_ids,
            out_mask_logits,
        ) in self.video_predictor.propagate_in_video(inference_state):
          regions = []
          for i, out_obj_id in enumerate(out_obj_ids):
            mask = (out_mask_logits[i] > 0.0).cpu().numpy()
            h, w = mask.shape[-2:]
            mask_image = (mask.reshape(h, w)*255).astype("uint8")
            mask_cl_image = Image.from_pil(PILImage.fromarray(mask_image))
            
            region = Region()
            region.proto.id = uuid.uuid4().hex
            region.proto.track_id = str(int(out_obj_id))
            region.proto.region_info.mask.image.CopyFrom(mask_cl_image.to_proto())
            regions.append(region)
          frame = Frame(regions=regions)
          frame.proto.frame_info.index = out_frame_idx
          
          yield frame

    torch.cuda.empty_cache()
  
  
  @ModelClass.method
  def segment_anything(self, image: Image) -> List[Region]:

    torch.cuda.empty_cache()
    
    np_image = image.to_numpy()
    masks_data = self.automask_generator.generate(np_image)
    outputs = []
    for mask in masks_data:
      bin_mask = mask["segmentation"]
      mask_image = (bin_mask*255).astype("uint8")
      mask_cl_image = Image.from_pil(PILImage.fromarray(mask_image))
      region = Region(concepts=[Concept(name="1", value=1.)])
      region.proto.id = uuid.uuid4().hex
      region.proto.region_info.mask.image.CopyFrom(mask_cl_image.to_proto())
      outputs.append(region)
    
    torch.cuda.empty_cache()
    
    return outputs

  
  def test(self):
    image = PILImage.open(os.path.join(ROOT, "data/img_sam2.jpg"))
    w, h = image.size
    image = Image.from_pil(image)
    ## Test auto mask
    masks = self.segment_anything(image)
    logger.info(f"Num gen masks = {len(masks)}")
    
    ## Test creating mask
    
    iregion = Region()
    iregion.box = [10, 20.2, 30, 40]
    
    iregion1 = Region()
    iregion1.proto.region_info.point.col = 50
    iregion1.proto.region_info.point.row = 50
    iregion1.concepts = [
        Concept(
            name="1",
            value=1
        )
    ]
    iregion2 = Region()
    iregion2.proto.region_info.point.col = 100
    iregion2.proto.region_info.point.row = 100
    iregion2.concepts = [
        Concept(
            name="0",
            value=0
        )
    ]
    out_regions = self.predict(
      image=image,
      regions=[iregion, iregion2]
    )
    # test correct norm
    iregion = Region()
    iregion.box = [10/w, 20.2/h, 30/w, 40/h]
    out_regions = self.predict(
      image=image,
      regions=[iregion]
    )
    iregion = Region()
    iregion.box = [10, 20.2, 30, 40]
    out_regions2 = self.predict(
      image=image,
      regions=[iregion],
      denormalize_coord=False
    )
    for i, each in enumerate(out_regions + out_regions2):
      mask = each.proto.region_info.mask.image.base64
      with open(f"mask_proto_{i}.jpg", "wb") as f:
        f.write(mask)
    
    out_regions = self.predict(
      image=image,
      dict_inputs=dict(
        box=[
          [100.1, 200.3, 500, 500],
          #[50.1, 85.3, 70, 150]
        ],
        points=[[50, 50],],
        labels=[0,]
        )
    )
    for i, each in enumerate(out_regions):
      mask = each.proto.region_info.mask.image.base64
      with open(f"mask_dict_{i}.jpg", "wb") as f:
        f.write(mask)
    
    ################################################################################
    
    ## Test tracking
    print("Test tracking..")
    video_path = os.path.join(ROOT, "data/bedroom.mp4")
    with open(video_path, "rb") as f:
      video = Video(bytes=f.read())
    
    frames = []
    region1 = Region()
    region1.proto.region_info.point.col = 50
    region1.proto.region_info.point.row = 50
    region1.proto.track_id = "0"
    region1.concepts = [
        Concept(
            name="1",
            value=1
        )
    ]
    frame1 = Frame()
    frame1.proto.frame_info.index = 0
    frame1.proto.data.regions.append(region1.to_proto())
    
    region2 = Region()
    region2.proto.region_info.point.col = 100
    region2.proto.region_info.point.row = 100
    region2.proto.track_id = "0"
    region2.concepts = [
      Concept(
        name="1",
        value=1
      )
    ]
    frame1.proto.data.regions.append(region2.to_proto())
    
    region5 = Region()
    region5.proto.region_info.point.col = 150
    region5.proto.region_info.point.row = 100
    region5.proto.track_id = "0"
    region5.concepts = [
      Concept(
        name="0",
        value=0
      )
    ]
    frame1.proto.data.regions.append(region5.to_proto())
    
    region6 = Region()
    region6.proto.region_info.point.col = 99 
    region6.proto.region_info.point.row = 88
    region6.proto.track_id = "3"
    region6.concepts = [
      Concept(
        name="1",
        value=1
      )
    ]
    frame1.proto.data.regions.append(region6.to_proto())
    
    region3 = Region()
    region3.box = [150, 150, 250, 250]
    region3.proto.track_id = "1"
    
    frame2 = Frame()
    frame2.proto.frame_info.index = 1
    frame2.proto.data.regions.append(region3.to_proto())
    
    region4 = Region()
    region4.box = [150, 180, 290, 250]
    region4.proto.track_id = "1"
    frame1.proto.data.regions.append(region4.to_proto())
    
    frames.append(frame1)
    frames.append(frame2)

    for each in self.generate(video, frames=frames):
      # print(each.proto.frame_info.index, [each_reg.proto.track_id for each_reg in each.regions])
      pass
    
    frame_objs = [
      dict(
        points=[[50, 50], [100, 100]],
        box=None,
        obj_id=0,
        labels=[1, 1],
        frame_idx=0
      ),
      dict(
          points=[[150, 150], [88, 77]],
          box=None,
          obj_id=2,
          labels=[1, 1],
          frame_idx=0
      ),
      dict(
          points=None,
          box=[1, 2, 3, 4],
          obj_id=1,
          labels=None,
          frame_idx=6
      )
    ]

    for each in self.generate(video, list_dict_inputs=frame_objs):
      # print(each.proto.frame_info.index, [each_reg.proto.track_id for each_reg in each.regions])
      pass
    
    
    ###
    
    mask = out_regions[0].proto.region_info.mask.image.base64
    region1 = Region(
      concepts=[Concept(name="1")],
      mask=Image(bytes=mask),
      track_id="1"
    )
    frame1 = Frame()
    frame1.proto.frame_info.index = 6
    frame1.proto.data.regions.append(region1.to_proto())
    frames = [frame1]
    
    for each in self.generate(video, frames=frames):
      # print(each.proto.frame_info.index, [each_reg.proto.track_id for each_reg in each.regions])
      pass
    
    region1 = [dict(
          points=None,
          box=None,
          obj_id=1,
          labels=None,
          frame_idx=6,
          mask=base64.b64encode(mask).decode('utf-8')
      )]
    
    for each in self.generate(video, list_dict_inputs=region1):
      # print(each.proto.frame_info.index, [each_reg.proto.track_id for each_reg in each.regions])
      pass
    