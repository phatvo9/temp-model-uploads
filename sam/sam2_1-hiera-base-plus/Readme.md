# sam-2.1-hiera-base-plus

[Model source](https://huggingface.co/facebook/sam2.1-hiera-base-plus)

# Supported features

## Automatic mask generation - `segment_anything` (Unary-Unary)
Generates masks of all objects in given image.

**Args**:
  * image (Image): input image

**Returns**:
  List[Region]: List of generated masks accessed by region.mask or region.proto.region_info.mask

Example:

```python
from PIL import Image as PILImage
from clarifai.runners.utils.data_types import (Image, Region)
from clarifai.client import Model

model = Model(url="")

input_image = Image.from_pil(PILImage.open("path/to/image.png"))
regions = model.segment_anything(image=image)

for reg in regions:
  mask = region.mask
  # or mask = Image.from_proto(region.proto.region_info.mask)
  # do something with mask here

```

## Create Mask by Points/Boxes - `predict` (Unary-Unary)
Create mask by points or boxes prompt.

**Args**:
  * image (Image): input image.
  
  * regions (List[Region]): List of region prompts where `point` or `box` can be used either. 
    To identify positive (objectness) or negative (background) point, 
    set `region.concepts = [Concepts(name="1", value=1)]` and `region.concepts = [Concepts(name="0", value=0)]` respectively. You can use either `regions` or `dict_inputs`, but not both at the same time.
  
  * dict_inputs (Dict): kwargs of SAM2ImagePredictor.predict(..) method. See this [example](https://github.com/facebookresearch/sam2/blob/main/notebooks/image_predictor_example.ipynb) for details. You can use either `regions` or `dict_inputs`, but not both at the same time.

  * multimask_output (bool): Whether or not return multiple masks for each. Default to False.

**Returns**:
  List[Region]: List of generated masks accessed by region.mask or region.proto.region_info.mask.

Example:
```python
from PIL import Image as PILImage
from clarifai.runners.utils.data_types import (Image, Region, Concept)
from clarifai.client import Model

model = Model(url="")

input_image = Image.from_pil(PILImage.open("path/to/image.png"))

# points
## Region
prompt = [
  Region(point=[0.1, 0.2, 0], concept=[Concept(name="1", value=1.0)]),
  Region(point=[0.2, 0.3, 0], concept=[Concept(name="0", value=.0)]),
]
regions = model.predict(image=image, regions=prompt)
## dict_inputs
prompt = dict(
  points=[[0.1, 0.2], [0.2, 0.3]],
  labels=[1, 0]
)
regions = model.predict(image=image, dict_inputs=prompt)

# box
## Region
prompt = [
  Region(box=[0.1, 0.2, 0.3, 0.4]),
]
regions = model.predict(image=image, regions=prompt)
## dict_inputs
prompt = dict(
  box=[0.1, 0.2, 0.3, 0.4]
)
regions = model.predict(image=image, dict_inputs=prompt)

# Use outputs
for reg in regions:
  mask = region.mask
  # or mask = Image.from_proto(region.proto.region_info.mask)
  # do something with mask here

```

## Track - `generate` (Unary-Stream)
Track objects in video

**Args**:
  * video (Video): Input video

  * frames (List[Frame]): Each frame, `frame.data.regions` contains positions (points or box) like explained in `predict` method; `frame_idx` identified by `frame.frame_info.index` and id of object by `frame.data.regions[...].track_id`
  
  * list_dict_inputs (List[Dict]): Follow `SAM2VideoPredictor.add_new_points_or_box()` method including `points, box, obj_id, labels, frame_idx` see this [example](https://github.com/facebookresearch/sam2/blob/main/notebooks/video_predictor_example.ipynb) for details

**Return**:
  Iterator[Frame]: contains generated masks and tracked result of objects in frame.

Example:
```python
from PIL import Image as PILImage
from clarifai.runners.utils.data_types import (Video, Region, Concept, Frame)
from clarifai.client import Model

model = Model(url="")
video_path = ""
with open(video_path, "rb") as f:
  video = Video(bytes=f.read())

# 1. Use `frames`
frame0 = Frame(
    regions=[
      Region(point=[0.1, 0.2, 0], concept=[Concept(name="1", value=1.0)]),
      Region(point=[0.2, 0.3, 0], concept=[Concept(name="0", value=.0)]),
    ],
    track_id="1"
  )
frame0.proto.frame_info.index = 0
frame1 = Frame(
    regions=[
      Region(point=[0.11, 0.22, 0], concept=[Concept(name="1", value=1.0)]),
      Region(point=[0.22, 0.33, 0], concept=[Concept(name="0", value=.0)]),
    ],
    track_id="2"
  )
frame1.proto.frame_info.index = 1

# region00 = Region(concept=[Concept(name="1", value=1.0)])
# region00.proto.region_info.point.col = 0.1
# region00.proto.region_info.point.row = 0.2
# region01 = Region(concept=[Concept(name="0", value=.0)])
# region01.proto.region_info.point.col = 0.2
# region01.proto.region_info.point.row = 0.3
# frame0 = Frame(regions=[region00, region01])
# frame0.proto.frame_info.index = 0

# region10 = Region(concept=[Concept(name="1", value=1.0)])
# region10.proto.region_info.point.col = 0.11
# region10.proto.region_info.point.row = 0.22
# region11 = Region(concept=[Concept(name="0", value=.0)])
# region11.proto.region_info.point.col = 0.22
# region11.proto.region_info.point.row = 0.33
# frame1 = Frame(regions=[region10, region11])
# frame1.proto.frame_info.index = 1

output_frames = model.generate(frames=[frame0, frame1])

# 2. Use `list_dict_inputs`
frame_objs = [
      dict(
        points=[[0.1, 0.2], [0.2, 0.3]],
        box=None,
        obj_id=0,
        labels=[1, 0],
        frame_idx=0
      ),
      dict(
          points=[[0.11, 0.22], [0.22, 0.33]],
          box=None,
          obj_id=1,
          labels=[1, 0],
          frame_idx=1
      ),
]

output_frames = model.generate(list_dict_inputs=frame_objs)

# 3. Use output
for frame in output_frames:
  for region in frame.regions:
    mask = region.mask
    track_id = region.track_id
    # mask = Image.from_proto(region.proto.region_info.mask)
    # track_id = region.proto.track_id
    # do something with mask here

```