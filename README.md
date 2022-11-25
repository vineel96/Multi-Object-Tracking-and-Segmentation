# Multiple Object Tracking and Segmentation on Videos

We peform multiple object tracking along with segmentation in a given input video.
![](https://github.com/vineel96/Multi-Object-Tracking-and-Segmentation/blob/master/mot.png)

## Implementation and Model Evaluation
- Siamese Network Backbone: ResNet-50
- Exemplar image size: 127 x 127 x 3
- Search image size: 255 x 255 x 3
- Inference: 
  - During tracking, SiamMask is simply evaluated once per frame. 
  - Output mask is selected using the location attaining the maximum score in the classification branch
- Mask refinement module: 
  - Made of upsampling layers and skip connections to get more accurate object mask
- Model Evaluation:
  - cd SiamMask/experiments/siammask_sharp
  - python ../../tools/demo_original_edit_2.py --resume SiamMask_DAVIS.pth --config config_davis.json --input_video testvideos/video.mp4 --classes person,tennis-racket


## Qualitative Results

For qualitative results on videos please visit drive link : [Qualitative Results](https://drive.google.com/drive/folders/1vJKqBHHg9aKMJFjp5ePKTyZ2ITlZOu-A?usp=share_link)

## Acknowledgment
This repo uses base codes with modifications from the following methods: [SiamMask](https://github.com/foolwood/SiamMask)
