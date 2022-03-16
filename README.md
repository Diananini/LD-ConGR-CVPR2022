# LD-ConGR-CVPR2022-Long-Distance-Continuous-Gesture-Recognition
A Large RGB-D Video Dataset for Long-Distance Continuous Gesture Recognition

## Introduction
Gesture recognition plays an important role in natural human-computer interaction and sign language recognition. Existing research on gesture recognition is limited to close-range interaction such as vehicle gesture control and face-to-face communication. To apply gesture recognition to long-distance interactive scenes such as meetings and smart homes, we establish a large RGB-D video dataset LD-ConGR. LD-ConGR is distinguished from existing gesture datasets by its **long-distance gesture collection**, **fine-grained annotations**, and **high video quality**. Specifically, 1) the farthest gesture provided by the LD-ConGR is captured 4m away from the camera while existing gesture datasets collect gestures within 1m from the camera; 2) besides the gesture category, the temporal segmentation of gestures and hand location are also annotated in LD-ConGR; 3) videos are captured at high resolution (1280 x 720 for color streams and 640 x 576 for depth streams) and high frame rate (30 fps).

## The LD-ConGR Dataset
![image]()
The LD-ConGR dataset is developed for the long-distance gesture recognition task. It contains 10 gesture classes (), of which three are static gestures and seven are dynamic gestures. A total of 542 videos and 44,887 gesture instances are collected in LD-ConGR. The videos are collected from 30 subjects in 5 different scenes and captured in a third perspective with Kinect V4. Each video contains a color stream and a depth stream. The two streams are recorded synchronously at 30 fps with resolutions of 1280 x 720 and 640 x 576, respectively.
