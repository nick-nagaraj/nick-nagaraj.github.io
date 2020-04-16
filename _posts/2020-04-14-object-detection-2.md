---
layout: post
title: License Plate Transcription using Object Detection (Part 2)
permalink: objdetect-2
---

Last week we looked into the basic functionality of the TensorFlow Object Detection API. We ran our object detection model to transcribe license plates. This week, we'll look into some of the options available to us during the training process. Next, we'll build some OpenCV logic to order the detections, creating a meaningful number. Finally, deployment will be the name of the game.  

## Ordering our detections
If you ran the driver program from last week, you'll notice that the detections are ordered by the confidence of each bounding box. While this is useful for understanding which letters our model is most accurate on, it does not reflect the order of the characters present on the license plate. So, how do we fix this?

Let's take a look at one of our license plates.

![Sample_2](/assets/NewYork.jpg)

We have access to the (x,y) coordinates of each bounding box. Thus, we sort the detections in ascending order of x-coordinates. This becomes our license plate number! Making some modifications to our driver code, and running it.

{% highlight py %}
{% endhighlight %}

![Final_Detection](/assets/FINAL_Detection.png)

Looks good!

## Ordering with Multiple Lines
I wish it were that simple. Sure, the code works for the above license plate. But it only works because all the characters present in the license plate are aligned along only one line. What happens if the content is divided along two lines?

![Line_2](/assets/Two_Line.jpeg)

This particular scenario is not common, but it does present an interesting edge case challenge to our 'out of the box' solution. The next bit is a bit technical in case you'd like to skip the details. It isn't particularly important.

Our first order of business is to detect how many lines of characters are present. License plates typically do not contain more than two lines. Thus our goal is to detect how many lines are present, and which line each character belongs to. Once we know which line each character belongs to, we sort the characters present on each line in ascending orders of their x-coordinates. Finally we concatenate the characters present on the first line with that of the second line.

### Which Character is on Which Line?
Here's what we know. Once we sort each detection by their y-coordinate, we can say for sure that the first element will be on the first line. We also know that the last element will be on the last line. Whether this 'last line' is the first line or the second line remains to be seen. So, we draw a horizontal line through the midpoint of the first element. Next, we draw a horizontal line through the midpoint of the last element. If the distance between the two horizontal lines is greater than a particular threshold, then we know that the first character on the license plate and the last character on the licence plate are seperated by a considerable gap. This means that the first and last character are on two seperate lines. If the distance between the two horizontal lines is lesser than a particular threshold, then odds are, it's on a single line.

![Line_1](/assets/Lines_1.png)
![Line_2](/assets/Lines_2.png)

### Sorting Each Line
Whether there's one line or two, once we know which line each character belongs to, we sort the line in ascending order of x-coordinates. Assuming there's one, we now have the transcribed number plate! If there's two, we sort the first line. Then, we sort the second line. Finally we append the sorted detections from the second line to the sorted detections from the first line. You can write your own script to automate this and output this information into a more useful format such as a CSV file.

## Options during Training
Often the best increases in performance lies in your training data. As the number of classes is large, it becomes difficult to ensure that the model can accurately detect each character. Increasing the amount of training data is the first thing you should try.

### Data Augmentation
The Tensorflow Object Detection API allows the user to customize data augmentation options according to need. Here's a list of all the options you can try.

{% highlight py %}
NormalizeImage normalize_image = 1;
RandomHorizontalFlip random_horizontal_flip = 2;
RandomPixelValueScale random_pixel_value_scale = 3;
RandomImageScale random_image_scale = 4;
RandomRGBtoGray random_rgb_to_gray = 5;
RandomAdjustBrightness random_adjust_brightness = 6;
RandomAdjustContrast random_adjust_contrast = 7;
RandomAdjustHue random_adjust_hue = 8;
RandomAdjustSaturation random_adjust_saturation = 9;
RandomDistortColor random_distort_color = 10;
RandomJitterBoxes random_jitter_boxes = 11;
RandomCropImage random_crop_image = 12;
RandomPadImage random_pad_image = 13;
RandomCropPadImage random_crop_pad_image = 14;
RandomCropToAspectRatio random_crop_to_aspect_ratio = 15;
RandomBlackPatches random_black_patches = 16;
RandomResizeMethod random_resize_method = 17;
ScaleBoxesToPixelCoordinates scale_boxes_to_pixel_coordinates = 18;
ResizeImage resize_image = 19;
SubtractChannelMean subtract_channel_mean = 20;
SSDRandomCrop ssd_random_crop = 21;
SSDRandomCropPad ssd_random_crop_pad = 22;
SSDRandomCropFixedAspectRatio ssd_random_crop_fixed_aspect_ratio = 23;
{% endhighlight %}

Edit the config file and replace the options as provided above.

### Different Models
Different deployment scenarios call for different models. Models that need to be run on low end hardware devices such the Raspberry Pi or your smartphone might require a lightweight model such as MobileNet. Assuming we have the resources to do so, you may choose to run a RCNN or NasNet. Whatever the model is, there are two things we need to swap out our model and begin training.

1. Config file
2. Pretrained weights
