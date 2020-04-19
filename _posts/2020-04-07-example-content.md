---
layout: post
title: License Plate Transcription using Object Detection (Part 1)
permalink: objdetect-1
---

Today, we'll be looking at how to train your own object detection model that will learn to detect characters on a license plate (Part 1). Next, we'll build some logic to translate our detections into a meaningful number (Part 2).

> I will not be going over installation of the required packages. This bit of information is highly variable across systems and is best left to the reader to take care of. However, I will include resources on the mainstream installation methods.

 Finally, we'll be looking into deployment onto [Render](https://render.com/), so that it may be accessed through the internet for anyone to test. (Part 2)

## What other methods exist?
A license plate can be considered a block of text. Those of you familiar with OpenCV might realize that the library offers the [EAST Text Detector](https://www.pyimagesearch.com/2018/08/20/opencv-text-detection-east-text-detector/) to localize a continous block of text. However, this merely identifies the location of the text, and not the contents of the text itself.

For the actual character recognition, you make take advantage of an OCR algorithm called [pytesseract](https://pypi.org/project/pytesseract/). Passing the localized region of text obtained from our text detection algorithm to our OCR algorithm.

### Why Deep Learning then?

Two reasons.

The EAST Text Detector is excellent at localizing text on billboards, signs, and hand writing. However, it struggles to generate an accurate bounding box around the license plate itself. Passing this potentially incorrect bounding box to our Object Detection model as a preprocessing measure can reduce accuracy. Hence we avoid this step.

Pytesseract, while useful for variety of character recognition situations, does not do well on license plates as it has not been trained on those types of images. While there [exists a method to train it on your own dataset](https://tesseract-ocr.github.io/tessdoc/Training-Tesseract.html), learning how to build your own object detector for any general application is an essential skill.

![Gif](/assets/EAST.gif)


## Preparing the Training Data

There are 36 different classes (26 alphabets and 10 numbers) that the model will need to classify and localize. For each observation/character we need the following information.

- Name of the image containing the character.
- Width of the image.
- Height of the image.
- Class of the annotation.
- The (x,y) coordinate of the left-top corner of the annotated bounding box
- The (x,y) coordinate of the right-bottom corner of the annotated bounding box



The final csv file will look something like this:

![Sample_CSV](/assets/Screenshot_CSV.png)

As for our file structure, all the images (both training and testing images) will be kept in a single folder.

Let us split our training data into **train**, **test** and **validation** sets. We will be using sklearn's train_test_split function. We will be doing a 70-15-15 split.

Below is the code to split our data:
{% highlight py %}
from numpy.random import RandomState
from sklearn.model_selection import train_test_split
import pandas as pd

df = pd.read_csv('FINAL.csv')

train, test = train_test_split(df, test_size=0.3) #Divide the original csv file into 70-30.
train.to_csv('Train.csv', index=False) #Create training csv file from train dataframe.
test,validation = train_test_split(test, test_size = 0.5) #Divide the test dataframe into test and validation dataframes with a 50-50 split.
test.to_csv('Test.csv', index=False) #Create test csv file from test dataframe.
validation.to_csv('Validation.csv', index = False) #Create validation csv file from validation dataframe.
{% endhighlight %}

We will be using the [Tensorflow Object Detection API](https://github.com/tensorflow/models/tree/master/research/object_detection) to train our model. The installation instructions are [here](https://github.com/tensorflow/models/blob/master/research/object_detection/g3doc/installation.md).

![Flow_Process](/assets/Process_Flow.png)
The above image illustrates the process to input our training data into the model.

The API requires that we convert our CSV files into TFRecord files for the training process. The code to do so is given [here](). Running the script:


{% highlight bash %}
(cv) martianspeaks@KS-MSI:~/tensorflow-master/models-master/research$ python object_detection/generate_tfrecord.py --csv_input='/home/martianspeaks/Study/FINAL_Train.csv' --output_path='/home/martianspeaks/Study/FINAL_Train.record'

(cv) martianspeaks@KS-MSI:~/tensorflow-master/models-master/research$ python object_detection/generate_tfrecord.py --csv_input='/home/martianspeaks/Study/FINAL_Test.csv' --output_path='/home/martianspeaks/Study/FINAL_Test.record'
{% endhighlight %}



## Choosing a Model
Analysing our training data, we see that there are numerous classes that have to be accurately detected. As each detection is crucial to ensure that the license plate is recognized correctly, model accuracy has to be relatively high compared to other object detection tasks. Hence for the purposes of our task, we will be retraining the popular [Faster R-CNN](https://arxiv.org/pdf/1506.01497.pdf).

### Creating the Label Map

The label map will encode our class as integers which will be used in the training process. Given below is an example of a label map:

{% highlight py %}
item {
  id: 1
  name: 'class_1'
}
item {
  id: 2
  name: 'class_2'  
}
{% endhighlight %}


Edit the label map to reflect our 36 classes and save it as **object_detection.pbtxt**

### Setting up Model Parameters

Different deployment scenarios call for different models. Models that need to be run on low end hardware devices such the Raspberry Pi or your smartphone might require a lightweight model such as MobileNet. Assuming we have the resources to do so, you may choose to run a RCNN or NasNet. Whatever the model is, there are two things we need to swap out our model and begin training.

1. Pretrained weights
2. Config file

* [Download](http://download.tensorflow.org/models/object_detection/faster_rcnn_resnet50_coco_2018_01_28.tar.gz) the pretrained model file from the [model zoo](https://github.com/tensorflow/models/blob/master/research/object_detection/g3doc/detection_model_zoo.md).


 * [Download](https://github.com/tensorflow/models/blob/master/research/object_detection/samples/configs/faster_rcnn_resnet50_pets.config) the config file that specifies our training parameters. For the list of all the corresponding config files, click [here](https://github.com/tensorflow/models/tree/master/research/object_detection/samples/configs).

 Next, we edit the config file and make the following changes:

 1. Specify the path to the downloaded pretrained model in the **fine_tune_checkpoint** field.
 2. Specify the path to the **train.record** file in the **train_input_reader** field
 3. Specify the path to the **label_map (object_detection.pbtxt)** in the **label_map** field.
 4. Specify the path to the **test.record** file in the **eval_input_reader** field.
 5. Edit the **num_steps** field and specify the required number of steps. We will be experimenting with this value until we obtain an optimal model.

 *Search for "PATH_TO_BE_CONFIGURED" to find the fields that should be configured.*





## Training

Taking a look at what we've done so far:
- Generate TFRecord files for both train.csv and test.csv files
- Create a label map for our dataset
- Change the config files to reflect the required paths

We are now finally ready to begin training our model! Let's go ahead and move to the required folder.

{% highlight bash %}
(cv) martianspeaks@KS-MSI:~$ cd tensorflow-master/models-master/research
{% endhighlight %}

Exporting the PYTHONPATH so that our training files can find the included libraries:
{% highlight bash %}
(cv) martianspeaks@KS-MSI:~/tensorflow-master/models-master/research$ export PYTHONPATH=$PYTHONPATH:`pwd`:`pwd`/slim
{% endhighlight %}

**Running the training script:**
{% highlight bash %}
(cv) martianspeaks@KS-MSI:~/tensorflow-master/models-master/research$ python object_detection/model_main.py --logtostderr --model_dir='/home/martianspeaks/Study/' --pipeline_config_path='/home/martianspeaks/Study/Research/FasterRCNN_InceptionV2/faster_rcnn_inception_v2_coco_2018_01_28/faster_rcnn_inception_v2_coco.config' --num_train_steps=20000 --num_eval_steps=50
{% endhighlight %}

Explanation of the command parameters:

* logtostderr - Path to where the logs are displayed. Leaving this parameter blank indicated that the logs will be displayed onto STDOUT.
* model_dir - Path to where all the checkpoints will be saved.
* pipeline_config_path - Path to the config file.

If you've done everything perfectly up till this point, then you should start seeing the step number and the corresponding loss value. If not, try the following:

1. Double check the paths in your config file.
2. Export the PYTHONPATH in the appropriate folder as mentioned above. You should be in the 'research' folder when doing so.
3. Check to make sure your label map is created properly.
4. Reduce your batch_size to 1 in the config file.

### Monitoring the Training
The advantage of using the Tensorflow Object Detection API, is that we have access to Tensorboard. A useful tool for evaluating and monitoring all sorts of useful metrics and information.

Run the following command in another terminal window and click the link generated:
{% highlight bash %}
(cv) martianspeaks@KS-MSI:~$ tensorboard --logdir='/home/martianspeaks/Study/'
{% endhighlight %}

Where **logdir** is the path where all the checkpoints are being saved. You may run this **during** or **after** training.

Two useful tabs are the SCALARS and IMAGES tab. The SCALARS tab consists of useful graphs such as:
* mAP
* AR
* Various losses.

Under the IMAGES tab, you can move the slider to visualize the model performance at various steps in the training process.

Here's what it looks like at 3 different steps:

![Step_1](/assets/Step_1.png)

![Step_2](/assets/Step_2.png)

![Step_3](/assets/Step_3.png)

We can clearly see the impact of larger steps in model performance.


## Inference

Once training is done, it's time to visualize the fruits of our training! However, before we can plug in the model into our driver program, we need to convert our saved **checkpoint** into a **protobuf** model file. Running the following command to do so:

{% highlight bash %}
(cv) martianspeaks@KS-MSI:~/tensorflow-master/models-master/research$ python object_detection/export_inference_graph.py --input_type image_tensor --pipeline_config_path='/home/martianspeaks/Study/Research/FasterRCNN_InceptionV2/faster_rcnn_inception_v2_coco_2018_01_28/faster_rcnn_inception_v2_coco.config' --trained_checkpoint_prefix='/home/martianspeaks/Study/model.ckpt-14763' --output_directory='/home/martianspeaks/Study/'
{% endhighlight %}

Your saved model will be called **frozen_inference_graph.pb** and will be saved in the path specified by the **output_directory** parameter in the command above.

Run the following driver program to generate detections on your own image. Make sure it's from your validation set to get a more accurate understanding of model performance!

{% highlight py %}
import numpy as np
import tensorflow as tf
import cv2 as cv

x1 = (1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29,30,31,32,33,34,35,36)
x2 = (1,2,3,4,5,6,7,8,9,'A','B','C','D','E','F','G','H','I','J','K','L','M','N','O','P','Q','R','S','T','U','V','W','X','Y','Z',0)

dict = {}
for A, B in zip(x1, x2):
    dict[A] = B
font = cv.FONT_HERSHEY_SIMPLEX
# Read the graph.
with tf.gfile.FastGFile('frozen_inference_graph.pb', 'rb') as f:
    graph_def = tf.GraphDef()
    graph_def.ParseFromString(f.read())

with tf.Session() as sess:
    # Restore session
    sess.graph.as_default()
    tf.import_graph_def(graph_def, name='')

    # Read and preprocess an image.
    img = cv.imread('/home/martianspeaks/Study/datasets/kaggle/images/Arizona.jpg')
    rows = img.shape[0]
    cols = img.shape[1]
    #inp = cv.resize(img, (300, 300))
    inp = img[:, :, [2, 1, 0]]  # BGR2RGB

    # Run the model
    out = sess.run([sess.graph.get_tensor_by_name('num_detections:0'),
                    sess.graph.get_tensor_by_name('detection_scores:0'),
                    sess.graph.get_tensor_by_name('detection_boxes:0'),
                    sess.graph.get_tensor_by_name('detection_classes:0')],
                   feed_dict={'image_tensor:0': inp.reshape(1, inp.shape[0], inp.shape[1], 3)})

    # Visualize detected bounding boxes.
    num_detections = int(out[0][0])
    for i in range(num_detections):
        classId = int(out[3][0][i])
        score = float(out[1][0][i])
        bbox = [float(v) for v in out[2][0][i]]

        if score > 0.8:
            x = bbox[1] * cols
            y = bbox[0] * rows
            right = bbox[3] * cols
            bottom = bbox[2] * rows
            cv.rectangle(img, (int(x), int(y)), (int(right), int(bottom)), (125, 255, 51), thickness=2)
            cv.putText(img, str(dict[classId]), (int(x), int(y) - 5), font, 0.8, (0,0,255), 2, cv.LINE_AA)
            print (classId)
            print (score)


cv.imshow('TensorFlow MobileNet-SSD', img)
cv.waitKey()

{% endhighlight %}

### Results:

Some perfect detections:

![Test_1.png](/assets/Test_1.png)

![Test_3.png](/assets/Test_3.png)

Some not so perfect detections:

![Test_2.png](/assets/Test_2.png)

![Test_4.png](/assets/Test_4.png)

The model is by no means perfect yet. But it's performance has already exceeded what current OCR algorithms are capable of!




## Improvements

What can we possibly do to improve model performance?

1. Increase training data samples
2. Choose a more complex model
3. Visit data augmentation options

Next week, we'll be looking more in-depth into training parameters to boost model performance. We'll also look into how you can plug in other pretrained models available to us.
