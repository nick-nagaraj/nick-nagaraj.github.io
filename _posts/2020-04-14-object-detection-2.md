---
layout: post
title: License Plate Transcription using Object Detection (Part 2)
permalink: objdetect-2
---

Last week we looked into the basic functionality of the TensorFlow Object Detection API. We ran our object detection model to transcribe license plates. This week, we'll look into some of the options available to us during the training process. Next, we'll build some OpenCV logic to order the detections, creating a meaningful number. Finally, deployment will be the name of the game.  

## Ordering our detections
If you run the driver program from last week, you'll notice that the detections are ordered by the confidence of each bounding box. While this is useful for understanding which letters our model is most accurate on, it does not reflect the order of the characters present on the license plate. So, how do we fix this?

Let's take a look at one of our license plates.

![Sample_2](/assets/NewYork.jpg)

We have access to the (x,y) coordinates of each bounding box. Thus, we sort the detections in the ascending order of **x-coordinates**. This becomes our license plate number! Making some modifications to our driver code, and running it.

![Final_Detection](/assets/FINAL_Detection.png)

Looks good!

## Ordering with Multiple Lines
I wish it were that simple. Sure, the code works for the above license plate. But it only works because all the characters present in the license plate are aligned along only one line. What happens if the content is divided into two lines?

![Line_2](/assets/Two_Line.jpeg)

This particular scenario is not common, but it does present an interesting edge case challenge to our 'out of the box' solution. The next bit is a bit technical in case you'd like to skip the details. It isn't particularly important.

Our first order of business is to detect how many lines of characters are present. License plates typically do not contain more than two lines. Thus our goal is to detect how many lines are present, and which line each character belongs to. Once we know which line each character belongs to, we sort the characters present on each line in ascending orders of their **x-coordinates**. Finally, we concatenate the characters present on the first line with that of the second line.

### Which Character is on Which Line?
Here's what we know. Once we sort each detection by their **y-coordinate**, we can say for sure that the first element will be on the first line. We also know that the last element will be on the last line. Whether this 'last line' is the first line or the second line remains to be seen. So, we draw a horizontal line through the midpoint of the first element. Next, we draw a horizontal line through the midpoint of the last element. If the distance between the two horizontal lines is greater than a particular threshold, then we know that the first character on the license plate and the last character on the license plate are separated by a considerable gap. This means that the first and last character is on two separate lines. If the distance between the two horizontal lines is lesser than a particular threshold, then odds are, it's on a single line.

![Line_1](/assets/Lines_1.png)
![Line_2](/assets/Lines_2.png)

### Sorting Each Line
Whether there's one line or two, once we know which line each character belongs to, we sort the line in ascending order of **x-coordinates**. Assuming there's one, we now have the transcribed number plate! If there's two, we sort the first line. Then, we sort the second line. Finally, we append the sorted detections from the second line to the sorted detections from the first line. You can write your own script to automate this and output this information into a more useful format such as a CSV file. Given [here](/) (Yet to be uploaded) is the modified driver program.


## Deployment
[Render](https://render.com/) is a brilliant service that allows you to host your containerized applications on the cloud, allowing executable access to anyone to access your project! By the end of this, we'll be able to upload an image, and view the license plate number on the website.So how do we package our application so that all our dependencies are available? The answer, is that we create a isolated environment for our application so that it can run independently on any system or service. [Docker](https://docs.docker.com/get-started/) is the magic word. Docker allows us to neatly package our code into a container, which can later be run through a Render web service.

### Containerization
Let's go ahead and setup Docker on our system. Installation instructions are provided [here](/https://docs.docker.com/engine/install/ubuntu/). The key step is to create a Dockerfile. What is a Dockerfile? It consists instructions that the container runs to ensure that the application encased can run properly. We specify the instructions that are needed to run our application within the container. For this particular application, here is my Dockerfile:

{% highlight py %}
FROM "ubuntu:bionic"

RUN apt-get update && yes | apt-get upgrade

RUN apt-get install -y libsm6 libxext6 libxrender-dev libglib2.0-0

RUN apt-get install -y git python3-pip

RUN pip3 install --upgrade pip

RUN pip3 install opencv-python

RUN pip3 install tensorflow==1.15.0

RUN pip3 install fastai==1.0.52

RUN apt-get install -y protobuf-compiler python3-pil python3-lxml

RUN pip3 install matplotlib

RUN mkdir -p /tensorflow

RUN pip3 install aiofiles==0.4.0

RUN pip3 install uvicorn==0.7.1

RUN pip3 install aiohttp==3.5.4

RUN pip3 install asyncio==3.4.3

RUN pip3 install pillow~=6.0

RUN pip3 install python-multipart==0.0.5

RUN pip3 install starlette==0.12.0

RUN git clone https://github.com/tensorflow/models.git /tensorflow/models

COPY required_files /tensorflow/models/research/object_detection/required_files

WORKDIR /tensorflow/models/research

EXPOSE 8888

CMD ["python3", "/tensorflow/models/research/object_detection/required_files/app/server.py", "serve"]

{% endhighlight %}

For an explanation of the respective commands used, take a look at the Docker documentation.

Next, go ahead and create your own repository on GitHub. Upload the Dockerfile as well as the required files to the repo. Next, create an account on Render, and link the repo. Everything should be up and running! Given below is a screenshot of our page!

>I won't go into the creation of the webpage itself, as it delves too much into HTML, CSS and JS. I have uploaded the page onto the repo where you can pick up a copy.

![Webpage](/assets/Webpage.png)

Of course, I can't actually host the webpage at all times as the service costs $7 a month. I will explore [Heroku](https://www.heroku.com)'s free option at a later date. Note that we will probably have to switch to a lighter model such as SSD MobileNet as the service only allows for 512MB of RAM + ROM. 
