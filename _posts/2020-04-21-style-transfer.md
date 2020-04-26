---
layout: post
title: Neural Style Transfer with Convolutional Neural Networks
permalink: style-transfer
---
Let's make some art! I'm terrible, but luckily pre-trained CNNs are great at it. What is Neural Style Transfer (NST)? It's probably best to explain with an image.

![Example](/assets/Example.jpeg)

The input consists of two images. One image provides the structural information and will form the basis of the output styled image. The other, will control the color and texture of our resultant image.

![Example](/assets/Example2.png)


## Why Convolutional Neural Networks?
The original purpose of CNNs was to produce better feature representations for classifying, detecting and segmenting objects in images. Their main purpose in NST's will be used to **encode representations of image features**. It is important to distinguish the difference between **encoding the image**, and **encoding image features**. Image features are inherently present across different images and make up the image itself. CNNs learn these image features, rather than learn what makes up a single image. That is why it is important to feed multiple images. Encoding images does not let the CNN learn what exactly distinguishes each image from another.



### CNNs for Representations
>This won't be an explanation on the actual working of a CNN. There are numerous available [resources](https://medium.com/technologymadeeasy/the-best-explanation-of-convolutional-neural-networks-on-the-internet-fbb8b1ad5df8). Instead, we'll focus on adapting it for NST.

Let's take a look at the original paper, [Gatys et al.](https://www.cv-foundation.org/openaccess/content_cvpr_2016/papers/Gatys_Image_Style_Transfer_CVPR_2016_paper.pdf) The original CNN that is used to extract image representations is VGG-19. Taking a look at the model architecture,

![CNN_Arch](/assets/CNN_Arch.jpg)

The main idea behind the paper was to learn important feature representations behind both the **content** image and **style** image. Across each subsequent convolutional layer, more complex features are learned. Thus, the key is to learn enough structural information from the **content** image and enough style information from the **style** image to form the perfect blended result image. The paper provides useful information on which layer to get this information from. The entire idea can be broken up into two key components.

* Since high level structural information is best represented in the final convolutional layers, we will be taking the output of the 5th convolutional layer.

* As for style information, it is difficult to pinpoint a single layer. Style is ambiguous, and requires both simple and complex features to truly be incorporated. The authors of the paper have determined that it is best to include information from each convolutional layer.

### Loss Function
Traditional loss functions try to quantify the difference between the **ground truth** and **predicted** values. This loss function is typically the main focus of optimization during the gradient descent process. For example, the loss function used in binary classification is Binary Cross-Entropy.

![Cross_Entropy](/assets/Cross_Entropy.png)

However, this loss function has no practical meaning for our application. What exactly is the "ground truth"? The paper introduces a custom loss function that simply consists of the weighted losses of the **content** and **style** image. Taking a look at the loss function,

![Loss_Style](/assets/Loss_Formula.jpeg)

* **L(content)** defines the structural difference between the content image and resultant output image. Intuitively we understand that this should be zero. We want our resultant image to ideally have the exact same structure as our input **content** image. This term is multiplied with weight **α**

* **L(style)** defines the style difference between the content image and resultant output image. Intuitively we understand that this should be zero. We want our resultant image to ideally have the exact same style as our input **style** image. This term is multiplied with weight **β**

What is the value of **α** and **β**? Well athough these weights can be tuned, the paper does suggest a value near 1 for **α** and 1e-6 for **β**

## Implementation
The framework of choice for this application will be PyTorch. We'll be using a pretrained VGG-19 network, but everything else will be from scratch. Installation instructions for the like will be [here](https://pytorch.org/)

### Setting up VGG-19
Let's go ahead and define the model.
