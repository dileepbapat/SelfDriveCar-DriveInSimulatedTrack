# **Behavioral Cloning** 

#### Driving car in a simulated track to imitate the human actions for driving.


---

**Behavioral Cloning Project**

The goals/steps of this project are the following:
* Use the simulator to collect data on good driving behavior
* Build, a convolution neural network in Keras that predicts steering angles from images
* Train and validate the model with a training and validation set
* Test that the model successfully drives around track one without leaving the road
* Summarize the results with a written report


[//]: # (Image References)

[model]: ./images/map.png "Model Visualization"
[center]: ./images/center.jpg "Cenetr lane driving"
[left]: ./images/left.jpg "Recovery Image"
[right]: ./images/right.jpg "Recovery Image"
[flipped_left]: ./images/flipped_left.png "Flipped left Image"
[right_camera]: ./images/right_camera.png "Right Image"
[flipped_right]: ./images/flipped_right.png "Flipped right Image"



---
### Files Submitted & Code Quality

#### 1. Files in this project

My project includes the following files:
* model.py containing the script to create and train the model
* drive.py for driving the car in autonomous mode
* model.h5 containing a trained convolution neural network 
* writeup_report.md summarizing the results

#### 2. Submission includes functional code
Using the Udacity provided simulator and my drive.py file, the car can be driven
autonomously around the track by executing 
```sh
python drive.py model.h5
```
Select Autonomous mode to start running the model controlled driving.

#### 3. Submission code is usable and readable

The model.py file contains the code for training and saving the convolution neural network. The file shows the pipeline I used for training and validating the model, and it contains comments to explain how the code works.

The common functions are extracted for using with in-memory training or batch training.

### Model Architecture and Training Strategy

#### 1. An appropriate model architecture has been employed



| Layer        | Config        | Output Shape                 | weights |
|--------------|---------------|------------------------------|------:|
| Lambda       | Normalize     | (None, 160, 320, 3)          |     0 |
| Cropping2D   | (70, 25)      | (None, 65, 320, 3)           |     0 |
| Lambda       | 8, (5,5)      | (None, 61, 316, 8)           |   608 |
| MaxPooling   | (2,4)         | (None, 30, 79, 8)            |     0 |
| Conv2D       | 16, (3,3)     | (None, 28, 77, 16)           |  1168 |
| MaxPooling   | (2,4)         | (None, 14, 19, 16)           |     0 |
| Conv2D       | 24, (3,3)     | (None, 12, 17, 24)           |  3480 |
| Flatten      |               | (None, 4896)                 |     0 |
| Dropout      |  0.4          | (None, 4896)                 |     0 |
| Dense        |  elu          | (None, 108)                  |528876 |
| Dense        |  elu          | (None, 21)                   |  2289 |
| Dense        |  elu          | (None, 7)                    |   154 |
| Dense        |               | (None, 1 )                   |     8 |

My model consists of a convolution neural network with 5x5 filter sizes and depths 8, 15, 24. 
First two convolution layer are having max pooling layer followed by Conv2D layer. As images are
more wide in x (horizontal axis) max pooling is 2, 4. Another important aspect of these images is 
a line detection on x is more sensitive than on y, as car is deciding on left or right turn y small 
variations will not be causing any issue. 

The model includes Dense layers with ELU activation to introduce nonlinearity and the data is normalized in the model using a Keras lambda layer. 

#### 2. Attempts to reduce over-fitting in the model

The model contains dropout layers in order to reduce over-fitting with drop ratio of 0.4.  

The model was trained and validated on different data sets to ensure that the model was not over-fitting.
The model was tested by running it through the simulator and ensuring that the vehicle could stay on the track.

#### 3. Model parameter tuning

The model used an adam optimizer, so the learning rate decay was used with adam optimizer. Number of epoch to run was 
calibrated by running and comparing the cross-validation error. 

#### 4. Appropriate training data

* Initially started with normal driving in the lane, with the mouse. however, I noticed that car was not 
able to handle the edge cases when it goes close to curb near the turn.
Training data was chosen to keep the vehicle driving on the road. I used a combination of center lane driving, recovering from the left and right sides of the road.
A separate lap of training data was generated for recovering from the curbs.  

For details about how I created the training data, see the next section. 

### Model Architecture and Training Strategy

#### 1. Solution Design Approach

The overall strategy for deriving a model architecture was to ...

My first step was to use a convolution neural network model similar to the le net. I thought this model might be appropriate because it was
used successfully in traffic sign classification so it has capability to detect basic shapes such as curbs.

In order to gauge how well the model was working, I split my image and steering angle data into a training and validation set. 
I found that my first model had a high mean squared error on the training set and a high mean squared error on the validation set. This implied that the model was not generalizing. 
I had to add additional layers to make it fit well for the training set. However then I noticed still validation set error was high. This 
implied model was over-fitting for the training set. Adding the drop out layer with .4 reduced the gap in training error and validation error.

Then I looked at convolution kernel size and strides, adding strides was simplified the model however error rate increased. However,
Maxpooling for 4 pixel helped in improving the model performance on x-axis. Also tried different number of weights for convolution layers. 1,2,4 weights 
was not fitting the training set well, and increasing beyond 8,16,24 did not yield considerable improvement.

The final step was to run the simulator to see how well the car was driving around track one. There was one spot where the vehicle fell off the track.
In order to  improve the driving behavior in these cases, I had to add additional training data for handing recovery from curbs.

At the end of the process, the vehicle is able to drive autonomously around the track without leaving the road.

As next improvements, I am trying inception model with 3 parallel convolutional nets with different kernel size. 

#### 2. Final Model Architecture

The final model architecture (model.py) consisted of a convolution neural network with the following layers and layer sizes:
```
Sequential([
     Lambda(lambda x: (x / 255.0) - 0.5, input_shape=(160,320,3)),
     Cropping2D(((70, 25), (0,0))),
     Conv2D(8, (5,5)),
     MaxPooling2D((2,4)),
     Conv2D(16,  (3,3)),
     MaxPooling2D((2,4)),
     Conv2D(24, (3,3)),
     Flatten(),
     Dropout(.4),
     Dense(108, activation='elu'),
     Dense(21, activation='elu'),
     Dense(7, activation='elu'),
     Dense(1)
 ])

```
Here is a visualization of the architecture 

![model architecture][model]

#### 3. Creation of the Training Set & Training Process

To capture good driving behavior, I first recorded two laps on track one using center lane driving. Here is an example image of center lane driving:

![Center lane driving image][center]

I then recorded the vehicle recovering from the left side and right sides of the road back to center so that the vehicle would learn to .... These images show what a recovery looks like starting from left:

![Left image][left]
![center image][center]
![right image][right]

Then I repeated this process on track two in order to get more data points.

To augment the data sat, I also flipped images and angles thinking that this would augment the training set for bias towards left turn as the track is circular.
 For example, here is an image that has then been flipped:

![Flipped left camera image][flipped_left]

Right camera image and flipped one below:

![right camera image][right_camera]
![flipped right camera image][flipped_right]


After the collection process, I had 23k + 8k number of data points. Preprocessing of data was just normaliztion in the Keras pipeline.

I finally randomly shuffled the data set and put 20% of the data into a validation set. 

I used this training data for training the model. The validation set helped determine if the model was over or under fitting. 
The ideal number of epochs was 12 as evidenced by trying at different epochs on simulator after confirming with cross validation score.
I used an adam optimizer so that manually training the learning rate wasn't necessary. However I did use decay in Adam training optimization.


