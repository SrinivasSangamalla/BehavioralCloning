Behavioral Cloning
Writeup Template
You can use this file as a template for your writeup if you want to submit it as a markdown file, but feel free to use some other method and submit a pdf if you prefer.
Behavioral Cloning Project

The goals / steps of this project are the following:

Use the simulator to collect data of good driving behavior
Build, a convolution neural network in Keras that predicts steering angles from images
Train and validate the model with a training and validation set
Test that the model successfully drives around track one without leaving the road
Summarize the results with a written report
Rubric Points
I considered the rubric points individually and described how I addressed each point in my implementation.
Files Submitted & Code Quality
1. Submission includes all required files and can be used to run the simulator in autonomous mode
My project includes the following files:

model.py containing the script to create and train the model
drive.py for driving the car in autonomous mode
model.h5 containing a trained convolution neural network
README.md summarizing the results
2. Submission includes functional code
Using the Udacity provided simulator and my drive.py file, the car can be driven autonomously around the track by executing

3. Submission code is usable and readable
The model.py file contains the code for training and saving the convolution neural network. The file shows the pipeline I used for training and validating the model, and it contains comments to explain how the code works.

Model Architecture and Training Strategy
1. An appropriate model architecture has been employed
My model consists of a convolution neural network with 3x3 filter sizes and depths between 24 and 64 (model.py lines 53-73)

The model includes RELU layers to introduce nonlinearity (code lines 57-66), and the data is normalized in the model using a Keras lambda layer (code line 62).

2. Attempts to reduce overfitting in the model
The model contains dropout layers in order to reduce overfitting (model.py lines 56 and 63).

The model was trained and validated on different data sets to ensure that the model was not overfitting. The model was tested by running it through the simulator and ensuring that the vehicle could stay on the track.

3. Model parameter tuning
The model used an adam optimizer, so the learning rate was not tuned manually (model.py line 82).

4. Appropriate training data
Training data was chosen to keep the vehicle driving on the road. I used a combination of center lane driving, recovering from the left and right sides of the road.

For details about how I created the training data, see the next section.

Model Architecture and Training Strategy
1. Solution Design Approach
The overall strategy for deriving a model architecture was to train and increment the findings until I found something that works.

My first step was to use a convolution neural network model similar to the LeNet used in the tutorials. This did OK, but the car had difficulty making the sharp right turn 2/3 of the track.

In order to gauge how well the model was working, I split my image and steering angle data into a training and validation set. I found that my first model had a low mean squared error on the training set but a high mean squared error on the validation set. This implied that the model was overfitting.

To combat the overfitting, I modified the model so that the images are cropp

The final step was to run the simulator to see how well the car was driving around track one. There were a few spots where the vehicle fell off the track. Biggest issue was first with the are that was missing the side yellow line and the right hand turn.

To improve the driving behavior in these cases, I recorded a few laps going backwards and I also flipped the images

At the end of the process, the vehicle is able to drive autonomously around the track without leaving the road.

2. Final Model Architecture
The final model architecture (model.py lines 53-73) consisted of a convolution neural network similar to the Nvidia's self driving car architecture.

Output from the tensorflow log. Only 3 epochs were used, as the loss rate dropped rapidly.

(/opt/carnd_p3/behavioral) root@1ebb787954af:/home/workspace/CarND-Behavioral-Cloning-P3# python model.py
Using TensorFlow backend.
Train on 38572 samples, validate on 9644 samples
Epoch 1/3
2020-12-13 18:55:04.639337: W tensorflow/core/platform/cpu_feature_guard.cc:45] The TensorFlow library wasn't compiled to use SSE4.1 instructions, but these are available on your machine and could speed up CPU computations.
2020-12-13 18:55:04.639397: W tensorflow/core/platform/cpu_feature_guard.cc:45] The TensorFlow library wasn't compiled to use SSE4.2 instructions, but these are available on your machine and could speed up CPU computations.
2020-12-13 18:55:04.639418: W tensorflow/core/platform/cpu_feature_guard.cc:45] The TensorFlow library wasn't compiled to use AVX instructions, but these are available on your machine and could speed up CPU computations.
2020-12-13 18:55:04.639428: W tensorflow/core/platform/cpu_feature_guard.cc:45] The TensorFlow library wasn't compiled to use AVX2 instructions, but these are available on your machine and could speed up CPU computations.
2020-12-13 18:55:04.639437: W tensorflow/core/platform/cpu_feature_guard.cc:45] The TensorFlow library wasn't compiled to use FMA instructions, but these are available on your machine and could speed up CPU computations.
2020-12-13 18:55:04.856560: I tensorflow/stream_executor/cuda/cuda_gpu_executor.cc:893] successful NUMA node read from SysFS had negative value (-1), but there must be at least one NUMA node, so returning NUMA node zero
2020-12-13 18:55:04.857399: I tensorflow/core/common_runtime/gpu/gpu_device.cc:955] Found device 0 with properties: 
name: Tesla K80
major: 3 minor: 7 memoryClockRate (GHz) 0.8235
pciBusID 0000:00:04.0
Total memory: 11.17GiB
Free memory: 11.10GiB
2020-12-13 18:55:04.857451: I tensorflow/core/common_runtime/gpu/gpu_device.cc:976] DMA: 0 
2020-12-13 18:55:04.857474: I tensorflow/core/common_runtime/gpu/gpu_device.cc:986] 0:   Y 
2020-12-13 18:55:04.857534: I tensorflow/core/common_runtime/gpu/gpu_device.cc:1045] Creating TensorFlow device (/gpu:0) -> (device: 0, name: Tesla K80, pci bus id: 0000:00:04.0)
38572/38572 [==============================] - 112s 3ms/step - loss: 0.0426 - val_loss: 0.0366
Epoch 2/3
38572/38572 [==============================] - 106s 3ms/step - loss: 0.0272 - val_loss: 0.0334
Epoch 3/3
38572/38572 [==============================] - 106s 3ms/step - loss: 0.0257 - val_loss: 0.0324
3. Creation of the Training Set & Training Process
To capture good driving behavior, I first recorded three laps on track one using center lane driving.

And here are some sample images from the test track

I then recorded the vehicle recovering from the left side and right sides of the road back to center so that the vehicle would learn to recover when going close to the edge.

This recovery turned out to be not accepted, so I made improvements to the model so that the car doesn't go outside the track

To augment the data sat, I also flipped images and angles thinking that this would provide more data

I then preprocessed this data by normalizing it and cropping it.

I finally randomly shuffled the data set and put 20% of the data into a validation set.

I had issues with the car making the right turn, so I went back and recorded a few recoveries from left to right. This helped.

I used this training data for training the model. The validation set helped determine if the model was over or under fitting. The ideal number of epochs was 20 as evidenced by the loss rate stabilizing.

In the second submission I only used 3 epochs, as the loss rate dropped really fast.

I used an adam optimizer so that manually training the learning rate wasn't necessary.