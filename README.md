# Traffic Sign Recognition 

---

**Build a Traffic Sign Recognition Project**

The goals / steps of this project are the following:
* Load the data set (see below for links to the project data set)
* Explore, summarize and visualize the data set
* Design, train and test a model architecture
* Use the model to make predictions on new images
* Analyze the softmax probabilities of the new images
* Summarize the results with a written report


[//]: # (Image References)

[image1]: ./examples/visualization.jpg "Visualization"
[image2]: ./examples/grayscale.jpg "Grayscaling"
[image3]: ./examples/random_noise.jpg "Random Noise"
[image4]: ./examples/colour_image_example.png "Color image example"
[image5]: ./examples/grayscale_example.png "Grayscale example"
[image6]: ./examples/Task3/crop1_lg.png "New image 1"
[image7]: ./examples/Task3/crop2_lg.png "New image 2"
[image8]: ./examples/Task3/crop3_lg.png "New image 3"
[image9]: ./examples/Task3/crop4_lg.png "New image 4"
[image10]: ./examples/Task3/crop5_lg.png "New image 5"

## Rubric Points
### Here I will consider the [rubric points](https://review.udacity.com/#!/rubrics/481/view) individually and describe how I addressed each point in my implementation.  

---
### Writeup / README

#### 1. Provide a Writeup / README that includes all the rubric points and how you addressed each one. You can submit your writeup as markdown or pdf. You can use this template as a guide for writing the report. The submission includes the project code.

You're reading it! and here is a link to my [project code](https://github.com/Pepperrs/CarND-Traffic-Sign-Classifier-Project).

### Data Set Summary & Exploration

#### 1. Provide a basic summary of the data set and identify where in your code the summary was done. In the code, the analysis should be done using python, numpy and/or pandas methods rather than hardcoding results manually.

The code for this step is contained in the second code cell of the IPython notebook.  

I used the pandas library to calculate summary statistics of the traffic
signs data set:

* The size of training set is 34799 images.
* The size of test set is 12630 images.
* The shape of a traffic sign image is (32, 32, 3).
* The number of unique classes/labels in the data set is 43.

#### 2. Include an exploratory visualization of the dataset and identify where the code is in your code file.

The code for this step is contained in the xxx code cell of the IPython notebook.  

#todo 


![alt text][image1]

### Design and Test a Model Architecture

#### 1. Describe how, and identify where in your code, you preprocessed the image data. What tecniques were chosen and why did you choose these techniques? Consider including images showing the output of each preprocessing technique. Pre-processing refers to techniques such as converting to grayscale, normalization, etc.

The code for this step is contained in the sixth to eigth code cell of the IPython notebook.

My preprocessing pipeline consisted of 
1. **a conversion to grayscale**
	This was done to reduce the required computation to extract features from available images. As it tourned out, color is nice to have for the human eye but it does not increase detection rates noticeably for our purposes
2. **normalization of all images in each set**
	I applied normalization to make it easier for the algorithm to get transferable information of the given images.
3. **reshaping the images to (32, 32, 1) get rid of the now unused dimensions of rgb**
	The images were reshaped to allow for processing with LeNet architecture.

Here is an example of a traffic sign image before and after preprocessing.

![alt text][image4]
![alt text][image5]

#### 2. Describe how, and identify where in your code, you set up training, validation and testing data. How much data was in each set? Explain what techniques were used to split the data into these sets. (OPTIONAL: As described in the "Stand Out Suggestions" part of the rubric, if you generated additional data for training, describe why you decided to generate additional data, how you generated the data, identify where in your code, and provide example images of the additional data)

The images were already split into training, testing and validation set beforehand.

My training set had 34799 images. My validation and test set had 12630 and 4410 images

The images were not further augmented, however this could have been used to enlarge the training set with techniques such as mirroring, stretching, or otherwise squeing certain images.

#### 3. Describe, and identify where in your code, what your final model architecture looks like including model type, layers, layer sizes, connectivity, etc.) Consider including a diagram and/or table describing the final model.

My final model consisted of a LeNet-5 Architecture with added dropout in Layer 3 and 4 resulting in the following layers:

| Layer         		|     Description	        		| 
|:-----------------------------:|:---------------------------------------------:| 
| Input         		| 32x32x1 Grayscale image   			| 
| Convolution 3x3     		| 1x1 stride, valid padding, outputs 28x28x6 	|
| RELU				|						|
| Max pooling		      	| 2x2 stride, outputs 14x14x6			|
|				|						|
| Convolution 3x3		| 1x1 stride, valid padding, outputs 10x10x16	|
| RELU				| 						|
| Max pooling		      	| 2x2 stride, outputs 5x5x16			|
|				|						|
| Flatten			| outputs 400					|
|				|						|
| Fully connected		| input 400, outputs 120			|
| RELU				| 						|
| Dropout			| keep probability 0.5				|
|				|						|
| Fully connected		| input 120, outputs 84				|
| RELU				| 						|
| Dropout			| keep probability 0.5				|
|				|						|
| Fully connected		| input 83, outputs 43				|


#### 4. Describe how, and identify where in your code, you trained your model. The discussion can include the type of optimizer, the batch size, number of epochs and any hyperparameters such as learning rate.

The code for training the model is located in the eleventh cell of the ipython notebook. 

To train the model, I did not use an optimizer, my batch size was set to 64 and i trained the model for 50 epochs. The learning rate was set to 0.001, with a mu of 0 and a sigma of 0.1.

#### 5. Describe the approach taken for finding a solution. Include in the discussion the results on the training, validation and test sets and where in the code these were calculated. Your approach may have been an iterative process, in which case, outline the steps you took to get to the final solution and why you chose those steps. Perhaps your solution involved an already well known implementation or architecture. In this case, discuss why you think the architecture is suitable for the current problem.

The code for calculating the accuracy of the model is located in the 19th cell of the Ipython notebook.

My final model results were:
* validation set accuracy of 0.948
* test set accuracy of 0.941

I chose LeNet-5 architecture for its straight forwardness and it being well explored and tested for image classification tasks. I ammended the architecture by adding two layers of dropouts to the first and second fully connected layers to enhance the stability of the model.
having a good accuracy of almost 0.95 for some runs I believe this model to be suitable for a first start in image classification. This is also supported by a score of 3 out of 5 new images being recognized by the model.


### Test a Model on New Images

#### 1. Choose five German traffic signs found on the web and provide them in the report. For each image, discuss what quality or qualities might be difficult to classify.

Here are five German traffic signs that I found on the web:
source: Sonja Krause-Harder on the Carnd slack. 

![alt text][image6] 

This image should be easy to identify, as it has a distinct shape and there are almost no shadows or other distortions.

![alt text][image7] 

The pedestrian image might be difficult to classify as it has stickers on it and also there are many other signs that have a similiar shape.

![alt text][image8] 

Same as the image before, this image might be difficult to classify, as it again has the large round shape with some distinguishing content.

![alt text][image9] 

The no Vehicles image should be straight forward to classify, however there are artifacts from other signs present on the image.

![alt text][image10]

In this image the dominant signpost might prove the image difficult to describe.

#### 2. Discuss the model's predictions on these new traffic signs and compare the results to predicting on the test set. Identify where in your code predictions were made. At a minimum, discuss what the predictions were, the accuracy on these new predictions, and compare the accuracy to the accuracy on the test set (OPTIONAL: Discuss the results in more detail as described in the "Stand Out Suggestions" part of the rubric).

The code for making predictions on my final model is located in the 19th cell of the Ipython notebook.




| Image			        |     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| Yield | Yield | 
| Pedestrian | Traffic signals |
| Turn Right | No Passing |
| No Vehicles | No Vehicles |
| Priority Road | Priority Road |


The model was able to correctly guess 3 of the 5 traffic signs, which gives an accuracy of 60%.

#### 3. Describe how certain the model is when predicting on each of the five new images by looking at the softmax probabilities for each prediction and identify where in your code softmax probabilities were outputted. Provide the top 5 softmax probabilities for each image along with the sign type of each probability. (OPTIONAL: as described in the "Stand Out Suggestions" part of the rubric, visualizations can also be provided such as bar charts)

The code for making predictions on my final model is located in the 21st cell of the Ipython notebook.


![alt text][image6]

For the first image, the model is completely sure that this is a yield sign (probability of 1.0), and the image does contain a yield sign. The top five soft max probabilities were

| Probability         	|     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| 1.0         			| Stop sign   									| 
| .0     				| Priority Road 										|
| .0					| Keep right										|
| .0	      			| Speed limit (20km/h)				 				|
| .0				    | Speed limit (30km/h)    							|

-----
![alt text][image7]

For the secound image, the model is relatively sure that this is a traffic signals sign (probability of 0.89), however the image does contain a pedestrian sign. The top five soft max probabilities were

| Probability         	|     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| 0.89         			| Traffic signals 			| 
| .10     				| Go straight 			|
| .003					| Bumpy road			|
| .001	      			| Children crossing 				 	|
| .001	      			| Road narrows				 	|

-----

![alt text][image8]


For the third image, the model is somewhat sure that this is a No passing sign (probability of 0.41), however the image contains a Turn right sign. The top five soft max probabilities were

| Probability         	|     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| .41         			| No Passing   				| 
| .25     				| No Passing for 3.5 Tonnes	|
| .20					| Speed limit (120km/h)		|
| .05	      			| Turn right			 	|
| .03	      			| Speed limit (100km/h)		 	|

-----
![alt text][image9]

For the fourth image, the model is completely sure that this is a No vehicles sign (probability of 1.0), and the image does contain a No Vehicles sign. The top five soft max probabilities were

| Probability         	|     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| 1.0         			| No Vehicles   				| 
| .0     				| No passing	|
| .0					| Yield		|
| .0	      			| Priority road			 	|
| .0	      			| Speed limit (60km/h)		 	|
-----
![alt text][image10]

For the fifth image, the model is completely sure that this is a Priority road sign (probability of 1.0), and the image does contain a Priority road sign. The top five soft max probabilities were

| Probability         	|     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| 1.0         			| Priority road				| 
| .0     				| Yield	|
| .0					| No Passing		|
| .0	      			| Roundabout			 	|
| .0	      			| End of no passing		 	|

