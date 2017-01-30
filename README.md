
# Behavioral Cloning

The purpose of this project is to train a neural network to drive a virtual car.  
This was done as part of the
[Udacity Self Driving Car Nano-Degree]([https://www.udacity.com/course/self-driving-car-engineer-nanodegree--nd013).

![](https://github.com/Justin-Kuehn/CarND-Behavioral-Cloning/blob/master/img/track1.gif)
![](https://github.com/Justin-Kuehn/CarND-Behavioral-Cloning/blob/master/img/track2.gif)

## Usage

**To train the model**
```
python model.py --directory <data_directory>
```

**To run the model**

```
python drive.py ./steering_model/model.json
```

## Implementation

### Creating the Training Data

Creating a good set of training data proved to the hardest challenge in a training a successful model.  Since the track is mostly straight, simply driving around the track repeatable will produce steering angles of mostly zero and almost no right turns.  This imbalance of data will prevent the network from learning how to appropriately handle turns and the car will simply run off the track more often than not.

In order to correct for the Left/Right imbalance I drove the track in reverse several times to create more right turn data and recorded additional passes on each turn going both ways.  I was also able to utilize the left and right side cameras. Since each of the side cameras is simply offset by an amount from the center,  I simply added a slight steering offset in the opposite direction.  I also randomly flipped the image and corresponding steering angle to produce even greater data variation.

All these methods ended up working very well for the first track, which the trained model was able to run laps on indefinitely. However I found that the model would get confused by the shadows present in the second track. In order to fix this, I simulated shadows in the training data by shading random regions in the image.  This produced a model that was much more resilient to shadows. 

Shadow on Track Two:

![Track Shadow](https://github.com/Justin-Kuehn/CarND-Behavioral-Cloning/blob/master/img/shadow.jpg)

Simulated Shadow:

![Simulated Shadow](https://github.com/Justin-Kuehn/CarND-Behavioral-Cloning/blob/master/img/simshadow.png)


### Image Pre-processing

I found that using the raw images collected by the simulator did not produce very good results, so employed several pre-processing methods.

First I boost the gamma ranges in the image using the technique described here: 
http://www.pyimagesearch.com/2015/10/05/opencv-gamma-correction/  
I found that doing this helped significantly on the second track where there were many shadows occluding the track.  

Next I clipped the top and bottom of the image to ensure that most of the image was of the road ahead and not background objects like trees or the sky.  This helps prevent outfitting by forcing the model to learn how to follow the contours of the roads instead of using track specific landmarks.

Finally I resize the image to 64x64 pixels to speed up training and then normalized the values to lie between -1 and 1.

### Network Architecture

The network used was inspired by the comma.ai steering model found here: https://github.com/commaai/research/blob/master/train_steering_model.py

The model contains three convolution layers each followed by a ELU activation function.  ELU's were used over RELU's for faster training. There is one fully connected layer with 512 nodes followed by a single output node.  Dropout are added between the last convolution and the output to help prevent over-fitting.

![Simulated Shadow](https://github.com/Justin-Kuehn/CarND-Behavioral-Cloning/blob/master/img/model.png)

 * Input: 3x64x64
 * Layer One: Convolution 8x8x16 with ELU activation
 * Layer Two: Convolution 5x5x32 with ELU activation
 * Layer Three: Convolution 5x5x64 with ELU activation and dropout of 20%
 * Flatten
 * Layer Four: Fully connected with 512 nodes with ELU activation and dropout of 50%
 * Layer Five: Fully connected with one output node

### Training

I preformed a 90/10 split for training and validation data respectively to help prevent over-fitting the model.  An ADAM optimizer was used with a learning rate of 0.0001.  I used a lower learning rate because I found that the the default learning rate of 0.001 stopped learning too quickly and MSE leveled out after only the first epoch.

Batches of 128 images were sampled randomly from the training set.  I trained only for four epochs each with 33093 total samples with a final MSE of 0.453 on the validation set.  I found that training more than four epochs continued to lower the MSE (mean squared error) of the validation set, however the real performance on both track one and two began to degrade sharply with the car often veering off the road. This indicated that MSE alone is a poor criteria to measure model fitness, in the end I stopped training well before the MSE leveled off. 


## Results

**Track One:**

[![](https://i.ytimg.com/vi/vo0urfZqfn4/hqdefault.jpg)](https://www.youtube.com/watch?v=vo0urfZqfn4)

**Track Two:**

[![](https://i.ytimg.com/vi/iikTwr_XE8E/hqdefault.jpg)](https://www.youtube.com/watch?v=iikTwr_XE8E)


