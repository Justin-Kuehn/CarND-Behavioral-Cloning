
# Behavioral Cloning

The purpose of this project is to train a neural network to drive a virtual car.  
This was done as part of the
[Udacity Self Driving Car Nano-Degree]([https://www.udacity.com/course/self-driving-car-engineer-nanodegree--nd013).

![](https://github.com/Justin-Kuehn/CarND-Behavioral-Cloning/blob/master/img/track1.gif)
![](https://github.com/Justin-Kuehn/CarND-Behavioral-Cloning/blob/master/img/track2.gif)

## Implementation

### Creating the Training Data

Creating a good set of training data proved to the hardest challenge in a training a successful model.  Since the track is mostly straight, simply driving around the track repeatable will produce steering angles of mostly zero and almost no right turns.  This imbalance of data will prevent the network from learning how to appropriately handle turns and the car will simply run off the track more often than not.

In order to correct for the Left/Right imbalance I drove the track in reverse several times to create more right turn data and recorded additional passes on each turn going both ways.  I was also able to utilize the left and right side cameras. Since each of the side cameras is simply offset by an amount from the center,  I simply added a slight steering offset in the opposite direction.  I also randomly flipped the image and corresponding steering angle to produce even greater data variation.

All these methods ended up working very well for the first track, which the trained model was able to run lapson indefinitely. However I found that the model would get confused by the shadows present in the second track. In order to fix this, I simulated shadows in the training data by shading random regions in the image.  This produced a model that was much more resilient to shadows and 

Shadow on Track Two:

![Track Shadow](https://github.com/Justin-Kuehn/CarND-Behavioral-Cloning/blob/master/img/shadow.jpg)

Simulated Shadow:

![Simulated Shadow](https://github.com/Justin-Kuehn/CarND-Behavioral-Cloning/blob/master/img/simshadow.png)

### Image Pre-processing


### Network Architecture


## Usage

**To train the model**
```
python model.py --directory <data_directory>
```


**To run the model**

```
python drive.py ./steering_model/model.json
```
