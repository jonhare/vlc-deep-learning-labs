# Learning to Deep Learn using Python, Keras, Theano, TensorFlow and a GPU

## Introduction and Acknowledgements
This tutorial is largely based on the Jason Brownlee's ["Handwritten Digit Recognition using Convolutional Neural Networks in Python with Keras"](http://machinelearningmastery.com/handwritten-digit-recognition-using-convolutional-neural-networks-python-keras/). A number of changes have been made to ensure that it better fits our format, and we've added additional bits and exercises.

A popular demonstration of the capability of deep learning techniques is object recognition in image data. The "hello world" of object recognition for machine learning and deep learning is the MNIST dataset for handwritten digit recognition.

In this post you will discover how to develop a deep learning model to achieve near state of the art performance on the MNIST handwritten digit recognition task in Python using the Keras deep learning library.

Through this tutorial you'll learn how to:

* How to load the MNIST dataset in Keras.
* How to develop and evaluate a baseline neural network model for the MNIST problem.
* How to implement and evaluate a simple Convolutional Neural Network for MNIST.
* How to implement a close to state-of-the-art deep learning model for MNIST.
* How to implement advanced network features like branching.
* How to load your own image created outside of the MNIST dataset, and pass it through the network.
* How to visualise the filters learned by the network.

## Prerequisites
To use this tutorial you'll use the Python 2 language with the `keras` deep learning library and the `theano` and `tensorflow` backends. We'll also use the `scikit-learn` and `numpy` packages.

You'll need access to a computer with the following installed:

- `Python` (> 2.6)
- `keras` (>= 1.0.0)
- `theano` (>= 0.8)
- `tensorflow` (>= 0.11)
- `NumPy` (>= 1.6.1)
- `SciPy` (>= 0.9)
- `scikit-learn` (>= 0.17.0)

For the purposes of the doing the tutorial in the lab, we'll provide shell access to a purpose built deep-learning machine with these pre-installed. The machine has an i7 with 4 physical cores (8 with hyperthreading), 32G RAM and a Maxwell-generation nvidia Titan X GPU with 3072 cores and 12G RAM. 

## The MNIST Dataset
MNIST is a dataset developed by Yann LeCun, Corinna Cortes and Christopher Burges for evaluating machine learning models on the handwritten digit classification problem.

The dataset was constructed from a number of scanned document dataset available from the National Institute of Standards and Technology (NIST). This is where the name for the dataset comes from, as the Modified NIST or MNIST dataset.

Images of digits were taken from a variety of scanned documents, normalized in size and centred. This makes it an excellent dataset for evaluating models, allowing the developer to focus on the machine learning with very little data cleaning or preparation required.

Each image is a 28 by 28 pixel square (784 pixels total). A standard spit of the dataset is used to evaluate and compare models, where 60,000 images are used to train a model and a separate set of 10,000 images are used to test it.

It is a digit recognition task. As such there are 10 digits (0 to 9) or 10 classes to predict. Results are reported using prediction error, which is nothing more than the inverted classification accuracy.

Excellent results achieve a prediction error of less than 1%. State-of-the-art prediction error of approximately 0.2% can be achieved with large Convolutional Neural Networks. There is a listing of the state-of-the-art results and links to the relevant papers on the MNIST and other datasets on [Rodrigo Benenson’s webpage](http://rodrigob.github.io/are_we_there_yet/build/classification_datasets_results.html#4d4e495354).

## Loading the MNIST dataset in Keras

The Keras deep learning library provides a convenience method for loading the MNIST dataset.

The dataset is downloaded automatically the first time this function is called and is stored in your home directory in ~/.keras/datasets/mnist.pkl.gz as a 15MB file.

This is very handy for developing and testing deep learning models.

To demonstrate how easy it is to load the MNIST dataset, we will first write a little script to download and visualize the first 4 images in the training dataset.

```python
# Plot ad hoc mnist instances
from keras.datasets import mnist
import matplotlib.pyplot as plt
# load (downloaded if needed) the MNIST dataset
(X_train, y_train), (X_test, y_test) = mnist.load_data()
# plot 4 images as gray scale
plt.subplot(221)
plt.imshow(X_train[0], cmap=plt.get_cmap('gray'))
plt.subplot(222)
plt.imshow(X_train[1], cmap=plt.get_cmap('gray'))
plt.subplot(223)
plt.imshow(X_train[2], cmap=plt.get_cmap('gray'))
plt.subplot(224)
plt.imshow(X_train[3], cmap=plt.get_cmap('gray'))
# show the plot
plt.show(block=False)
```

You can see that downloading and loading the MNIST dataset is as easy as calling the `mnist.load_data()` function. Running the above example, you should see the image below.

![Examples from the MNIST dataset](mnist-samples.png "Examples from the MNIST dataset")

## Baseline Multi-Layer Perceptron Model

Keras is a general purpose neural network toolbox. Before we start to look at deep convolutional architectures, we can start with something much simpler - a basic multilayer perceptron. Because the MNIST images are relatively small, a fully connected MLP network will have relatively few weights to train; with bigger images, an MLP might not be practical due to the number of weights.

In this section we will create a simple multi-layer perceptron model with a single hidden layer that achieves an error rate of 1.74%. We will use this as a baseline for comparing more complex convolutional neural network models later.

Let's start off by importing the classes and functions we will need.

```python
import numpy
from keras.datasets import mnist
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Dropout
from keras.utils import np_utils
```

When developing, it is always a good idea to initialize the random number generator to a constant to ensure that the results of your script are reproducible each time you run it.

```python
# fix random seed for reproducibility
seed = 7
numpy.random.seed(seed)
```
Now we can load the MNIST dataset using the Keras helper function.

```python
# load data
(X_train, y_train), (X_test, y_test) = mnist.load_data()
```

The training dataset is structured as a 3-dimensional array of instance, image width and image height. For a multi-layer perceptron model we must reduce the images down into a vector of pixels. In this case the 28×28 sized images will be 784 pixel input values.

We can do this transform easily using the reshape() function on the NumPy array. We can also reduce our memory requirements by forcing the precision of the pixel values to be 32 bit, the default precision used by Keras anyway.

```python
# flatten 28*28 images to a 784 vector for each image
num_pixels = X_train.shape[1] * X_train.shape[2]
X_train = X_train.reshape(X_train.shape[0], num_pixels).astype('float32')
X_test = X_test.reshape(X_test.shape[0], num_pixels).astype('float32')
```

The pixel values are gray scale between 0 and 255. It is almost always a good idea to perform some scaling of input values when using neural network models. Because the scale is well known and well behaved, we can very quickly normalize the pixel values to the range 0 and 1 by dividing each value by the maximum of 255.

```python
# normalize inputs from 0-255 to 0-1
X_train = X_train / 255
X_test = X_test / 255
```

Finally, the output variable is an integer from 0 to 9. This is a multi-class classification problem. As such, it is good practice to use a one hot encoding of the class values, transforming the vector of class integers into a binary matrix.

We can easily do this using the built-in np_utils.to_categorical() helper function in Keras.

```python
# one hot encode outputs
y_train = np_utils.to_categorical(y_train)
y_test = np_utils.to_categorical(y_test)
num_classes = y_test.shape[1]
```

We are now ready to create our simple neural network model. We will define our model in a function. This is handy if you want to extend the example later and try and get a better score.

```python
# define baseline model
def baseline_model():
	# create model
	model = Sequential()
	model.add(Dense(num_pixels, input_dim=num_pixels, init='normal', activation='relu'))
	model.add(Dense(num_classes, init='normal', activation='softmax'))
	# Compile model
	model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
	return model
```

The model is a simple neural network with one hidden layer with the same number of neurons as there are inputs (784). A rectifier activation function is used for the neurons in the hidden layer.

A softmax activation function is used on the output layer to turn the outputs into probability-like values and allow one class of the 10 to be selected as the model's output prediction. Logarithmic loss is used as the loss function (called categorical_crossentropy in Keras) and the efficient ADAM gradient descent algorithm is used to learn the weights.

We can now fit and evaluate the model. The model is fit over 10 epochs with updates every 200 images. The test data is used as the validation dataset, allowing you to see the skill of the model as it trains. A verbose value of 2 is used to reduce the output to one line for each training epoch.

Finally, the test dataset is used to evaluate the model and a classification error rate is printed.

```python
# build the model
model = baseline_model()
# Fit the model
model.fit(X_train, y_train, validation_data=(X_test, y_test), nb_epoch=10, batch_size=200, verbose=2)
# Final evaluation of the model
scores = model.evaluate(X_test, y_test, verbose=0)
print("Baseline Error: %.2f%%" % (100-scores[1]*100))
```

Running the example might take a few minutes when run on a CPU. You should see the output below. This very simple network defined in very few lines of code achieves a respectable error rate of 1.74%.

	Train on 60000 samples, validate on 10000 samples
	Epoch 1/10
	11s - loss: 0.2791 - acc: 0.9203 - val_loss: 0.1422 - val_acc: 0.9583
	Epoch 2/10
	11s - loss: 0.1121 - acc: 0.9680 - val_loss: 0.0994 - val_acc: 0.9697
	Epoch 3/10
	12s - loss: 0.0724 - acc: 0.9790 - val_loss: 0.0786 - val_acc: 0.9748
	Epoch 4/10
	12s - loss: 0.0508 - acc: 0.9856 - val_loss: 0.0790 - val_acc: 0.9762
	Epoch 5/10
	12s - loss: 0.0365 - acc: 0.9897 - val_loss: 0.0631 - val_acc: 0.9795
	Epoch 6/10
	12s - loss: 0.0263 - acc: 0.9931 - val_loss: 0.0644 - val_acc: 0.9798
	Epoch 7/10
	12s - loss: 0.0188 - acc: 0.9956 - val_loss: 0.0613 - val_acc: 0.9803
	Epoch 8/10
	12s - loss: 0.0149 - acc: 0.9967 - val_loss: 0.0628 - val_acc: 0.9814
	Epoch 9/10
	12s - loss: 0.0108 - acc: 0.9980 - val_loss: 0.0595 - val_acc: 0.9816
	Epoch 10/10
	12s - loss: 0.0072 - acc: 0.9989 - val_loss: 0.0577 - val_acc: 0.9826
	Baseline Error: 1.74%

## Simple Convolutional Neural Network for MNIST

Now that we have seen how to load the MNIST dataset and train a simple multi-layer perceptron model on it, it is time to develop a more sophisticated convolutional neural network or CNN model.

Keras does provide a lot of capability for creating convolutional neural networks.

In this section we will create a simple CNN for MNIST that demonstrates how to use all of the aspects of a modern CNN implementation, including Convolutional layers, Pooling layers and Dropout layers.

The first step is to import the classes and functions needed.

```python
import numpy
from keras.datasets import mnist
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Dropout
from keras.layers import Flatten
from keras.layers.convolutional import Convolution2D
from keras.layers.convolutional import MaxPooling2D
from keras.utils import np_utils
from keras import backend as K
K.set_image_dim_ordering('th')
```

Again, we always initialize the random number generator to a constant seed value for reproducibility of results.

```python
# fix random seed for reproducibility
seed = 7
numpy.random.seed(seed)
```

Next we need to load the MNIST dataset and reshape it so that it is suitable for use training a CNN. In Keras, the layers used for two-dimensional convolutions expect pixel values with the dimensions [pixels][width][height].

In the case of RGB, the first dimension pixels would be 3 for the red, green and blue components and it would be like having 3 image inputs for every color image. In the case of MNIST where the pixel values are gray scale, the pixel dimension is set to 1.


```python
# load data
(X_train, y_train), (X_test, y_test) = mnist.load_data()
# reshape to be [samples][pixels][width][height]
X_train = X_train.reshape(X_train.shape[0], 1, 28, 28).astype('float32')
X_test = X_test.reshape(X_test.shape[0], 1, 28, 28).astype('float32')
```

As before, it is a good idea to normalize the pixel values to the range 0 and 1 and one hot encode the output variables.

```python
# normalize inputs from 0-255 to 0-1
X_train = X_train / 255
X_test = X_test / 255
# one hot encode outputs
y_train = np_utils.to_categorical(y_train)
y_test = np_utils.to_categorical(y_test)
num_classes = y_test.shape[1]
```

Next we define our neural network model.

Convolutional neural networks are more complex than standard multi-layer perceptrons, so we will start by using a simple structure to begin with that uses all of the elements for state of the art results. Below summarizes the network architecture.

1 The first hidden layer is a convolutional layer called a Convolution2D. The layer has 32 feature maps, which with the size of 5×5 and a rectifier activation function. This is the input layer, expecting images with the structure outline above [pixels][width][height].
2 Next we define a pooling layer that takes the max called MaxPooling2D. It is configured with a pool size of 2×2.
3 The next layer is a regularization layer using dropout called Dropout. It is configured to randomly exclude 20% of neurons in the layer in order to reduce overfitting.
4 Next is a layer that converts the 2D matrix data to a vector called Flatten. It allows the output to be processed by standard fully connected layers.
5 Next a fully connected layer with 128 neurons and rectifier activation function.
6 Finally, the output layer has 10 neurons for the 10 classes and a softmax activation function to output probability-like predictions for each class.
As before, the model is trained using logarithmic loss and the ADAM gradient descent algorithm.

```python
def baseline_model():
	# create model
	model = Sequential()
	model.add(Convolution2D(32, 5, 5, border_mode='valid', input_shape=(1, 28, 28), activation='relu'))
	model.add(MaxPooling2D(pool_size=(2, 2)))
	model.add(Dropout(0.2))
	model.add(Flatten())
	model.add(Dense(128, activation='relu'))
	model.add(Dense(num_classes, activation='softmax'))
	# Compile model
	model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
	return model
```

We evaluate the model the same way as before with the multi-layer perceptron. The CNN is fit over 10 epochs with a batch size of 200.

```python
# build the model
model = baseline_model()
# Fit the model
model.fit(X_train, y_train, validation_data=(X_test, y_test), nb_epoch=10, batch_size=200, verbose=2)
# Final evaluation of the model
scores = model.evaluate(X_test, y_test, verbose=0)
print("Baseline Error: %.2f%%" % (100-scores[1]*100))
```

Running the example, the accuracy on the training and validation test is printed each epoch and at the end of the classification error rate is printed.

Epochs may take 60 to 90 seconds to run on the CPU, or about 15 minutes in total depending on your hardware. You can see that the network achieves an error rate of 1.10, which is better than our simple multi-layer perceptron model above.

	Train on 60000 samples, validate on 10000 samples
	Epoch 1/10
	84s - loss: 0.2065 - acc: 0.9370 - val_loss: 0.0759 - val_acc: 0.9756
	Epoch 2/10
	84s - loss: 0.0644 - acc: 0.9802 - val_loss: 0.0475 - val_acc: 0.9837
	Epoch 3/10
	89s - loss: 0.0447 - acc: 0.9864 - val_loss: 0.0402 - val_acc: 0.9877
	Epoch 4/10
	88s - loss: 0.0346 - acc: 0.9891 - val_loss: 0.0358 - val_acc: 0.9881
	Epoch 5/10
	89s - loss: 0.0271 - acc: 0.9913 - val_loss: 0.0342 - val_acc: 0.9891
	Epoch 6/10
	89s - loss: 0.0210 - acc: 0.9933 - val_loss: 0.0391 - val_acc: 0.9880
	Epoch 7/10
	89s - loss: 0.0182 - acc: 0.9943 - val_loss: 0.0345 - val_acc: 0.9887
	Epoch 8/10
	89s - loss: 0.0142 - acc: 0.9956 - val_loss: 0.0323 - val_acc: 0.9904
	Epoch 9/10
	88s - loss: 0.0120 - acc: 0.9961 - val_loss: 0.0343 - val_acc: 0.9901
	Epoch 10/10
	89s - loss: 0.0108 - acc: 0.9965 - val_loss: 0.0353 - val_acc: 0.9890
	Classification Error: 1.10%

## Larger Convolutional Neural Network for MNIST

Now that we have seen how to create a simple CNN, let’s take a look at a model capable of close to state of the art results.

We import classes and function then load and prepare the data the same as in the previous CNN example.

```python
import numpy
from keras.datasets import mnist
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Dropout
from keras.layers import Flatten
from keras.layers.convolutional import Convolution2D
from keras.layers.convolutional import MaxPooling2D
from keras.utils import np_utils
from keras import backend as K
K.set_image_dim_ordering('th')
 
# fix random seed for reproducibility
seed = 7
numpy.random.seed(seed)
 
# load data
(X_train, y_train), (X_test, y_test) = mnist.load_data()
# reshape to be [samples][pixels][width][height]
X_train = X_train.reshape(X_train.shape[0], 1, 28, 28).astype('float32')
X_test = X_test.reshape(X_test.shape[0], 1, 28, 28).astype('float32')
 
# normalize inputs from 0-255 to 0-1
X_train = X_train / 255
X_test = X_test / 255
# one hot encode outputs
y_train = np_utils.to_categorical(y_train)
y_test = np_utils.to_categorical(y_test)
num_classes = y_test.shape[1]
```

This time we define a large CNN architecture with additional convolutional, max pooling layers and fully connected layers. The network topology can be summarized as follows.

1 Convolutional layer with 30 feature maps of size 5×5.
2 Pooling layer taking the max over 2*2 patches.
3 Convolutional layer with 15 feature maps of size 3×3.
4 Pooling layer taking the max over 2*2 patches.
5 Dropout layer with a probability of 20%.
6 Flatten layer.
7 Fully connected layer with 128 neurons and rectifier activation.
8 Fully connected layer with 50 neurons and rectifier activation.
9 Output layer.

```python
def larger_model():
	# create model
	model = Sequential()
	model.add(Convolution2D(30, 5, 5, border_mode='valid', input_shape=(1, 28, 28), activation='relu'))
	model.add(MaxPooling2D(pool_size=(2, 2)))
	model.add(Convolution2D(15, 3, 3, activation='relu'))
	model.add(MaxPooling2D(pool_size=(2, 2)))
	model.add(Dropout(0.2))
	model.add(Flatten())
	model.add(Dense(128, activation='relu'))
	model.add(Dense(50, activation='relu'))
	model.add(Dense(num_classes, activation='softmax'))
	# Compile model
	model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
	return model
```

Like the previous two experiments, the model is fit over 10 epochs with a batch size of 200.

```python
# build the model
model = larger_model()
# Fit the model
model.fit(X_train, y_train, validation_data=(X_test, y_test), nb_epoch=10, batch_size=200, verbose=2)
# Final evaluation of the model
scores = model.evaluate(X_test, y_test, verbose=0)
print("Baseline Error: %.2f%%" % (100-scores[1]*100))
```

Running the example prints accuracy on the training and validation datasets each epoch and a final classification error rate.

The model takes about 100 seconds to run per epoch. This slightly larger model achieves the respectable classification error rate of 0.89%.

	Using Theano backend.
	Train on 60000 samples, validate on 10000 samples
	Epoch 1/10
	102s - loss: 0.3263 - acc: 0.8962 - val_loss: 0.0690 - val_acc: 0.9785
	Epoch 2/10
	103s - loss: 0.0858 - acc: 0.9737 - val_loss: 0.0430 - val_acc: 0.9862
	Epoch 3/10
	102s - loss: 0.0627 - acc: 0.9806 - val_loss: 0.0379 - val_acc: 0.9875
	Epoch 4/10
	101s - loss: 0.0501 - acc: 0.9842 - val_loss: 0.0342 - val_acc: 0.9891
	Epoch 5/10
	102s - loss: 0.0444 - acc: 0.9856 - val_loss: 0.0338 - val_acc: 0.9889
	Epoch 6/10
	101s - loss: 0.0389 - acc: 0.9878 - val_loss: 0.0302 - val_acc: 0.9897
	Epoch 7/10
	101s - loss: 0.0335 - acc: 0.9894 - val_loss: 0.0260 - val_acc: 0.9916
	Epoch 8/10
	102s - loss: 0.0305 - acc: 0.9898 - val_loss: 0.0267 - val_acc: 0.9911
	Epoch 9/10
	101s - loss: 0.0296 - acc: 0.9904 - val_loss: 0.0211 - val_acc: 0.9933
	Epoch 10/10
	102s - loss: 0.0272 - acc: 0.9911 - val_loss: 0.0269 - val_acc: 0.9911
	Classification Error: 0.89%

This is not an optimized network topology. Nor is a reproduction of a network topology from a recent paper. There is a lot of opportunity for you to tune and improve upon this model.

What is the best error rate score you can achieve?

