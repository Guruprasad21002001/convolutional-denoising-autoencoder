# Convolutional Autoencoder for Image Denoising

## AIM

To develop a convolutional autoencoder for image denoising application.

## Problem Statement and Dataset

![data set](https://user-images.githubusercontent.com/95342910/202714065-6e699e32-a5ee-4db1-9a9f-d920b115f481.png)

## Convolution Autoencoder Network Model

![nnm-1](https://user-images.githubusercontent.com/95342910/202713886-62897e7f-633a-48b0-b7ce-1acd4c36c7b4.png)

![nnm-2](https://user-images.githubusercontent.com/95342910/202713965-f4518e3f-32d5-4791-8616-f19bf996816a.png)

## DESIGN STEPS

## STEP 1:
Import the necessary libraries and download the mnist dataset.

## STEP 2:
Load the dataset and scale the values for easier computation.

## STEP 3:
Add noise to the images randomly for the process of denoising it with the convolutional denoising autoencoders for both the training and testing sets.

## STEP 4:
Build the Neural Model for convolutional denoising autoencoders using Convolutional, Pooling and Up Sampling layers. Make sure the input shape and output shape of the model are identical.

## STEP 5:
Pass test data for validating manually. Compile and fit the created model.

## STEP 6:
Plot the Original, Noisy and Reconstructed Image predictions for visualization.

## STEP 7:
End the program.

Write your own steps

## PROGRAM
~~~
### Developed by : GURU PRASAD.B
### Reg.No : 212221230032
Program to develop a convolutional autoencoder for image denoising application.

from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras import utils
from tensorflow.keras import models
from tensorflow.keras.datasets import mnist
import numpy as np
import matplotlib.pyplot as plt

(x_train, _), (x_test, _) = mnist.load_data()

x_train.shape

x_train_scaled = x_train.astype('float32') / 255.
x_test_scaled = x_test.astype('float32') / 255.
x_train_scaled = np.reshape(x_train_scaled, (len(x_train_scaled), 28, 28, 1))
x_test_scaled = np.reshape(x_test_scaled, (len(x_test_scaled), 28, 28, 1))

noise_factor = 0.8
x_train_noisy = x_train_scaled + noise_factor * np.random.normal(loc=0.0, scale=1.0, size=x_train_scaled.shape) 
x_test_noisy = x_test_scaled + noise_factor * np.random.normal(loc=0.0, scale=1.0, size=x_test_scaled.shape) 
x_train_noisy = np.clip(x_train_noisy, 0., 1.)
x_test_noisy = np.clip(x_test_noisy, 0., 1.)

n = 10
plt.figure(figsize=(20, 2))
for i in range(1, n + 1):
    ax = plt.subplot(1, n, i)
    plt.imshow(x_test_noisy[i].reshape(28, 28))
    plt.gray()
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)
plt.show()

input_img = keras.Input(shape=(28, 28, 1))

x = layers.Conv2D(32, (3, 3),activation = 'relu', padding='same')(input_img)
x = layers.MaxPooling2D((2, 2), padding='same')(x)
x = layers.Conv2D(32, (3, 3), activation = 'relu', padding='same')(x)
encoded = layers.MaxPooling2D((2, 2), padding='same')(x)

x = layers.Conv2D(32, (3, 3), activation = 'relu', padding='same')(encoded)
x = layers.UpSampling2D((2, 2))(x)
x = layers.Conv2D(32, (3, 3), activation = 'relu', padding='same')(x)
x = layers.UpSampling2D((2, 2))(x)
decoded = layers.Conv2D(1, (3, 3), activation='sigmoid', padding='same')(x)

autoencoder = keras.Model(input_img, decoded)
autoencoder.summary()

autoencoder.compile(optimizer='adam', loss='binary_crossentropy')

autoencoder.fit(x_train_noisy, x_train_scaled,
                epochs=3,
                batch_size=128,
                shuffle=True,
                validation_data=(x_test_noisy, x_test_scaled))
                
decoded_imgs = autoencoder.predict(x_test_noisy)

n = 10
plt.figure(figsize=(20, 4))
for i in range(1, n + 1):
    # Display original:
    ax = plt.subplot(3, n, i)
    plt.imshow(x_test_scaled[i].reshape(28, 28))
    plt.gray()
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)

    # Display noisy:
    ax = plt.subplot(3, n, i+n)
    plt.imshow(x_test_noisy[i].reshape(28, 28))
    plt.gray()
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)    

    # Display reconstruction:
    ax = plt.subplot(3, n, i + 2*n)
    plt.imshow(decoded_imgs[i].reshape(28, 28))
    plt.gray()
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)
    
plt.show()
~~~

## OUTPUT
### ADDING NOISE TO THE MNIST DATASET:

![op1](https://user-images.githubusercontent.com/95342910/202713454-e69e3796-55f4-4d26-aaa3-916eeabe0fb6.png)

### AUTOENCODER.SUMMARY():

![op2](https://user-images.githubusercontent.com/95342910/202713473-60ccea5a-eeac-4004-8814-fa680de9adef.png)

### ORIGINAL V/S NOISY V/S RECONSTRUCTED IMAGE:

![op3](https://user-images.githubusercontent.com/95342910/202713504-7f7d48fb-03d0-4fac-9e29-d80d27298add.png)

## RESULT
Thus, the program to develop a convolutional autoencoder for image denoising application is developed and executted successfully.
