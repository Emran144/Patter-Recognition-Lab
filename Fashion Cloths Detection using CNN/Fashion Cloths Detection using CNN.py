# -*- coding: utf-8 -*-
"""FMNIST : : Version-3(Experimental).ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/1sYhMVIuHlfX6DRySzzovdj3P0Uq37Azn
"""

import tensorflow as tf

fashion_mnist = tf.keras.datasets.fashion_mnist
# mnist = tf.keras.datasets.mnist

(x_train, y_train), (x_test, y_test) = fashion_mnist.load_data()
# (x_train, y_train), (x_test, y_test) = mnist.load_data()

print(x_train.shape)
print(y_train.shape)
print(x_test.shape)
print(y_test.shape)

import matplotlib.pyplot as plt

plt.imshow(x_train[0])
plt.show()

plt.imshow(x_train[0], cmap=plt.get_cmap('gray'))

print(x_train[0])

x_train = x_train/255
x_test = x_test/255
plt.imshow(x_train[0], cmap=plt.get_cmap('gray'))

print(x_test[0])

print(y_train[1], y_test[1])

import numpy as np

x_trainr = np.array(x_train).reshape(60000, 28, 28, 1)
x_testr = np.array(x_test).reshape(10000, 28, 28, 1)

print("Training sample dimention: ", x_trainr.shape)
print("Testing sample dimention: ", x_testr.shape)

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Activation, Flatten, Conv2D, MaxPooling2D, Dropout



"""# Model Create"""

from tensorflow.keras.callbacks import ModelCheckpoint
fmnist_model_checkpoint = ModelCheckpoint('FMNIST_Weight.h5',
                                   save_best_only=True,
                                   monitor='loss',
                                   verbose=1                            
                                   )

model = Sequential()

# 1st Layer
model.add(Conv2D(64, (3,3), input_shape = (28,28,1), activation='relu'))
model.add(MaxPooling2D(pool_size=(2,2)))

# model.add(Dropout(rate=0.1))
# 2nd Layer
# model.add(Conv2D(32, (3,3), activation='relu'))
# model.add(MaxPooling2D(pool_size=(2,2)))

model.add(Dropout(rate=0.1))
# 3rd Layer
model.add(Conv2D(16, (3,3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2,2)))

model.add(Flatten())

model.add(Dense(64))
model.add(Activation('relu'))

model.add(Dropout(rate=0.1))

model.add(Dense(32))
model.add(Activation('relu'))

model.add(Dense(10))
model.add(Activation('softmax'))

model.summary()

from keras.utils.vis_utils import plot_model
plot_model(model, to_file='model.png', show_shapes=True, show_layer_names=True)

print("Total training sample: ", len(x_trainr))

model.compile(optimizer="adam", loss='sparse_categorical_crossentropy', metrics=['accuracy'])

history = model.fit(x_trainr, y_train, batch_size=512, epochs=100, validation_split = 0.3, callbacks=[fmnist_model_checkpoint])

import os.path
if os.path.isfile("/content/FMNIST_Weight.h5") is False:
    model.save_weights("/content/FMNIST_Weight.h5")

test_loss, test_acc = model.evaluate(x_testr, y_test)
print("Loss: ", test_loss)
print("Accuracy: ", test_acc)

predictions = model.predict([x_testr])

print(predictions)

pred = np.argmax(predictions[1])

print(pred)

plt.imshow(x_test[101])

image_class = {}
image_class[0] = 'T-shirt/top'
image_class[1] = 'Trouser'
image_class[2] = 'Pullover'
image_class[3] = 'Dress'
image_class[4] = 'Coat'
image_class[5] = 'Sandal'
image_class[6] = 'Shirt'
image_class[7] = 'Sneaker'
image_class[8] = 'Bag'
image_class[9] = 'Ankle boot'

from PIL import Image

# for i in range(10):
demo_image = "/content/drive/MyDrive/University/12th Semester/CSI 416 [Pattern Recognition Lab]/Project/raw_images/fashion/bag.jpg"
img = Image.open(demo_image)


img = img.resize((28, 28))
imgGray = img.convert('L')
imgGray.save('test_gray.jpg')

image_array = np.array(imgGray)

plt.imshow(imgGray, cmap=plt.get_cmap('gray'))
plt.show()

image_array = image_array/255

new_img = np.array(image_array).reshape(1, 28, 28, 1)    # reshape for kerner operation

test_pred = model.predict(new_img)

predict_class = np.argmax(test_pred)
print("Prediction: ", image_class[predict_class])

"""```
0 T-shirt/top
1 Trouser
2 Pullover
3 Dress
4 Coat
5 Sandal
6 Shirt
7 Sneaker
8 Bag
9 Ankle boot
```
"""

print(len(test_pred[0]))
for i in range(10):
    print(f"Similarity with {image_class[i]} is --> [{round(test_pred[0][i]*100, 4)} %]")

import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
fashion_mnist = tf.keras.datasets.fashion_mnist
(x_train, y_train), (x_test, y_test) = fashion_mnist.load_data()
x_train = x_train/255
x_test = x_test/255
x_trainr = np.array(x_train).reshape(60000, 28, 28, 1)
x_testr = np.array(x_test).reshape(10000, 28, 28, 1)

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Activation, Flatten, Conv2D, MaxPooling2D, Dropout

"""# Use Model Explicitely"""

model2 = Sequential()

# 1st Layer
model2.add(Conv2D(64, (3,3), input_shape = (28,28,1), activation='relu'))
model2.add(MaxPooling2D(pool_size=(2,2)))
# 2nd Layer
# model2.add(Conv2D(32, (3,3), activation='relu'))
# model2.add(MaxPooling2D(pool_size=(2,2)))
model2.add(Dropout(rate=0.1))
# 3rd Layer
model2.add(Conv2D(16, (3,3), activation='relu'))
model2.add(MaxPooling2D(pool_size=(2,2)))

model2.add(Flatten())

model2.add(Dense(64))
model2.add(Activation('relu'))

model2.add(Dropout(rate=0.1))

model2.add(Dense(32))
model2.add(Activation('relu'))

model2.add(Dense(10))
model2.add(Activation('softmax'))

model2.load_weights("/content/FMNIST_Weight.h5")
# model2.get_weights()

model2.compile(optimizer="adam", loss='sparse_categorical_crossentropy', metrics=['accuracy'])

test2_loss, test2_acc = model2.evaluate(x_testr, y_test)
print("Loss: ", test2_loss)
print("Accuracy: ", test2_acc)

predictions2 = model2.predict([x_testr])

print(predictions2)

pred2 = np.argmax(predictions2[101])
print(pred2)

image_class = {}
image_class[0] = 'T-shirt/top'
image_class[1] = 'Trouser'
image_class[2] = 'Pullover'
image_class[3] = 'Dress'
image_class[4] = 'Coat'
image_class[5] = 'Sandal'
image_class[6] = 'Shirt'
image_class[7] = 'Sneaker'
image_class[8] = 'Bag'
image_class[9] = 'Ankle boot'

from PIL import Image

demo_image2 = "/content/drive/MyDrive/University/12th Semester/CSI 416 [Pattern Recognition Lab]/Project/raw_images/fashion/shirt_1.jpg"
img2 = Image.open(demo_image2)


plt.imshow(img2)
plt.show()

img2 = img2.resize((28, 28))
imgGray2 = img2.convert('L')
imgGray2.save('test_gray2.jpg')

plt.imshow(imgGray2)
plt.show()

image_array2 = np.array(imgGray2)

plt.imshow(imgGray2, cmap=plt.get_cmap('gray'))
plt.show()

image_array2 = image_array2/255

new_img2 = np.array(image_array2).reshape(1, 28, 28, 1)    # reshape for kerner operation

test_pred2 = model2.predict(new_img2)

label = np.argmax(test_pred2)

print("Predict: ", image_class[label])

"""

> **Class Label**


```
0 T-shirt/top
1 Trouser
2 Pullover
3 Dress
4 Coat
5 Sandal
6 Shirt
7 Sneaker
8 Bag
9 Ankle boot
```"""

print(len(test_pred2[0]))
for i in range(10):
    print(f"Similarity with {image_class[i]} is --> [{round(test_pred2[0][i]*100, 4)} %]")

# history.history.keys()

# summarize history for accuracy
plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()

# summarize history for loss
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()