# Patter-Recognition-Lab

MNIST is one of the famous datasets used by members of the AI/ML/Data Science community. They love this dataset and use it as a benchmark to validate their algorithms. Fashion-MNIST is a direct drop-in replacement for the original MNIST dataset. In our project we want to predict real life images through the labels from this trending dataset in Kaggle.

#Our proposed Approach 

We have used Convolutional Neural Network (CNN) over Fashion-MNIST dataset to establish our model. Though this dataset is almost processed, we had to make slight changes or we’d preprocessed for the input to our model. Tensorflow is the state of the art library that we’ve used here. 

(I)	To create our CNN model we have initialized two convolutional layers (Conv2D) and applied a MaxPool (MaxPooling2D) function after each convolutional layer.

(II)	After learning features from the CNN model we’ve flattened it so that we can make an input for the neural network. Multi-dimensional shapes can't be direct inputs to Neural Networks. And that is why we need flattening. Flattening is converting the data into a 1-dimensional array for inputting it to the next layer. We flatten the output of the convolutional layers to create a single long feature vector. 

(III)	This flattened vector is added to the dense function which has got a 64 fully connected dense layer in the model. The rectified linear activation function or ReLU for short is a piecewise linear function that will output the input directly if it is positive, otherwise, it will output zero. That is why we’ve used it in our model.

(IV)	In the similar way we added another fully connected dense layer where the input was from the output of the previous layer and it got output of 32 fully connected dense layers. 

(V)	After adding the same activation function we passed it to the output layer which defined 10 individual labels. We’ve used the softmax activation function for multiclass classification. It gives the probability of each individual label and the maximum probability is defined as the output.

(VI)	As the model requires a lot of time to run and huge computational power, we used a callback method of keras named ModelCheckpoint. 

(VII)	Then we’ve trained the model by fit function and predict the accuracy of the model.

(VIII)	We also have tried to predict the model by comparing a real life example of fashion image.
