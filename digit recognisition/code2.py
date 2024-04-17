import tensorflow as tf
from tensorflow import keras
from keras import layers,models
import numpy as np
import os
import matplotlib.pyplot as plt
import cv2

print("Welcome to the NeuralNine (c) Handwritten Digits Recognition v0.1")

(X_train, y_train) , (X_test, y_test) = keras.datasets.mnist.load_data()

X_train = X_train / 255
X_test = X_test / 255

X_train = X_train.reshape(-1,28,28,1)
X_test = X_test.reshape(-1,28,28,1)

convolutional_neural_network = models.Sequential([
    layers.Conv2D(filters=25, kernel_size=(3, 3), activation='relu', input_shape=(28,28,1)),
    layers.MaxPooling2D((2, 2)),
    
    layers.Conv2D(filters=64, kernel_size=(3, 3), activation='relu'),
    layers.MaxPooling2D((2, 2)),

    layers.Conv2D(filters=64, kernel_size=(3, 3), activation='relu'),
    layers.MaxPooling2D((2, 2)),
    
    layers.Flatten(),
    layers.Dense(64, activation='relu'),
    layers.Dense(10, activation='softmax')
])

convolutional_neural_network.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
convolutional_neural_network.fit(X_train, y_train, epochs=10)

# convolutional_neural_network.evaluate(X_test, y_test)

# y_predicted_by_model = convolutional_neural_network.predict(X_test)
# y_predicted_by_model[0]

# np.argmax(y_predicted_by_model[0])
# y_predicted_labels = [np.argmax(i) for i in y_predicted_by_model]
# y_predicted_labels[:5]


 # Evaluating the model
val_loss, val_acc = convolutional_neural_network.evaluate(X_test, y_test)
print(val_loss)
print(val_acc)


image_number = 1
while os.path.isfile('digits/digit{}.png'.format(image_number)):
    try:
        img = cv2.imread('digits/digit{}.png'.format(image_number))[:,:,0]
        img = np.invert(np.array([img]))
        prediction = convolutional_neural_network.predict(img)
        print("The number is probably a {}".format(np.argmax(prediction)))
        plt.title("The number is probably a {}".format(np.argmax(prediction)))
        plt.imshow(img[0], cmap=plt.cm.binary)
        plt.show()
        image_number += 1
    except:
        print("Error reading image! Proceeding with next image...")
        image_number += 1