import numpy as np
import os
import pandas as pd
from PIL import Image, ImageOps
from matplotlib import pyplot as plt
from sklearn.model_selection import train_test_split
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras import layers

size_img_1 = 64
size_img_2 = 64
output_shape = 3

# define output
color = ['RED', 'GREEN', 'YELLOW']


def get_index_color(color_name):
    if color_name == 'green':
        return 1
    if color_name == 'red':
        return 0
    if color_name == 'yellow':
        return 2


def get_one_hot_vector(length, index):
    vector = np.zeros(length)
    vector[index] = 1

    return vector


# get matrix and label for training model from folder
def get_data(path_data):
    images = []
    labels = []
    fix_size_img = (size_img_1, size_img_2)
    image_paths = [os.path.join(path_data, f) for f in os.listdir(path_data)]

    for image_path in image_paths:
        list_image = [os.path.join(image_path, f) for f in os.listdir(image_path)]

        color_fruit = os.path.basename(image_path)
        index = get_index_color(color_fruit)
        label = get_one_hot_vector(output_shape, index)

        # each image class
        for each_image in list_image:
            fruit_image = Image.open(each_image)
            fruit_image = fruit_image.resize(fix_size_img)

            image_numpy = np.array(fruit_image, 'uint8')

            images.append(image_numpy)
            labels.append(label)

    return np.array(images), np.array(labels)


def build_network(shape, size_output):
    # Convolutional Neural Network
    network = Sequential()

    network.add(layers.Flatten(input_shape=shape))
    network.add(layers.BatchNormalization())
    network.add(layers.Dense(units=256, activation='relu'))
    network.add(layers.Dropout(0.3))
    network.add(layers.Dense(units=64, activation='relu'))
    network.add(layers.Dropout(0.3))
    network.add(layers.Dense(units=size_output, activation='softmax'))

    network.summary()

    return network


# get data
matrix, labels = get_data('data/')
X_train, X_test, y_train, y_test = train_test_split(matrix, labels, test_size=0.1, random_state=42)

# ===================================================================================
# training
input_shape = (size_img_1, size_img_2, 3)

model = build_network(input_shape, output_shape)
model.compile(optimizer='adam',
              loss=tf.keras.losses.categorical_crossentropy,
              metrics=['accuracy'])
history = model.fit(X_train, y_train,
                    epochs=100)

# plot the acc and loss
pd.DataFrame(history.history).plot(figsize=(8, 5))
plt.show()

# save model
model.save('model.h5')

# ====================================================================================
# Test model
model = tf.keras.models.load_model('model.h5')

# test for test set
predict = model.predict(X_test)
count = 0
for i in range(len(X_test)):
    print(y_test[i], '====', predict[i])
    if y_test[i].tolist().index(max(y_test[i])) == predict[i].tolist().index(max(predict[i])):
        count += 1
print(count, len(predict))

# test for training set
count = 0
predict = model.predict(X_train)
for i in range(len(X_train)):
    if y_train[i].tolist().index(max(y_train[i])) == predict[i].tolist().index(max(predict[i])):
        count += 1

print(count, len(predict))