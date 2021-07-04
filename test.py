import tensorflow.keras
from PIL import Image, ImageOps
import numpy as np
import os

# define output
color = ['RED', 'GREEN', 'YELLOW']

# Disable scientific notation for clarity
np.set_printoptions(suppress=True)

# Load the model
model = tensorflow.keras.models.load_model('model.h5')

data = np.ndarray(shape=(1, 64, 64, 3), dtype=np.float32)


def check_color(image_array):
    # Load the image into the array
    data[0] = image_array

    # run the inference
    prediction = model.predict(data)

    index = prediction[0].tolist().index(max(prediction[0]))
    return color[index]


def test(folder_image):
    list_image = [os.path.join(folder_image, f) for f in os.listdir(folder_image)]

    # each face class
    for each_image in list_image:
        size = (64, 64)
        image = Image.open(each_image)
        image_resize = image.resize(size)

        # turn the image into a numpy array
        image_array = np.array(image_resize, 'uint8')

        print(check_color(image_array))



test('data/green')


