import tensorflow.keras
from PIL import Image, ImageOps
import numpy as np
import cv2  # using opencv

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


def realtime_cam():
    webcam = cv2.VideoCapture(0)

    while True:
        ret, image = webcam.read()
        # show the images
        cv2.waitKey(1)

        # convert numpy to image to process
        PIL_image = Image.fromarray(image.astype('uint8'), 'RGB')
        print(type(PIL_image))

        size = (64, 64)
        image_resize = PIL_image.resize(size)

        # turn the image into a numpy array
        image_array = np.array(image_resize, 'uint8')

        color_predict = check_color(image_array)

        # =============================
        # In here, can process another work with color output
        print(color_predict)
        # ==============================

        cv2.putText(image, color_predict, (100, 100), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2, cv2.LINE_AA)
        cv2.imshow("images", image)

        if cv2.waitKey(10) & 0xFF == ord('q'):  # close the camera with press 'q'
            break

    # close the camera
    webcam.release()
    cv2.destroyAllWindows()


realtime_cam()
