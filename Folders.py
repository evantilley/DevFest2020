from pathlib import Path
import os
from keras_preprocessing import image

class Folders:

    def __init__(self):
        #supplies the test and train data in the format
        training_data_folders = []
        training_data_types = []
        entries = os.listdir('model/')

        for entry in entries:
            training_data_folders.append(Path("model") / entry)
            training_data_types.append(entry)

        images = []
        labels = []

        for folder in training_data_folders:

            for img in folder.glob("*.png"):
                img = image.load_img(img)
                image_array = image.img_to_array(img)
                images.append(image_array)
                labels.append()

            for img in folder.glob("*.png"):
                img = image.load_img(img)
                image_array = image.img_to_array(img)
                images.append(image_array)
                labels.append()



