import os
import numpy as np
from pathlib import Path
from keras_preprocessing import image
from zipfile import ZipFile

class Folders:

    def __init__(self):
        #supplies the test and train data in the format
        with ZipFile('model.zip', 'r') as zipObj:
            zipObj.extractall()

        training_data_folders = {}
        entries = os.listdir('model/')

        for entry in entries:
            training_data_folders[(Path("model") / entry)] = entry

        images = []
        labels = []

        for folder in training_data_folders:

            for img in folder.glob("*.png"):
                img = image.load_img(img)
                image_array = image.img_to_array(img)
                images.append(image_array)
                labels.append(training_data_folders[folder])

            for img in folder.glob("*.png"):
                img = image.load_img(img)
                image_array = image.img_to_array(img)
                images.append(image_array)
                labels.append(training_data_folders[folder])

        x_train = np.arrays(images)
        y_train = np.array(labels)



