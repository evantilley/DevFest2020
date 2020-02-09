import os
from PIL import Image
import numpy as np
from pathlib import Path
from keras_preprocessing import image
from zipfile import ZipFile
from pymongo import MongoClient
from pprint import pprint
from keras.datasets import cifar10

#client = MongoClient('mongodb://localhost:27017/MachineLearning')
#db = client.admin

#serverStatusResult = db.command("serverStatus")
#pprint(serverStatusResult)

class Folder:

    def __init__(self):

        with ZipFile('model.zip', 'r') as zipObj:
            zipObj.extractall()

        training_data_folders = {}
        entries = os.listdir('model/')

        size = 224

        for entry in entries:
            training_data_folders[(Path("model") / entry)] = entry

        images = []
        labels = []

        for folder in training_data_folders:

            for img in folder.glob("*.png"):
                img = Image.open(img)
                wpercent = (size / float(img.size[0]))
                hsize = int((float(img.size[1]) * float(wpercent)))
                img = img.resize((size, hsize), Image.ANTIALIAS)
                img.save(img)
                img = image.load_img(img)
                image_array = image.img_to_array(img)
                images.append(image_array)
                labels.append(training_data_folders[folder])

            for img in folder.glob("*.jpg"):
                img = Image.open(img)
                wpercent = (size / float(img.size[0]))
                hsize = int((float(img.size[1]) * float(wpercent)))
                img = img.resize((size, hsize), Image.ANTIALIAS)
                img.save(img)
                img = image.load_img(img)
                image_array = image.img_to_array(img)
                images.append(image_array)
                labels.append(training_data_folders[folder])

            for img in folder.glob("*.jpg"):
                img = Image.open(img)
                wpercent = (size / float(img.size[0]))
                hsize = int((float(img.size[1]) * float(wpercent)))
                img = img.resize((size, hsize), Image.ANTIALIAS)
                img.save(img)
                img = image.load_img(img)
                image_array = image.img_to_array(img)
                images.append(image_array)
                labels.append(training_data_folders[folder])

        self.x_train = np.arrays(images)
        self.y_train = np.array(labels)

class Dataset:

    def __init__(self):
        ((self.x_train, self.y_train), (self.x_test, self.y_test)) = cifar10.load_data()
        
    #supplies the test and train data in the format
    def build_data(self):

        #folder = Folder()


        return ((self.x_train, self.y_train),(self.x_test, self.y_test))
    #returns the number of categories we are sorting into
    def get_num_classes(self):
        return 10
    #returns a string of the dataset name
    def get_name(self):
        return "cifar10"
    #returns an integer of the proper epoch count for the dataset
    def get_epoch_count(self):
        return 40
