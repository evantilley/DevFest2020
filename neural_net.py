import numpy as np
import joblib
from keras.preprocessing import image
from keras.applications import nasnet
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten
from pathlib import Path
from keras.models import model_from_json
from keras.applications import vgg16

class_labels = [
        "Plane",
        "Car",
        "Bird",
        "Cat",
        "Deer",
        "Dog",
        "Frog",
        "Horse",
        "Boat",
        "Truck"
    ]
class NeuralNetwork:


    # These are the CIFAR10 class labels from the training data (in order from 0 to 9)
    

    def __init__(self, dataset, test_data):
        self.dataset = dataset
        self.x_train=None
        self.y_train= None
        self.features_x= None
        self.test_data = None
        self.model= None 
        self.images= None 
        self.features= None 
        self.img = None
        self.x_test = None
        self.y_test = None
        self.__import_data()
        self.__build_neural_network()
        self.__export_data()
        self.__load_data()
        self.__finish_model()
        self.__compile_model()
        self.__train_model()
        self.__export_model()
        self.__reload_model()
        self.predict(self.test_data)



    def __import_data(self):
        # Load data from our dataset
        ((self.x_train, self.y_train),(self.x_test,self.y_test)) = self.dataset.build_data()

        # Normalize image data to 0-to-1 range
        self.x_train = vgg16.preprocess_input(self.x_train)
    
    def __build_neural_network(self):
        # Load a pre-trained neural network to use as a feature extractor
        pretrained_nn = vgg16.VGG16(weights='imagenet', include_top=False, input_shape=(32, 32, 3))

        # Extract features for each image (all in one pass)
        self.features_x = pretrained_nn.predict(self.x_train)

    def __export_data(self):
        # Save the array of extracted features to a file
        joblib.dump(self.features_x, "x_train.dat")

        # Save the matching array of expected values to a file
        joblib.dump(self.y_train, "y_train.dat")

    def __load_data(self):
        # Load data set
        self.x_train = joblib.load("x_train.dat")
        self.y_train = joblib.load("y_train.dat")

    def __finish_model(self):
        # Create a model and add layers
        self.model = Sequential()

        self.model.add(Flatten(input_shape=self.x_train.shape[1:]))
        self.model.add(Dense(256, activation='relu'))
        self.model.add(Dropout(0.5))
        self.model.add(Dense(1, activation='sigmoid'))

    def __compile_model(self):
        # Compile the model
        self.model.compile(
            loss="categorical_crossentropy",
            optimizer="adam",
            metrics=['accuracy']
        )

    def __train_model(self):
        # Train the model to be specific to our dataset
        self.model.fit(
            self.x_train,
            self.y_train,
            batch_size=100,
            validation_data= (self.x_test, self.y_test),
            epochs=self.dataset.get_epoch_count(),
            shuffle=True
        )

    def __export_model(self):
        # Save neural network structure to a JSON file 
        self.model_structure = self.model.to_json()
        self.f = Path(self.dataset.get_name() + "_model_structure.json")
        self.f.write_text(self.model_structure)

        # Save neural network's trained weights
        self.model.save_weights(self.dataset.get_name() + "_model_weights.h5")
    def __reload_model(self):
        # Load the json file that contains the model's structure
        self.f = Path(self.dataset.get_name() + "_model_structure.json")
        self.model_structure = self.f.read_text()

        # Recreate the Keras model object from the json data
        self.model = model_from_json(self.model_structure)

        # Re-load the model's trained weights
        self.model.load_weights(self.dataset.get_name()+"_model_weights.h5")
    
    def __prepare_image(self,path_to_file):
        # Load an image file to test, resizing it to 224x224 pixels (as allowed by NASNet)
        img = image.load_img(path_to_file, target_size=(224, 224))

        # Convert the image to a numpy array
        image_array = image.img_to_array(img)

        # Add a forth dimension to the image (since Keras expects a bunch of images, not a single image)
        self.images = np.expand_dims(image_array, axis=0)

        # Normalize the data
        self.images = vgg16.preprocess_input(self.images)

    def __test_image(self):
        # Use the pre-trained neural network to extract features from our test image (the same way we did to train the model)
        feature_extraction_model = vgg16.VGG16(weights='imagenet' , include_top=False, input_shape=(32, 32, 3))
        features = feature_extraction_model.predict(self.images)

        # Given the extracted features, make a final prediction using our own model
        results = self.model.predict(features)

        # Since we are only testing one image with possible class, we only need to check the first result's first element
        image_prediction = results[0]
        self.export_prediction(image_prediction)

    # TODO: now write this to JSON for evan
    def export_prediction(self, image_prediction):
        most_likely_class_index = int(np.argmax(image_prediction))
        class_likelihood = image_prediction[most_likely_class_index]

        class_label = class_labels[most_likely_class_index]
        print("This is a {}, with {} certainty.".format(class_label,class_likelihood))

    def predict(self, path_to_file):
        self.__prepare_image(path_to_file)
        self.__test_image()

from dataset import Dataset

def main():
    ds = Dataset()
    nn = NeuralNetwork(ds, "dog.jpg")
    nn.predict("dog.jpg")

if __name__ == "__main__":
    main()

