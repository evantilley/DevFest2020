import keras
from keras.models import Sequential
from keras.models import Dense, Dropout, Flatten, Conv2D, MaxPooling2D
from pathlib import Path 

# Load The Data Set
(x_train, y_train), (x_test, y_test) = Dataset.load_data()

# Normalize the dataset range to between 0 and 1 float values
x_train = x_train.astype("float32")
x_test = x_test.astype("float32")
x_train /= 255
x_test /= 255

# Convert class vectors to binary class matrices
# Our labels are single values between 0 and num_labels
# Instead we want to map each label to be assigned to an
# array containing the correct element set to "1" that
# corresponds to the correct dataset 
y_train = keras.utils.to_categorical(y_train, num_classes)
y_test = keras.utils.to_categorical(y_test, num_classes)

# Creating our model with layers
model = Sequential()

model.add(Conv2D(32,(4,4),padding="same",activation="relu", input_shape=(128,128,3)))
model.add(Conv2D(32,(4,4),activation="relu"))
model.add(MaxPooling2D(pool_size=(2,2)))
model.add(Dropout(0.25))

model.add(Conv2D(64,(4,4),padding="same",activation="relu"))
model.add(Conv2D(64,(4,4),activation="relu"))
model.add(MaxPooling2D(pool_size=(2,2)))
model.add(Dropout(0.25))


model.add(Flatten())
model.add(Dense(512, activation="relu"))
model.add(Dropout(0.5)) #makes the model work really hard to get the right answer
model.add(Dense(10, activation="softmax"))

#Compile the model
model.compile(
    loss="categorical_crossentropy",
    optimizer = "adam",
    metrics = ["accuracy"]
)

#Train the model
model.fit(
    x_train,
    y_train,
    batch_size= Dataset.get_batch_size(), #Retrieves the proper number of images to use in a batch
    epochs = Dataset.get_epoch_count(), #Retrieves the proper 
    validation_data= [x_test, y_test],
    shuffle=True

)

#Save neural network structure
model_structure = model.to_json()
f = Path(Dataset.get_name() + "_model_structure.json")
f.write_text(model_structure)

#Save neural network's trained weights
model.sample_weights(Dataset.get_name() + "_model_weights.h5")


# Print a summary of the model 
model.summary()