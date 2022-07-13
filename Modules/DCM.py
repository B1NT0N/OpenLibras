# Data Classification Module

#import Packages
import tensorflow.keras
import numpy as np
import cv2


class Classifier:

    def __init__(self, modelPath, labelsPath=None):
        self.model_path = modelPath
        # Disable scientific notation for clarity
        np.set_printoptions(suppress=True)
        # Load the model
        self.model = tensorflow.keras.models.load_model(self.model_path)

        # Create the array of the right shape to feed into the keras model
        # The 'length' or number of images you can put into the array is
        # determined by the first position in the shape tuple, in this case 1.
        self.data = np.ndarray(shape=(1, 224, 224, 3), dtype=np.float32)
        self.labels_path = labelsPath
        if self.labels_path:
            with open(self.labels_path, "r") as label_file:
                self.list_labels = []
                for line in label_file:
                    stripped_line = line.strip()
                    self.list_labels.append(stripped_line)
            
        else:
            print("Labels Not Found")

    def Predict(self, img, size = (224, 224)):
        # resize the image to a 224x224 with the same strategy as in TM2:
        img_resized = cv2.resize(img, size)
        # turn the image into a numpy array
        image_array = np.asarray(img_resized)
        # Normalize the image
        normalized_image_array = (image_array.astype(np.float32) / 127.0) - 1

        # Load the image into the array
        self.data[0] = normalized_image_array

        # run the inference
        prediction = self.model.predict(self.data)
        indexVal = np.argmax(prediction)

        return list(prediction[0]), indexVal