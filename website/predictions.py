from ultralytics import YOLO
import tensorflow as tf
from keras.models import load_model
from os import listdir, path
import pandas as pd
import numpy as np
import cv2

# Define the first model for object identification
object_model = YOLO('Models/best_musa.pt')

# Define the second model for object classification
class_model = load_model('Models/resnet_model_1')

image_path = 'website/static/70.png'
image = cv2.imread(image_path)

# Define a function to preprocess the input image
def preprocess_image(img, results):

  a = results[0].boxes.boxes
  px = pd.DataFrame(a).astype("float")
  
  for index, row in px.iterrows():
    x1 = int(row[0])
    y1 = int(row[1])
    x2 = int(row[2])
    y2 = int(row[3])

    if (x1 != 0) and (x2 != 0):
        # Crop the object from the image
        # convert PIL image to numpy array
        numpy_image = np.array(img)

        # convert numpy array to cv2 image object
        cv2_image = cv2.cvtColor(numpy_image, cv2.COLOR_RGB2BGR)
        object_image = cv2_image[y1:y2, x1:x2]

        img_height, img_width = 180, 180
        image_resized = cv2.resize(object_image, (img_height, img_width))
        image = np.expand_dims(image_resized, axis=0)
        
        return image
    else:
        continue
  return image


def pipeline(image):

  # img = cv2.imread(image_path)
  results = object_model(image)
  # Preprocess the input image
  img = preprocess_image(image, results)

  # Use the class model to classify the identified object
  class_probabilities = class_model.predict(img)

   # TODO: check whether class names can be obtained from saved model
  class_names = ['cow1', 'cow10', 'cow11', 'cow12', 'cow13', 'cow14', 'cow15', 'cow16', 'cow17', 'cow18', 'cow19', 'cow2', 'cow20', 'cow21', 'cow22', 'cow3', 'cow4', 'cow5', 'cow6', 'cow7', 'cow8', 'cow9']


  # Get the predicted class label
  output_class = class_names[np.argmax(class_probabilities)]

  return output_class


print(pipeline(image))
