from keras.preprocessing.image import load_img, img_to_array
from keras.models import load_model
import numpy as np
from keras.models import load_model
import cv2
import numpy as np

# Load the pre-trained deepfake detection model (XceptionNet or similar)
model = load_model('deepfake_xception_model.h5')

# Read and preprocess the input image
img = cv2.imread('image.jpg')
img = cv2.resize(img, (299, 299))  # Resize to the input size expected by Xception
img = np.expand_dims(img, axis=0)
img = img / 255.0  # Normalize the image

# Predict the deepfake likelihood
prediction = model.predict(img)
print(f"Deepfake likelihood: {prediction[0][0]}") 