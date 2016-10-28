from keras.models import load_model
from scipy.misc import imread

# load a model
model = load_model('bettercnn.h5')

# load an image
image = imread('7.PNG').astype(float)

# normalise it in the same manner as we did for the training data
image = image / 255.0

print model.predict(image)