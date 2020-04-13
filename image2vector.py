from keras.applications.vgg16 import VGG16
import keras.applications.vgg16
from keras.preprocessing import image
import numpy as np

from keras.applications.vgg19 import VGG19
import keras.applications.vgg19

from keras.applications.xception import Xception
import keras.applications.xception

from skimage.feature import hog
import skimage.io

import numpy as np

vgg16_model=None
vgg19_model=None
xception_model=None

def get_vgg16_features(img_path,flatten=True):
    global vgg16_model
    if vgg16_model is None:
        vgg16_model = VGG16(weights='imagenet', include_top=False)
    img = image.load_img(img_path, target_size=(224, 224))
    x = image.img_to_array(img)
    x = np.expand_dims(x, axis=0)
    x = keras.applications.vgg16.preprocess_input(x)

    features = vgg16_model.predict(x)[0]# last max pooling layer (7 x 7 x 512)
    feature_np = np.array(features)

    if flatten is True:
        return feature_np.flatten()
    else:
        return feature_np

def get_vgg19_features(img_path,flatten=True):
    global vgg19_model
    if vgg19_model is None:
        vgg19_model = VGG19(weights='imagenet', include_top=False)
    img = image.load_img(img_path, target_size=(224, 224))
    x = image.img_to_array(img)
    x = np.expand_dims(x, axis=0)
    x = keras.applications.vgg19.preprocess_input(x)

    features = vgg19_model.predict(x)[0]# last max pooling layer (7 x 7 x 512)
    feature_np = np.array(features)

    if flatten is True:
        return feature_np.flatten()
    else:
        return feature_np

def get_xception_features(img_path,flatten=True):
    global xception_model
    if xception_model is None:
        xception_model=Xception(weights='imagenet',include_top=False)
    img = image.load_img(img_path, target_size=(299, 299))
    x = image.img_to_array(img)
    x = np.expand_dims(x, axis=0)
    x = keras.applications.xception.preprocess_input(x)
    features = xception_model.predict(x)[0]  #
    feature_np = np.array(features)

    if flatten is True:
        return feature_np.flatten()
    else:
        return feature_np

def get_hog_features(img_path):
    image=skimage.io.imread(img_path,as_gray=False)
    features= hog(image, orientations=8, pixels_per_cell=(16, 16),
                        cells_per_block=(1, 1), visualize=False, multichannel=True)
    return features

def get_features(img_path,type):
    if type == 'vgg16':
        return (get_vgg16_features(img_path),type)
    elif type == 'vgg19':
        return (get_vgg19_features(img_path),type)
    elif type == 'xception':
        return (get_xception_features(img_path),type)
    elif type == 'hog':
        return (get_hog_features(img_path),type)
    else:
        return "Unknown feature type"


print(get_features(img_path='test_data/chest-xray.jpg',type='vgg16'))
