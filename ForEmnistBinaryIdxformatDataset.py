#!/usr/bin/env python
# coding: utf-8

# In[1]:


from keras.layers import Convolution2D, MaxPool2D, Flatten, Dense, Dropout
import matplotlib.pyplot as plt
import idx2numpy
import cv2
from keras.utils.np_utils import to_categorical
from keras_preprocessing.image import ImageDataGenerator
from keras.models import Sequential
import tensorflow as tf
from tensorflow.keras import layers
from tensorflow import keras
from sklearn.model_selection import train_test_split
import pandas as pd
import numpy as np
import os
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'


# In[3]:


def rotate(image):
    #image = image.reshape(28, 28)
    image = np.fliplr(image)
    image = np.rot90(image.reshape(-1, 28, 28), k=-1, axes=(-2, -1))

    return image


# In[4]:


X_train = idx2numpy.convert_from_file(
    'gzip/emnist-balanced-train-images-idx3-ubyte')
# X_train.shape
X_train = rotate(X_train)

X_test = idx2numpy.convert_from_file(
    'gzip/emnist-balanced-test-images-idx3-ubyte')
X_test = rotate(X_test)
# X_test.shape
print(X_train.shape, " ", X_test.shape)
Y_train = idx2numpy.convert_from_file(
    'gzip/emnist-balanced-train-labels-idx1-ubyte')
print(Y_train.shape)
print(Y_train)
Y_test = idx2numpy.convert_from_file(
    'gzip/emnist-balanced-test-labels-idx1-ubyte')
mapp = pd.read_csv("gzip/emnist-balanced-mapping.txt",
                   delimiter=' ', index_col=0, header=None, squeeze=True)
print(mapp.shape)
X_train = X_train.reshape(
    X_train.shape[0], X_train.shape[1], X_train.shape[2], 1)
# X_train = np.apply_along_axis(rotate, 1, X_train) # rotating
X_test = X_test.reshape(X_test.shape[0], X_test.shape[1], X_test.shape[2], 1)
# X_tests = np.apply_along_axis(rotate, 1, X_test)# rotating
print(X_train.shape, " ", X_test.shape)
X_train, X_validation, Y_train, Y_validation = train_test_split(
    X_train, Y_train, test_size=0.2, random_state=42)
print(Y_train.shape)


# In[5]:


# Normalization
X_train = X_train.astype("float32") / 255
X_test = X_test.astype("float32") / 255
X_validation = X_validation.astype("float32") / 255
# X_train[6]


# In[6]:


nsamples = []
n = 0
for x in range(0, 47):
    #print(len(np.where(y_train==x)[0]),end=" ")
    # n=n+1
    nsamples.append(len(np.where(Y_train == x)[0]))
# print("\n",n)
print(nsamples)
print(len(nsamples))


# In[7]:


plt.figure(figsize=(10, 5))
plt.bar(range(0, 47), nsamples)
plt.xlabel("Class no")
plt.ylabel("number of images")
plt.show()


# In[17]:


# print(X_train[9090])
img = X_train[9090]
img = cv2.resize(img, (300, 300))
cv2.imshow(chr(mapp[Y_train[9090]]), img)
cv2.waitKey(0)
# X_train[9090]


# In[9]:


# Slight image augmentation
dataGeneration = ImageDataGenerator(width_shift_range=0.1,
                                    height_shift_range=0.1,
                                    zoom_range=0.2,
                                    shear_range=0.1,
                                    rotation_range=10)


# generating X_train before sending it for training
dataGeneration.fit(X_train)

# One hot encoding

Y_train = to_categorical(Y_train, 47)
Y_test = to_categorical(Y_test, 47)
Y_validation = to_categorical(Y_validation, 47)
print(Y_train.shape)


# In[11]:


# The Model
keras.backend.clear_session()
tf.random.set_seed(12)
np.random.seed(12)

model = Sequential()
model.add(Convolution2D(64, 5, 5, input_shape=(
    28, 28, 1), padding='same', activation='relu'))
model.add(Convolution2D(64, 5, 5, padding='same', activation='relu'))
model.add(MaxPool2D(pool_size=(2, 2), padding='same'))
model.add(Convolution2D(32, 3, 3, input_shape=(
    28, 28, 1), padding='same', activation='relu'))
model.add(Convolution2D(32, 3, 3, input_shape=(
    28, 28, 1), padding='same', activation='relu'))
model.add(MaxPool2D(pool_size=(2, 2), padding='same'))
model.add(Dropout(0.50))
model.add(Flatten())
# Full connection
model.add(Dense(128, activation='relu'))
model.add(Dropout(0.50))
model.add(Dense(47, activation='softmax'))

model.compile(loss="categorical_crossentropy", optimizer=keras.optimizers.Adam(
    learning_rate=0.001), metrics=["accuracy"])
epochs = 10
batchsize = 50
stepPerEpoch = 2000
history = model.fit(dataGeneration.flow(X_train, Y_train, batch_size=batchsize),
                    epochs=epochs, validation_data=(X_validation, Y_validation), shuffle=1)


# In[12]:


model.save("cnn_trainned_model3.h5")


# In[13]:


plt.figure(1)
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.legend(['training', 'validation'])
plt.title('Loss')
plt.xlabel('epoch')
plt.show()


# In[14]:


plt.figure(2)
plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.legend(['training', 'validation'])
plt.title('Accuracy')
plt.xlabel('epoch')
plt.show()


# In[39]:


path1 = 'test'
model = tf.keras.models.load_model('EmnistBalanedDatasetTrained.h5')


imgOriginal = os.listdir(path1)
print(imgOriginal)
imgOrig = imgOriginal[0]
print(imgOrig)


# In[40]:


imgOriginal = cv2.imread(path1+"/"+str(imgOrig))
imgn = imgOriginal
imgn = cv2.resize(imgn, (300, 300))
cv2.imshow("fdfdf", imgn)
cv2.waitKey(0)
img = np.asarray(imgOriginal)
img = cv2.resize(img, (28, 28))
img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
img = cv2.equalizeHist(img)


# In[41]:


img = np.array(img)
img = img.flatten()  # flattening in 1d array
for m in range(0, 784):  # coverting the color value to make the image black on white background
    img[m] = 255-img[m]
img = img.astype("float32")/255  # Normalize
# print(len(curimg))

img = img.reshape(1, 28, 28, 1)
print(img.shape)


# In[42]:


predictions = np.argmax(model.predict(img))
percent = np.amax(model.predict(img))
print(chr(mapp[predictions]))
print(predictions, "\n", percent)
# print(model.predict(img))
