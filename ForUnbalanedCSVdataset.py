#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import os
os.environ['CUDA_VISIBLE_DEVICES']='-1'
import pandas as pd
import matplotlib.pyplot as plt 
from sklearn.model_selection import train_test_split
from tensorflow import keras
from tensorflow.keras import layers
import tensorflow as tf
from keras.models import Sequential
from keras.layers import Convolution2D,MaxPool2D,Flatten,Dense,Dropout 
from keras_preprocessing.image import ImageDataGenerator
from keras.utils.np_utils import to_categorical
import cv2


# In[2]:


data = pd.read_csv('A_Z Handwritten Data.csv')
data.info()


# In[3]:


class_names = ["A", "B", "C", "D", "E", "F", "G", "H", "I", "J", "K", "L", "M", "N", "O",
               "P", "Q", "R", "S", "T", "U", "V", "W", "X", "Y", "Z"]
len(class_names)


# In[4]:


ndata=[]
n=0
for x in range(0,len(class_names)):
    #print(len(np.where(y_train==x)[0]),end=" ")
    #n=n+1
    ndata.append(len(np.where(data['0']==x)[0]))
#print("\n",n)
print(ndata)


# In[5]:


plt.figure(figsize=(10,5))
plt.bar(range(0,len(class_names)),ndata)
plt.xlabel("Class no")
plt.ylabel("number of images")
plt.show()


# In[6]:


sample = data.iloc[1000].values # value of a particular row
sample_label = sample[0] # 1st data of that row which indicates the index of english letter
sample_letter = sample[1:].reshape(28,28) #taking from index[1] to the last and reshaping it to 28x28 fromat


# In[7]:


plt.imshow(sample_letter, cmap="binary")
plt.axis('off')
plt.title(class_names[sample_label])


# In[8]:


labels = data['0'].values.astype('uint8')
X = data.drop('0', axis=1)
print(X.shape)
labels


# In[9]:


X = np.array(X).reshape(len(data), 28, 28, 1)
X.shape


# In[10]:


# Spliting trainning and test data sets
X_train, X_test, y_train, y_test = train_test_split(X, labels, test_size = 0.2, random_state = 42)
X_train, X_validation, y_train, y_validation = train_test_split(X_train, y_train, test_size = 0.2, random_state = 42)

# scale images to [0, 1] range # Normalization
X_train = X_train.astype("float32") / 255
X_test = X_test.astype("float32") / 255
X_validation=X_validation.astype("float32") / 255


# Check image shape
print("x_train shape:", X_train.shape)
print(X_train.shape[0], "train samples")
print(X_test.shape[0], "test samples")
print(X_validation.shape[0], "validation samples")


# In[12]:


nsamples=[]
n=0
for x in range(0,len(class_names)):
    #print(len(np.where(y_train==x)[0]),end=" ")
    #n=n+1
    nsamples.append(len(np.where(y_train==x)[0]))
#print("\n",n)
print(nsamples)


# In[13]:


plt.figure(figsize=(10,5))
plt.bar(range(0,len(class_names)),nsamples)
plt.xlabel("Class no")
plt.ylabel("number of images")
plt.show()


# In[14]:


img = X_train[10000]
img = cv2.resize(img,(300,300))
cv2.imshow(str(y_train[10000]),img)
cv2.waitKey(0)


# In[15]:


# Slight image augmentation
dataGeneration = ImageDataGenerator(width_shift_range=0.1,
                                    height_shift_range=0.1,
                                    zoom_range=0.2,
                                    shear_range=0.1,
                                    rotation_range=10)
dataGeneration.fit(X_train) #generating X_train before sending it for training

# One hot encoding

y_train = to_categorical(y_train,len(class_names))
y_test =to_categorical(y_test,len(class_names))
y_validation = to_categorical(y_validation,len(class_names))
print(y_train.shape)


# In[16]:


# The Model
keras.backend.clear_session()
tf.random.set_seed(12)
np.random.seed(12)

model = Sequential()
model.add(Convolution2D(64,5,5,input_shape=(28,28,1),padding='same', activation='relu'))
model.add(Convolution2D(64,5,5,padding='same', activation='relu'))
model.add(MaxPool2D(pool_size=(2,2),padding='same'))
model.add(Convolution2D(32,3,3,input_shape=(28,28,1),padding='same', activation='relu'))
model.add(Convolution2D(32,3,3,input_shape=(28,28,1),padding='same', activation='relu'))
model.add(MaxPool2D(pool_size=(2,2),padding='same'))
model.add(Dropout(0.50))
model.add(Flatten())
#Full connection
model.add(Dense(128, activation='relu'))
model.add(Dropout(0.50))
model.add(Dense(len(class_names), activation='softmax'))

model.compile(loss="categorical_crossentropy", optimizer=keras.optimizers.Adam(learning_rate=0.001),metrics = ["accuracy"])
epochs = 10
batchsize = 70
stepPerEpoch = 2000
history = model.fit(dataGeneration.flow(X_train, y_train,batch_size=batchsize), epochs=epochs, validation_data=(X_validation, y_validation),shuffle=1)  


# In[17]:


model.save("CsvDataSetTrainded.h5")


# In[18]:


plt.figure(1)
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.legend(['training','validation'])
plt.title('Loss')
plt.xlabel('epoch')
plt.show()


# In[19]:


plt.figure(2)
plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.legend(['training','validation'])
plt.title('Accuracy')
plt.xlabel('epoch')
plt.show()

