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


# In[8]:


#Getting data & preprocessing
path1 = 'ImagesOfLetters/Uppercase'
path2 = 'ImagesOfLetters/Lowercase'  # Folder/path name
mylist = os.listdir(path1)  # list of directories/folders in that path
mylist = mylist + os.listdir(path2)
nclass = len(mylist)
print(mylist," \n", nclass,"\n")
#c = 0
images=[]
classNo=[] #Labels. also dependent variable
for x in range(0, nclass):
    if x<26:  # searching directories in ascii value
        myPiclist = os.listdir(path1+"/"+str(mylist[x]))
        #c=c+1
    else:
        myPiclist = os.listdir(path2+"/"+str(mylist[x]))
        #c=c+1
    #print(len(myPiclist),"\n")
    for y in myPiclist:
        if x<26:
            currentImg = cv2.imread(path1+"/"+str(mylist[x])+'/'+y)
        else:
            currentImg = cv2.imread(path2+"/"+str(mylist[x])+"/"+y)
         
        images.append(currentImg)
        classNo.append(x) 
    print(x,end=" ")

def preprocessing(currentImg):
    currentImg = cv2.cvtColor(currentImg, cv2.COLOR_BGR2GRAY)
    currentImg = cv2.resize(currentImg,(28,28))
    currentImg=cv2.equalizeHist(currentImg)
    #currentImg=currentImg/255 # scale images to [0, 1] range
    return currentImg

images = np.array(list(map(preprocessing,images)))
print("\nTotal Number of Image = ",len(images))       
print("Total Number of classno = ",len(classNo))


# In[9]:


print(images.shape)
print(len(classNo))
# Converting it into numpy array
images = np.array(images)
classNo = np.array(classNo)
print(images.shape)
#Adding depth
images = images.reshape(images.shape[0],images.shape[1],images.shape[2],1)
print(images.shape)


# In[10]:


# Spliting trainning and test data sets
X_train, X_test, y_train, y_test = train_test_split(images, classNo, test_size = 0.2, random_state = 42)
X_train, X_validation, y_train, y_validation = train_test_split(X_train, y_train, test_size = 0.2, random_state = 42)

# scale images to [0, 1] range
X_train = X_train.astype("float32") / 255
X_test = X_test.astype("float32") / 255
X_validation=X_validation.astype("float32") / 255

# Check image shape
print("x_train shape:", X_train.shape)
print(X_train.shape[0], "train samples")
print(X_test.shape[0], "test samples")
print(X_validation.shape[0], "validation samples")


# In[11]:


# to check number of images for each class
nsamples=[]
#n=0
for x in range(0,nclass):
    #print(len(np.where(y_train==x)[0]),end=" ")
    #n=n+1
    nsamples.append(len(np.where(y_train==x)[0]))
#print("\n",n)
print(nsamples)


# In[12]:


plt.figure(figsize=(10,5))
plt.bar(range(0,nclass),nsamples)
plt.xlabel("Class no")
plt.ylabel("number of images")
plt.show()


# In[15]:


img = X_train[1011]
img = cv2.resize(img,(300,300))
cv2.imshow(str(y_train[1011]),img)
cv2.waitKey(0)


# In[16]:


# Slight image augmentation
dataGeneration = ImageDataGenerator(width_shift_range=0.1,
                                    height_shift_range=0.1,
                                    zoom_range=0.2,
                                    shear_range=0.1,
                                    rotation_range=10)
dataGeneration.fit(X_train) #generating X_train before sending it for training

# One hot encoding

y_train = to_categorical(y_train,nclass)
y_test =to_categorical(y_test,nclass)
y_validation = to_categorical(y_validation,nclass)
y_train.shape


# In[21]:


# The Model
keras.backend.clear_session()
tf.random.set_seed(12)
np.random.seed(12)

model = Sequential()
model.add(Convolution2D(64,5,5,input_shape=(28,28,1),padding='same', activation='relu'))
#model.add(Convolution2D(64,5,5,input_shape=(28,28,1),padding='same', activation='relu'))
#model.add(Convolution2D(64,5,5,padding='same', activation='relu'))
model.add(MaxPool2D(pool_size=(2,2),padding='same'))
model.add(Convolution2D(32,3,3,input_shape=(28,28,1),padding='same', activation='relu'))
#model.add(Convolution2D(32,3,3,input_shape=(28,28,1),padding='same', activation='relu'))
model.add(MaxPool2D(pool_size=(2,2),padding='same'))
model.add(Dropout(0.50))
model.add(Flatten())
#Full connection
model.add(Dense(128, activation='relu'))
model.add(Dropout(0.50))
model.add(Dense(nclass, activation='softmax'))

#model.compile(loss="categorical_crossentropy", optimizer=keras.optimizers.Adam(learning_rate=0.001),metrics = ["accuracy"])
model.compile(loss="categorical_crossentropy", optimizer=keras.optimizers.Adam(learning_rate=0.001),metrics = ["accuracy"])

epochs = 10
batchsize = 20
stepPerEpoch = 2000
history = model.fit(dataGeneration.flow(X_train, y_train,batch_size=batchsize), epochs=epochs, validation_data=(X_validation, y_validation),shuffle=1)  

