#!/usr/bin/env python
# coding: utf-8

# In[137]:


import tensorflow as tf
import numpy as np
import cv2
import os
import pandas as pd
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'


# In[227]:


path1 = 'test'
model = tf.keras.models.load_model('EmnistBalanedDatasetTrained.h5')
mapp = pd.read_csv("gzip/emnist-balanced-mapping.txt",
                   delimiter=' ', index_col=0, header=None, squeeze=True)

imgOriginal = os.listdir(path1)
print(imgOriginal)
imgOrig =imgOriginal[3]
print(imgOrig)


# In[228]:


imgOriginal = cv2.imread(path1+"/"+str(imgOrig))
imgn = imgOriginal
imgn = cv2.resize(imgn,(300,300))
cv2.imshow("fdfdf",imgn)
cv2.waitKey(0)
img = np.asarray(imgOriginal)
img = cv2.resize(img, (28, 28))
img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
img = cv2.equalizeHist(img)


# In[229]:


img = np.array(img)
img = img.flatten()  # flattening in 1d array
for m in range(0, 784):  # coverting the color value to make the image black on white background
        img[m] = 255-img[m]
img = img.astype("float32")/255
# print(len(curimg))

img = img.reshape(1, 28, 28, 1)
print(img.shape)


# In[231]:


predictions = np.argmax(model.predict(img))
percent = np.amax(model.predict(img))
print(chr(mapp[predictions]))
print(predictions,"\n",percent)

