# -*- coding: utf-8 -*-
"""
Created on Mon Oct 18 08:45:36 2021

@author: Anurag
"""

import cv2
import numpy as np
import pandas as pd

from skimage.filters import roberts, sobel, scharr, prewitt
from scipy import ndimage as nd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

img = cv2.imread('Sandstone_Versa0000_train.tif')
img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

df = pd.DataFrame()

# adding pixel values to darta frame

img1 = img.reshape(-1)

df['original_image'] = img1

# add feature

# first set - gabor features

############################################################################    
#Generate Gabor features
num = 1  #To count numbers up in order to give Gabor features a lable in the data frame
kernels = []
for theta in range(2):   #Define number of thetas
    theta = theta / 4. * np.pi
    for sigma in (1, 3):  #Sigma with 1 and 3
        for lamda in np.arange(0, np.pi, np.pi / 4):   #Range of wavelengths
            for gamma in (0.05, 0.5):   #Gamma values of 0.05 and 0.5
                           
                gabor_label = 'Gabor' + str(num)  #Label Gabor columns as Gabor1, Gabor2, etc.
                ksize=9
                kernel = cv2.getGaborKernel((ksize, ksize), sigma, theta, lamda, gamma, 0, ktype=cv2.CV_32F)    
                kernels.append(kernel)
                
                #Now filter the image and add values to a new column 
                fimg = cv2.filter2D(img, cv2.CV_8UC3, kernel)
                filtered_img = fimg.reshape(-1)
                
                
                df[gabor_label] = filtered_img  #Labels columns as Gabor1, Gabor2, etc.
                # print(gabor_label, ': theta=', theta, ': sigma=', sigma, ': lamda=', lamda, ': gamma=', gamma)
                num += 1  #Increment for gabor column label
                
    
########################################

# canny edge detector

edges = cv2.Canny(img, 100, 200)
edge1 = edges.reshape(-1)
df['Canny_edge'] = edge1


edge_roberts = roberts(img)
edge_roberts1 = edge_roberts.reshape(-1) 
df['robert'] = edge_roberts1

edge_sobels = sobel(img)
edge_sobels1 = edge_sobels.reshape(-1) 
df['sobel'] = edge_sobels1

edge_scharr = scharr(img)
edge_scharr1 = edge_scharr.reshape(-1) 
df['scharr'] = edge_scharr1

edge_prewitt = prewitt(img)
edge_prewitt1 = edge_prewitt.reshape(-1) 
df['prewitt'] = edge_prewitt1

# gaussian filter with sigma 3
gaussing_img = nd.gaussian_filter(img, sigma=3)
gaussing_img1 = gaussing_img.reshape(-1)
df['gaussing_img_s3'] = gaussing_img1

# gaussian filter with sigma 7
gaussing_img_7 = nd.gaussian_filter(img, sigma=7)
gaussing_img2 = gaussing_img_7.reshape(-1)
df['gaussing_img_s7'] = gaussing_img2

#median with sigma 3
median_img = nd.median_filter(img, size=3)
median_img1 = median_img.reshape(-1)
df['median_img'] = median_img1

# #variance with size 3
# variance_img = nd.generic_filter(img, np.var,size=3)
# variance_img1 = variance_img.reshape(-1)
# df['variance_img'] = variance_img1

labeled_img = cv2.imread('Sandstone_Versa0000_mask.tif')
labeled_img = cv2.cvtColor(labeled_img, cv2.COLOR_BGR2GRAY)
labeled_img1 = labeled_img.reshape(-1)
df['labeled_img'] = labeled_img1

# print(df.head())

#split data set

X = df.iloc[:,:-1].values
y = df.iloc[:, -1].values


x_train, x_test, y_train, y_test =  train_test_split(X, y, test_size=0.4, random_state=0)

print(x_train.shape, y_train.shape)

#calling regressor

regressor = RandomForestClassifier(n_estimators=10, random_state=0)
regressor.fit(x_train, y_train)

pred = regressor.predict(x_test)

# accuracy calculation
result = accuracy_score(y_test, pred)
print('accuracy : ', result)

# feature_selection = list(regressor.feature_importances_)

feature_list = list(df.columns[:-1])
# print(feature_list)

feature_imp = pd.Series(regressor.feature_importances_, index=feature_list).sort_values(ascending=False)
print(feature_imp)

# savemodel
import pickle

file_name = 'segmentation_model'
pickle.dump(regressor, open(file_name, 'wb'))

#%%
load_model = pickle.load(open(file_name, 'rb'))

result = load_model.predict(X)

segmented = result.reshape((img.shape))

import matplotlib.pyplot as plt

plt.imshow(segmented, cmap='jet')
# plt.imsave('segmeneted_image_colored.jpg', segmented, cmap='jet')
plt.show()




