# -*- coding: utf-8 -*-
"""
Created on Mon Oct 18 18:49:59 2021

@author: Anurag
"""

import cv2
import numpy as np
import pandas as pd
from skimage.filters import roberts, sobel, scharr, prewitt
from scipy import ndimage as nd
import pickle
import matplotlib.pyplot as plt


def extract_feature(img):
    
    df = pd.DataFrame()

    # adding pixel values to darta frame
    
    img1 = img.reshape(-1)
    
    df['original_image'] = img1
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
    return df


if __name__ == '__main__':
    
    file_name = 'segmentation_model'

    load_model = pickle.load(open(file_name, 'rb'))
    
        
    img = cv2.imread('Sandstone_Versa0050_test.tif')
    img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    x = extract_feature(img_gray)
    
    result = load_model.predict(x)
    
    segmented_img = result.reshape((img_gray.shape))
    # plt.imsave('segmented_color_Sandstone_Versa0050.jpg', segmented_img, cmap='jet')
    # plt.imsave('segmented_grey_Sandstone_Versa0050.jpg', segmented_img, cmap='gray')
    # plt.imsave('result/'+ name + '.jpg', segmented_img, cmap='jet')
    # plt.subplot(1,3,1)
    # plt.imshow(img, cmap='gray')
    # plt.title('original image')
        
    # plt.subplot(1,3,2)
    # plt.imshow(segmented_img,cmap='gray')
    # plt.title('segmented gray image')
    
    # plt.subplot(1,3,3)
    # plt.imshow(segmented_img,cmap='jet')
    # plt.title('segmented color image')
    # plt.show()    
    
    f = plt.figure(figsize=(20,6))
    ax = f.add_subplot(131)
    ax.imshow(img, cmap='gray')
    ax.set_title('original image')
    ax2 = f.add_subplot(132)
    ax2.imshow(segmented_img,cmap='gray')
    ax2.set_title('segmented gray image')
    ax3 = f.add_subplot(133)
    ax3.imshow(segmented_img,cmap='jet')
    ax3.set_title('segmented color image')
    f.savefig('input_ouput_image')
    plt.show()
