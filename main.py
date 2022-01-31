
# -*- coding: utf-8 -*-
"""
Created on Tue Oct 27 20:37:29 2020
@author: aniket wattamwar
"""

import streamlit as st
from PIL import Image
import cv2 
import numpy as np

from skimage.io import imread, imshow
from skimage import transform
import matplotlib.pyplot as plt
#import numpy as np
import requests
from io import BytesIO




def main():

    selected_box = st.sidebar.selectbox(
    'Choose one of the following',
    ('Welcome','Image Processing')
    )
    
    if selected_box == 'Welcome':
        welcome() 
        
    if selected_box == 'Image Processing':
        photo()


def welcome():
    
    st.title('Image Processing using Streamlit')
    
    st.subheader('A simple app that shows different image processing algorithms. You can choose the options'
             + ' from the left. I have implemented only a few to show how it works on Streamlit. ' + 
             'You are free to add stuff to this app.')
    
    #st.image('hackershrine.jpg',use_column_width=True)


def load_image(filename):
    image = cv2.imread(filename)
    return image
 
def photo():

    url='https://images.unsplash.com/photo-1613048998835-efa6e3e3dc1b?ixlib=rb-1.2.1&ixid=MnwxMjA3fDB8MHxwaG90by1wYWdlfHx8fGVufDB8fHx8&auto=format&fit=crop&w=1074&q=80'

    response = requests.get(url)
    #imgfile = Image.open(BytesIO(response.content))


    st.header("Thresholding, Edge Detection and Contours")
    
    if st.button('See Original Image of Tom'):
        
        original = Image.open(BytesIO(response.content))
        st.image(original, use_column_width=True)
        
    image = cv2.imread("tom.jpg")
    image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    x = st.slider('Change Threshold value',min_value = 50,max_value = 255)  

    ret,thresh1 = cv2.threshold(image,x,255,cv2.THRESH_BINARY)
    thresh1 = thresh1.astype(np.float64)
    st.image(thresh1, use_column_width=True,clamp = True)
    
    st.text(x)
    histr = cv2.calcHist([image],[0],None,[256],[0,256])
    st.bar_chart(histr)
    
    # my own start

    response = requests.get(url)
    imgfile = Image.open(BytesIO(response.content))
    img = np.array(imgfile)


    # Setting Parameter
    phi = 25 # [-70~70]
             # unit degree
    scale_factor = 1 #(this is optional)

    k = 0.7 # need to be positive-value
    b = 0.5 # fix

    #increment = np.tan((phi/180)*3.14)
    #l_side = np.sqrt(5) + increment
    #r_side = np.sqrt(5) - increment

    increment = ((k+b)/b)*np.tan((phi/180)*3.14)
    l_side = np.sqrt( ((k+b)/b)**2 + (k+b)**2 ) + increment
    r_side = np.sqrt( ((k+b)/b)**2 + (k+b)**2 ) - increment


    origin_len = img.shape[0]/2
    origin_wid = img.shape[1]/2

    transform_center = [ origin_wid , origin_len ]
    #transform_l = (0.5*np.sqrt(5)/l_side)*origin_len
    #transform_r = (0.5*np.sqrt(5)/r_side)*origin_len

    transform_l = (np.sqrt(1+b**2)/l_side)*origin_len
    transform_r = (np.sqrt(1+b**2)/r_side)*origin_len
    transform_wid = origin_wid*(b/(k+b))




    #source coordinates

    src_i = np.array([0, 0, 
                    0, img.shape[0],
                    img.shape[1], img.shape[0],
                    img.shape[1], 0,]).reshape((4, 2))



    #destination coordinates

    dst_i = np.array([transform_center[0]-transform_wid*scale_factor, transform_center[1]-transform_l*scale_factor, 
                    transform_center[0]-transform_wid*scale_factor, transform_center[1]+transform_l*scale_factor,
                    transform_center[0]+transform_wid*scale_factor, transform_center[1]+transform_r*scale_factor,
                    transform_center[0]+transform_wid*scale_factor, transform_center[1]-transform_r*scale_factor,]).reshape((4, 2))    



    #using skimage’s transform module where ‘projective’ is our desired parameter
    tform = transform.estimate_transform('projective', src_i, dst_i)
    tf_img = transform.warp(img, tform.inverse)

    #plotting the original image
    plt.imshow(img)

    #plotting the transformed image
    fig, ax = plt.subplots()
    ax.imshow(tf_img)
    _ = ax.set_title('projective transformation')
    plt.plot(transform_center[0],transform_center[1],'x')
    plt.show()

    # my own end
    

    st.text("Press the button below to view Canny Edge Detection Technique")
    if st.button('Canny Edge Detector'):
        image = load_image("jerry.jpg")
        edges = cv2.Canny(image,50,300)
        cv2.imwrite('edges.jpg',edges)
        st.image(edges,use_column_width=True,clamp=True)
      
    y = st.slider('Change Value to increase or decrease contours',min_value = 50,max_value = 255)     
    
    if st.button('Contours'):
        im = load_image("jerry1.jpg")
          
        imgray = cv2.cvtColor(im,cv2.COLOR_BGR2GRAY)
        ret,thresh = cv2.threshold(imgray,y,255,0)
        image, contours, hierarchy = cv2.findContours(thresh,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
        
        img = cv2.drawContours(im, contours, -1, (0,255,0), 3)
 
        
        st.image(thresh, use_column_width=True, clamp = True)
        st.image(img, use_column_width=True, clamp = True)
        


  
    
    
if __name__ == "__main__":
    main()