
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
    
    '''
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
    '''     


  
    
    
if __name__ == "__main__":
    main()