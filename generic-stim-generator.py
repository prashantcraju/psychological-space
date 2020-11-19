#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Nov  5 14:43:49 2020

@author: prashantcraju
"""

#import numpy as np

def rgb_to_numpy(rgb):
    return np.array([(rgb[0]-128)/128, (rgb[1]-128)/128, (rgb[2]-128)/128])



def generate_circle_stim(rgb_vals):
    from PIL import Image, ImageDraw
    from keras.preprocessing.image import img_to_array
    
    im = Image.new('RGB', (320, 320), (255, 255, 255))
    draw = ImageDraw.Draw(im)
    draw.ellipse((85, 65, 125, 105), fill=rgb_vals[0])
    draw.ellipse((45, 115, 85, 155), fill=rgb_vals[1])
    draw.ellipse((45, 170, 85, 210), fill=rgb_vals[2])
    draw.ellipse((85, 215, 125, 255), fill=rgb_vals[3])
    
    draw.ellipse((140, 40, 180, 80), fill=rgb_vals[4])
    draw.ellipse((140, 240, 180, 280), fill=rgb_vals[5])
    
    draw.ellipse((195, 65, 235, 105), fill=rgb_vals[6])
    draw.ellipse((230, 115, 270, 155), fill=rgb_vals[7])
    draw.ellipse((230, 170, 270, 210), fill=rgb_vals[8])
    draw.ellipse((195, 215, 235, 255), fill=rgb_vals[9])
    im.thumbnail((32, 32))
    x = img_to_array(im) 
    x = (x - 128.0) / 128.0
    
    return x

def generate_single_color_circle_stim(rgb_vals):
    from PIL import Image, ImageDraw
    from keras.preprocessing.image import img_to_array
    
    im = Image.new('RGB', (320, 320), (255, 255, 255))
    draw = ImageDraw.Draw(im)
    draw.ellipse((85, 65, 125, 105), fill=rgb_vals[0])
    draw.ellipse((45, 115, 85, 155), fill=rgb_vals[0])
    draw.ellipse((45, 170, 85, 210), fill=rgb_vals[0])
    draw.ellipse((85, 215, 125, 255), fill=rgb_vals[0])
    
    draw.ellipse((140, 40, 180, 80), fill=rgb_vals[0])
    draw.ellipse((140, 240, 180, 280), fill=rgb_vals[0])
    
    draw.ellipse((195, 65, 235, 105), fill=rgb_vals[0])
    draw.ellipse((230, 115, 270, 155), fill=rgb_vals[0])
    draw.ellipse((230, 170, 270, 210), fill=rgb_vals[0])
    draw.ellipse((195, 215, 235, 255), fill=rgb_vals[0])
    im.thumbnail((32, 32))
    x = img_to_array(im) 
    x = (x - 128.0) / 128.0
    
    return x

def generate_single_color_square_stim(rgb_vals):
    from PIL import Image, ImageDraw
    from keras.preprocessing.image import img_to_array
    
    im = Image.new('RGB', (320, 320), (255, 255, 255))
    draw = ImageDraw.Draw(im)
    
    draw.rectangle((10, 110, 110, 210), fill=rgb_vals[0])
    draw.rectangle((310, 110, 210, 210), fill=rgb_vals[0])
    draw.rectangle((110, 10, 210, 110, ), fill=rgb_vals[0])
    draw.rectangle((110, 310, 210, 210), fill=rgb_vals[0])

    im.thumbnail((32, 32))
    x = img_to_array(im) 
    x = (x - 128.0) / 128.0
    
    return x


def generate_angle_stim():
    from PIL import Image, ImageDraw
    from keras.preprocessing.image import img_to_array
    import numpy as np
    

    vals = np.arange(0,151)
    dataset = np.ndarray(shape=(len(vals), 32, 32, 3),
                         dtype=np.float32)
    
    for i in range(len(vals)):
        y = 150 - vals[i]
        x = 150 - y
        im = Image.new('RGB', (320, 320), (255, 255, 255))
        draw = ImageDraw.Draw(im)
        draw.line([(160,160),(310-x,310-y)],width=10, fill=(0,0,0))
        draw.line([(10+x,10+y),(160,160)],width=10, fill=(0,0,0))
        im.thumbnail((32, 32))
        x = img_to_array(im) 
        x = (x - 128.0) / 128.0
        dataset[i] = x
    return dataset
    
    
def display_stim(stim):
    import matplotlib.pyplot as plt
    plt.imshow(stim)

