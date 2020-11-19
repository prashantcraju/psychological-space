#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Nov  3 10:23:28 2020

@author: prashantcraju
"""


if __name__ == "__main__": 
    import numpy as np
    import os, sys
    from scipy import ndimage
    from keras.preprocessing.image import ImageDataGenerator, array_to_img, img_to_array, load_img
    import tensorflow as tf
    from tqdm import tqdm
    from alibi_detect.models.losses import elbo
    from VAE_class import VAE
    #from alibi_detect.od import OutlierVAE
    #from alibi_detect.utils.fetching import fetch_detector
    #from alibi_detect.utils.perturbation import apply_mask
#    from alibi_detect.utils.saving import save_detector
    #from alibi_detect.utils.visualize import plot_instance_score, plot_feature_outlier_image
    
    
    
    file_path = 'imagenet.npy'
    dataset = np.load(file_path)
    
    beta_vals = np.arange(.25,4.5,.25)
    
    for beta in beta_vals:
        path = '{}{}'.format('trained_vae_beta_val_',beta)
        vae = VAE((32, 32, 3),1024, beta)
        vae.fit(dataset, 15, True, path)






        
