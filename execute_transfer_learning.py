# -*- coding: utf-8 -*-
"""
Created on Fri Sep 13 09:32:25 2019

@author: Dakota

Purpose: 
    Execute the transfer learining script. 

"""

import keras_transfer_learning as ktl

"""Initialize the retrain class"""
retrain = ktl.retrain()

"""Verify that a GPU is available"""
retrain.check_for_gpu()

""" Run the script """
data_path = 'D:/pupster-data/DATA/15-second'
retrain.train(data_path)

""" Show Training Loss & Accuracy Stats """
retrain.show_trainng_stats()

""" Save the Model """
retrain.save_model()

""" Make a predcition on an image """ 
image = 'D:/pupster-data/DATA/15-second/image_1'
retrain.predict_with_internal(image)
