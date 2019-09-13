# -*- coding: utf-8 -*-
"""
Created on Fri Sep 13 08:16:44 2019

@author: Dakota

Purpose: 
    Use transfer learning to quickly retrain a model on new data using 
    Tensorflow and Keras. 

"""

from __future__ import absolute_import, division, print_function, unicode_literals

import tensorflow as tf
tf.enable_eager_execution()

import tensorflow_hub as hub
from tensorflow.keras import layers
import keras 

import matplotlib.pylab as plt
import numpy as np
import time
import cv2

class CollectBatchStats(tf.keras.callbacks.Callback):
    """ Used for capturing training statistics"""
    def __init__(self):
        self.batch_losses = []
        self.batch_acc = []

    def on_train_batch_end(self, batch, logs=None):
        self.batch_losses.append(logs['loss'])
        self.batch_acc.append(logs['acc'])
        self.model.reset_metrics()
            
class retrain:
    def __init__(self, model_type='inception-v3'):
        self.model_type = model_type
        if self.model_type == 'inception-v3':
            self.IMAGE_SHAPE = (299,299)
        elif self.model_type == 'mobilnet-v2':
            self.IMAGE_SHAPE = (224,224)
    
    def check_for_gpu(self):
        print('Checking for GPUs...')
        available_gpus = keras.backend.tensorflow_backend._get_available_gpus()
        print('Available GPUs:', available_gpus)
        if len(available_gpus) < 1:
            print('WARNING: No GPUs available to Keras')
            print('Using CPU...')
        
    def get_data(self, data_path):
        print('Getting Data...')
        image_generator = tf.keras.preprocessing.image.ImageDataGenerator(rescale=1/255)
        self.image_data = image_generator.flow_from_directory(str(data_path), target_size=self.IMAGE_SHAPE)
        for self.image_batch, self.label_batch in self.image_data:
            print("Image batch shape: ", self.image_batch.shape)
            print("Label batch shape: ", self.label_batch.shape)
            break
        self.class_names = sorted(self.image_data.class_indices.items(), key=lambda pair:pair[1])
        self.class_names = np.array([key.title() for key, value in self.class_names])
        return self.class_names
    def get_pretrained_model(self):
        print('Retrieving Pretrained Model...')
        image_height, image_width = self.IMAGE_SHAPE
        if self.model_type == 'inception-v3':
            feature_extractor_url = 'https://tfhub.dev/google/tf2-preview/inception_v3/feature_vector/4'
        elif self.model_type == 'mobilnet-v2':
            feature_extractor_url = "https://tfhub.dev/google/tf2-preview/mobilenet_v2/feature_vector/2"
        feature_extractor_layer = hub.KerasLayer(feature_extractor_url,
                                                 input_shape=(image_height,image_width,3))
        feature_extractor_layer.trainable = False
        return feature_extractor_layer
    
    def create_model(self):
        print('Creating Model...')
        feature_extractor_layer = self.get_pretrained_model()
        self.model = tf.keras.Sequential([feature_extractor_layer,
                                          layers.Dense(self.image_data.num_classes, 
                                          activation='softmax')])
        self.model.compile(optimizer=tf.keras.optimizers.Adam(),
                           loss='categorical_crossentropy',
                           metrics=['acc'])
        return self.model.summary()
        
    def train(self,data_path, steps='default', epochs=2, callbacks=CollectBatchStats()):
        self.get_data(data_path)
        self.create_model()
        self.callbacks = callbacks
        print('Training...')
        if steps == 'default':
            steps_per_epoch = np.ceil(self.image_data.samples/self.image_data.batch_size)
        else:
            steps_per_epoch = steps
        self.history = self.model.fit(self.image_data,
                                      epochs=epochs,
                                      steps_per_epoch=steps_per_epoch,
                                      callbacks = [self.callbacks],
                                      verbose=1)
        
        return self.callbacks
    
    def save_model(self, save_path='/tmp/saved_models/model_'):
        print('Saving  Model...')
        date_time = time.ctime()
        export_path = save_path + date_time
        self.model.save(export_path)
        print('Model saved to: ', export_path)
        
    def show_trainng_stats(self):
        print('Displaying Training Stats...')
        plt.figure()
        plt.subplot(2,1,1)
        plt.ylabel("Loss")
        plt.xlabel("Training Steps")
        plt.ylim([0,2])
        plt.plot(self.callbacks.batch_losses)
        
        plt.subplot(2,1,2)
        plt.ylabel("Accuracy")
        plt.xlabel("Training Steps")
        plt.ylim([0,1])
        plt.plot(self.callbacks.batch_acc)
        
    def predict_with_interal(self, image):
        print('Predicting...')
        """ Use current model held in memory """
        t1 = time.time()
        img = cv2.imread(image)
        img = cv2.resize(img,self.IMAGE_SHAPE)
        imgnp = np.reshape(img, [1,self.IMAGE_SHAPE[0], self.IMAGE_SHAPE[1], 3])
        prediction_aucs = self.model.predict(imgnp)
        prediction_index = np.argmax(prediction_aucs, axis=-1)
        prediction_label = self.class_names[prediction_index]
        t2 = time.time()
        print('Prediction:', prediction_label)
        print('AUC:', prediction_aucs[prediction_index])
        print('Time to Predict:', t2-t1, 'seconds')
        
        
class predict_with_exteranl:
    """ Load a saved model """
    def __init__(self, model_path):
        self.model_path = model_path
        
    def load_saved_model(self):
        print('Loading Model...')
        self.model = tf.keras.models.load_model(self.model_path, 
                                                custom_objects={'KerasLaysers':hub.KerasLayers})
    
    def make_prediction(self, image):
        print('Predicting...')
        t1 = time.time()
        img = cv2.imread(image)
        img = cv2.resize(img,self.IMAGE_SHAPE)
        imgnp = np.reshape(img, [1,self.IMAGE_SHAPE[0], self.IMAGE_SHAPE[1], 3])
        prediction_aucs = self.model.predict(imgnp)
        prediction_index = np.argmax(prediction_aucs, axis=-1)
        prediction_label = self.class_names[prediction_index]
        t2 = time.time()
        print('Prediction:', prediction_label)
        print('Time to Predict:', t2-t1)
    
    
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        