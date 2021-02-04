# -*- coding: utf-8 -*-
"""
Created on Mon Jul 13 22:59:33 2020

@author: youss
"""


import cv2
import tensorflow as tf

CATEGORIES = ["No DR","Mild DR","Moderate DR","Severe DR","Proliferative DR"]


def prepare(filepath):
    IMG_SIZE = 255  # 50 in txt-based
    img_array = cv2.imread(filepath, cv2.IMREAD_GRAYSCALE)
    new_array = cv2.resize(img_array, (IMG_SIZE, IMG_SIZE))
    return new_array.reshape(-1, IMG_SIZE, IMG_SIZE, 1)


model = tf.keras.models.load_model("resnet50_model.h5")

prediction = model.predict([prepare('00cc2b75cddd.png')])
print(prediction)  # will be a list in a list.
print(CATEGORIES[int(prediction[0][0][0][0])])