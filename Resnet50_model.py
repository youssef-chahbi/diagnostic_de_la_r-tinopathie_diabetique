# -*- coding: utf-8 -*-
"""
Created on Thu Dec 12 10:26:15 2019

@author: asus
"""

import datetime
import os, sys, shutil

# basics
import numpy as np
from numpy import loadtxt
import pandas as pd
from tqdm import tqdm

# charting
import matplotlib.pyplot as plt
from PIL import Image
import seaborn as sns

# metrics
from sklearn.metrics import confusion_matrix, cohen_kappa_score

# keras
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau,ModelCheckpoint
from tensorflow.keras import optimizers, applications,regularizers
from tensorflow.keras import layers
from tensorflow.keras.layers import Dense, Dropout, GlobalAveragePooling2D, Input
from tensorflow.keras import models
from tensorflow.keras.models import Model
from tensorflow.keras.models import load_model
from tensorflow.keras import optimizers
from tensorflow.keras.applications import ResNet50
import tensorflow as tf
print(tf.__version__)

# set file path variables
train_path = 'C:/Users/youss/Desktop/PFA/train_images/'
test_path = 'C:/Users/youss/Desktop/PFA/test_images/'


# load csv files with image file names and labels as pandas dataframes
train_data = pd.read_csv('C:/Users/youss/Desktop/PFA/train.csv')
test_data = pd.read_csv('C:/Users/youss/Desktop/PFA/test.csv')

# store the class information in some variables for convenience
class_labels = [0,1,2,3,4]
class_dict = {0:'No DR', 1:'Mild DR', 2:'Moderate DR', 3:'Severe DR', 4:'Proliferative DR'}
class_list = ['No DR', 'Mild DR', 'Moderate DR', 'Severe DR', 'Proliferative DR']

train_dir = 'C:/Users/asus/data_organized/train/'
validation_dir = 'C:/Users/asus/data_organized/val/'
test_dir = 'C:/Users/asus/data_organized/test/'

# CNN model parameters
BATCH_SIZE = 32
EPOCHS = 10
WARMUP_EPOCHS = 2
LEARNING_RATE = 1e-5
WARMUP_LEARNING_RATE = 1e-4
HEIGHT = 128
WIDTH = 128
COLORS = 3
N_CLASSES = 5
RLROP_PATIENCE = 3
DECAY_DROP = 0.5


# get all the images in the training directory and reshape and augment them
train_datagen = ImageDataGenerator(
      rescale=1/255,
      rotation_range=20,
      width_shift_range=0.2,   # removed these for time savings
      height_shift_range=0.2,
      shear_range=0.2,
      zoom_range=0.5,
      horizontal_flip=True,
      fill_mode='nearest')

train_generator = train_datagen.flow_from_directory(
        train_dir, 
        target_size=(HEIGHT, WIDTH), 
        batch_size= BATCH_SIZE,
        shuffle = True,
        class_mode= 'categorical') 


# get all the data in the validation directory and reshape them
val_datagen = ImageDataGenerator(rescale=1/255)

val_generator = val_datagen.flow_from_directory(
        validation_dir, 
        target_size=(HEIGHT, WIDTH),
        batch_size = BATCH_SIZE,
        class_mode= 'categorical')

# get all the data in the test directory and reshape them
test_generator = ImageDataGenerator(rescale=1/255).flow_from_directory(
        test_dir, 
        target_size=(HEIGHT, WIDTH), 
        batch_size = 1,
        class_mode= 'categorical',
        shuffle = False)


def create_model(input_shape, n_out):
    input_tensor = Input(shape=input_shape)
    base_model = applications.ResNet50(weights='imagenet', #à modifier
                                       include_top=False,
                                       input_tensor=input_tensor)
    for layer in base_model.layers:
        layer.trainable = False
    x = GlobalAveragePooling2D()(base_model.output)
    output = Dense(2300, activation='relu',name='output')(x)
    output = Dropout(0.25)(output)
    model_prim = Model(input_tensor, output)
    final_output = Dense(n_out, activation='softmax',  kernel_regularizer=regularizers.l2(0.01),name='final_output')(model_prim.output)
    model = Model(input_tensor, final_output)

    return model
#next step
model = create_model(input_shape=(HEIGHT, WIDTH, COLORS), n_out=N_CLASSES)
model.summary()


metric_list = ["accuracy"]
optimizer = optimizers.Adam(lr=WARMUP_LEARNING_RATE)
model.compile(optimizer=optimizer, loss="categorical_crossentropy",  metrics=metric_list)



rlrop = ReduceLROnPlateau(monitor='val_loss', mode='min', patience=RLROP_PATIENCE, factor=DECAY_DROP, min_lr=1e-7, verbose=1)

callback_list = [ rlrop]


# warm up training phase, only two epochs
STEP_SIZE_TRAIN = train_generator.n//train_generator.batch_size
STEP_SIZE_VALID = val_generator.n//val_generator.batch_size

history_warmup = model.fit_generator(generator=train_generator,
                              steps_per_epoch=STEP_SIZE_TRAIN,
                              validation_data=val_generator,
                              validation_steps=STEP_SIZE_VALID,
                              epochs=WARMUP_EPOCHS,
                              verbose=1).history

# create the model, use early stopping
for layer in model.layers:
        layer.trainable = True

rlrop = ReduceLROnPlateau(monitor='val_loss', mode='min', patience=RLROP_PATIENCE, factor=DECAY_DROP, min_lr=1e-7, verbose=1)

callback_list = [ rlrop]
optimizer = optimizers.Adam(lr=LEARNING_RATE)
model.compile(optimizer=optimizer, loss="categorical_crossentropy",  metrics=metric_list)


STEP_SIZE_TRAIN = train_generator.n//train_generator.batch_size
STEP_SIZE_VALID = val_generator.n//val_generator.batch_size
history_finetunning = model.fit_generator(generator=train_generator,
                              steps_per_epoch=STEP_SIZE_TRAIN,
                              validation_data=val_generator,
                              validation_steps=STEP_SIZE_VALID,
                              epochs=EPOCHS, # revert to: epochs=EPOCHS
                              callbacks=callback_list,
                              verbose=1).history
                                          
               
history = {'loss': history_warmup['loss'] + history_finetunning['loss'], 
           'val_loss': history_warmup['val_loss'] + history_finetunning['val_loss'], 
           'acc': history_warmup['acc'] + history_finetunning['acc'], 
           'val_acc': history_warmup['val_acc'] + history_finetunning['val_acc']}

sns.set_style("whitegrid")
fig, (ax1, ax2) = plt.subplots(2, 1, sharex='col', figsize=(20, 14))

ax1.plot(history['loss'], label='Train loss')
ax1.plot(history['val_loss'], label='Validation loss')
ax1.legend(loc='best')
ax1.set_title('Loss')

ax2.plot(history['acc'], label='Train Accuracy')
ax2.plot(history['val_acc'], label='Validation accuracy')
ax2.legend(loc='best')
ax2.set_title('Accuracy')

plt.xlabel('Epochs')
sns.despine()
plt.show()

#model.save('new_model_resnet.h5')

#function to make a tensor of an image
def preprocess_image(image_path, desired_size=128):
    im = Image.open(image_path)
    im = im.resize((desired_size, )*2, resample=Image.LANCZOS)
    
    return im

N = train_data.shape[0]

x_train = np.empty((N, 128, 128, 3), dtype=np.uint8)

for i, image_id in enumerate(tqdm(train_data['id_code'])):
    x_train[i, :, :, :] = preprocess_image(
        os.path.join(train_path + "/" + image_id + '.png')
    )

x_train = x_train/255
#use the model to generate predictions for all of the training images
start = datetime.datetime.now()
print('Started predicting at {}'.format(start))

train_prediction = model.predict([x_train])

end = datetime.datetime.now()
elapsed = end - start
print('Predicting took a total of {}'.format(elapsed))

# take the highest predicted probability for each image
train_predictions = [np.argmax(pred) for pred in train_prediction]

# look at how the model performed for each class
labels = ['0 - No DR', '1 - Mild', '2 - Moderate', '3 - Severe', '4 - Proliferative DR']
cnf_matrix = confusion_matrix(train_data['diagnosis'].astype('int'), train_predictions)
cnf_matrix_norm = cnf_matrix.astype('float') / cnf_matrix.sum(axis=1)[:, np.newaxis]
df_cm = pd.DataFrame(cnf_matrix_norm, index=labels, columns=labels)
plt.figure(figsize=(16, 7))
sns.heatmap(df_cm, annot=True, fmt='.2f', cmap="Blues")
plt.show()

model.save('resnet50_model.h5')

