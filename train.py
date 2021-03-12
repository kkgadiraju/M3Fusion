import matplotlib
matplotlib.use('Agg')
import tensorflow as tf
import numpy as np
np.random.seed(100)
tf.random.set_seed(100)
gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
  try:
    for gpu in gpus:
      tf.config.experimental.set_memory_growth(gpu, True)
  except RuntimeError as e:
    print(e)
from tensorflow.keras.layers import Input, GRU, Dense, TimeDistributed, Concatenate, GlobalAveragePooling2D, Dropout, Conv2D, MaxPooling2D, BatchNormalization, Dot, Permute, Lambda
from tensorflow.keras.backend import squeeze
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.models import Model
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import numpy as np
from sklearn.metrics import classification_report
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, CSVLogger, TensorBoard
from time import time
import configparser, argparse, datetime, json, os
from tensorflow.keras.applications.inception_v3 import InceptionV3
from tensorflow.keras.applications.resnet50 import ResNet50
from tensorflow.keras.applications.densenet import DenseNet201
from tensorflow.keras.applications.vgg16 import VGG16
from tensorflow.keras.applications.vgg19 import VGG19
from tensorflow.keras.utils import plot_model
import visualize
import socket
import sys
import pandas as pd
import itertools

start_time_n = datetime.datetime.now()

start_time = start_time_n.strftime("%Y-%m-%d %H:%M:%S")



batch_size = 32
tr_path = #path_to_the_train_folder
val_path = #path_to_the_validation_folder
num_classes = # number of classes
learning_rate = # learning_rate
num_epochs = # number of epochs
temporal_dropout_rates = 0.4

target_size = (25, 25)

def spatial_loss():
    def sloss(y_true, y_pred):
        return tf.keras.losses.CategoricalCrossEntropy()(y_true, y_pred)


def temporal_loss():
    def tloss(y_true, y_pred):
        return tf.keras.losses.CategoricalCrossEntropy()(y_true, y_pred)
               
def combined_loss():
    def closs(y_true, y_pred):
        return tf.keras.losses.CategoricalCrossEntropy()(y_true, y_pred)



train_generator = # CUSTOM data generator for training 

val_generator = # CUSTOM data generator for validation

# NETWORK starts here

# spatial part starts here
spatial_input_layer = Input(batch_shape=(None, 25, 25, 3), name='spatial_input_layer')
            
conv_1 = Conv2D(filters=256, kernel_size=(7,7), activation='relu', input_shape=(25,25,3))(spatial_input_layer) # 256, 19, 19

bn_1 = BatchNormalization()(conv_1) # 256, 19, 19

pool_1 = MaxPooling2D(pool_size=(2,2), strides=2)(bn_1) # 256, 9, 9

conv_2 = Conv2D(filters=512, kernel_size=(3,3), activation='relu')(pool_1) # 512, 7, 7
            
bn_2 = BatchNormalization()(conv_2) # 512, 7, 7
            
conv_3 = Conv2D(filters=512, kernel_size=(3,3), activation='relu', padding='same')(bn_2)

bn_3 = BatchNormalization()(conv_3)
            
conv_concat_layer = Concatenate(axis=-1)([bn_2, bn_3])
            
conv_4 = Conv2D(filters=512, kernel_size=(1, 1), activation='relu')(conv_concat_layer)
            
bn_4 = BatchNormalization()(conv_4)
            
cnn_feat = GlobalAveragePooling2D()(conv_4)
            
spatial_predictions = Dense(num_classes, activation='softmax', name='spatial_preds')(cnn_feat)

# spatial part ends here

# temporal part starts here
temporal_input_layer = Input(batch_shape = (None, 23, 1), name='time_input_layer')

# add a GRU layer
gru_seq_output, gru_final_state = GRU(1024, input_shape=(23,1), dropout=temporal_dropout_rate, return_state=True, return_sequences=True)(temporal_input_layer)

v_a = TimeDistributed(Dense(1024, activation='tanh'))(gru_seq_output)
            
lambda_a = TimeDistributed(Dense(1, activation='softmax'))(v_a)

lambda_a_reshape = Permute([2,1])(lambda_a)

rnn_dot = Dot(axes=(2, 1))([lambda_a_reshape, gru_seq_output])

rnn_feat = Lambda(lambda y: squeeze(y, axis=1))(rnn_dot)


temporal_predictions = Dense(num_classes, activation='softmax', name='temporal_preds')(rnn_feat)
    
# temporal part ends here

# add a concatenation layer to combine output of spatial and temporal
final_merged = Concatenate()([cnn_feat, rnn_feat])
            
# and a softmax layer -- num_classes
merged_predictions = Dense(num_classes, activation='softmax', name='merged_preds')(final_merged)

optimizer = Adam(lr=learning_rate)

            
# this is the model we will train
model = Model(inputs=[spatial_input_layer, temporal_input_layer], outputs=[spatial_predictions, temporal_predictions, merged_predictions])

model.compile(loss={'spatial_preds':spatial_loss(), 'temporal_preds': temporal_loss(), 'merged_preds': combined_loss()}, optimizer=optimizer, loss_weights = [0.3, 0.3, 1.0], metrics=['accuracy']) 

## NN ends here

model_name = 'm3fusion.h5'

# generate a model by training 
history = model.fit(train_generator,  
		epochs=num_epochs,
		steps_per_epoch=num_train//batch_size,
	        validation_data= val_generator, 
		validation_steps=num_val//batch_size,
			verbose=1) 
