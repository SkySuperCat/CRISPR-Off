import os
import numpy as np
import tensorflow as tf
from keras_layer_normalization import LayerNormalization
from keras_multi_head import MultiHeadAttention
import tensorflow.keras.layers
from tensorflow.keras.models import model_from_json, load_model
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Embedding,MaxPool2D,Conv1D,Conv2D, Bidirectional, LSTM, Activation, Concatenate, AveragePooling1D, MaxPool1D, BatchNormalization, Attention, GlobalAveragePooling1D, GlobalMaxPool1D
from tensorflow.keras.initializers import VarianceScaling
from tensorflow.keras.utils import to_categorical
from sklearn.metrics import f1_score, precision_score, recall_score, accuracy_score, roc_auc_score, average_precision_score
from tensorflow.keras.initializers import glorot_normal
from keras_pos_embd import TrigPosEmbedding

import time
import shutil

from tensorflow.python.keras.layers.core import Reshape
from tensorflow.python.keras.models import *
from tensorflow.python.keras.layers import multiply
from tensorflow.python.keras.layers.core import Dense, Dropout, Flatten


VOCAB_SIZE = 16
EMBED_SIZE = 90
MAXLEN = 23
seed = 123

def conv2d_bn(x,
              filters,
              kernel_size,
              strides=1,
              padding='same',
              activation='relu',
              use_bias=True,
              name=None, trainable=True):

    x = Conv2D(filters,
                      kernel_size,
                      strides=strides,
                      padding=padding,
                      use_bias=use_bias,
                      kernel_initializer=glorot_normal(seed=seed),
                      name=name, trainable=trainable)(x)


    if activation is not None:
        ac_name = None if name is None else name + '_ac'
        x = Activation(activation, name=ac_name)(x)
    return x

class PositionalEncoding(tf.keras.layers.Layer):
    def __init__(self, sequence_len=None, embedding_dim=None,**kwargs):
        super(PositionalEncoding, self).__init__()
        self.sequence_len = sequence_len
        self.embedding_dim = embedding_dim

    def call(self, x):

        position_embedding = np.array([
            [pos / np.power(10000, 2. * i / self.embedding_dim) for i in range(self.embedding_dim)]
            for pos in range(self.sequence_len)])

        position_embedding[:, 0::2] = np.sin(position_embedding[:, 0::2])  # dim 2i
        position_embedding[:, 1::2] = np.cos(position_embedding[:, 1::2])  # dim 2i+1
        position_embedding = tf.cast(position_embedding, dtype=tf.float32)

        return position_embedding+x
    def get_config(self):
        config = super().get_config().copy()
        config.update({
            'sequence_len' : self.sequence_len,
            'embedding_dim' : self.embedding_dim,
        })
        return config

def transformIO(xtrain, xtest, ytrain, ytest,xval,yval,seq_len , coding_dim, num_classes):
    xtrain = xtrain.reshape(xtrain.shape[0], 1,seq_len, coding_dim)
    xtest = xtest.reshape(xtest.shape[0],1,seq_len, coding_dim)
    xval = xval.reshape(xval.shape[0], 1, seq_len, coding_dim)
    input_shape = (1,seq_len, coding_dim)
    xtrain = xtrain.astype('float32')
    xtest = xtest.astype('float32')
    xval = xval.astype('float32')
    print('xtrain shape:', xtrain.shape)
    print(xtrain.shape[0], 'train samples')
    print(xtest.shape[0], 'test samples')
    print(xval.shape[0], 'val samples')

    ytrain = to_categorical(ytrain, num_classes)
    ytest = to_categorical(ytest, num_classes)
    yval = to_categorical(yval, num_classes)
    return xtrain, xtest, ytrain, ytest,xval,yval, input_shape

def CRISPR_Net_transformIO(xtrain, xtest, ytrain, ytest,xval,yval,seq_len , coding_dim, num_classes):
    xtrain = xtrain.reshape(xtrain.shape[0], 1,seq_len, coding_dim)
    xtest = xtest.reshape(xtest.shape[0],1,seq_len, coding_dim)
    xval = xval.reshape(xval.shape[0], 1, seq_len, coding_dim)
    input_shape = (1,seq_len, coding_dim)
    xtrain = xtrain.astype('float32')
    xtest = xtest.astype('float32')
    xval = xval.astype('float32')
    print('xtrain shape:', xtrain.shape)
    print(xtrain.shape[0], 'train samples')
    print(xtest.shape[0], 'test samples')
    print(xval.shape[0], 'val samples')

    ytrain = to_categorical(ytrain, num_classes)
    ytest = to_categorical(ytest, num_classes)
    yval = to_categorical(yval, num_classes)
    return xtrain, xtest, ytrain, ytest,xval,yval, input_shape

def cnn_std_transformIO(xtrain, xtest, ytrain, ytest,xval,yval,seq_len , coding_dim, num_classes):
    xtrain = xtrain.reshape(xtrain.shape[0], 1,seq_len, coding_dim)
    xtest = xtest.reshape(xtest.shape[0],1,seq_len, coding_dim)
    xval = xval.reshape(xval.shape[0], 1, seq_len, coding_dim)
    input_shape = (1,seq_len, coding_dim)
    xtrain = xtrain.astype('float32')
    xtest = xtest.astype('float32')
    xval = xval.astype('float32')
    print('xtrain shape:', xtrain.shape)
    print(xtrain.shape[0], 'train samples')
    print(xtest.shape[0], 'test samples')
    print(xval.shape[0], 'xval samples')

    ytrain = to_categorical(ytrain, num_classes)
    ytest = to_categorical(ytest, num_classes)
    yval = to_categorical(yval, num_classes)
    return xtrain, xtest, ytrain, ytest,xval,yval, input_shape

def offt_transformIO(xtrain, xtest, ytrain, ytest ,xval,yval, num_classes):
    xtrain = xtrain.astype('float32')
    xtest = xtest.astype('float32')
    xval = xval.astype('float32')
    print('xtrain shape:', xtrain.shape)
    print(xtrain.shape[0], 'train samples')
    print(xtest.shape[0], 'test samples')
    print(xval.shape[0], 'val samples')

    ytrain = to_categorical(ytrain, num_classes)
    ytest = to_categorical(ytest, num_classes)
    yval = to_categorical(yval, num_classes)
    return xtrain, xtest, ytrain, ytest,xval,yval


def f1_metric(y_true, y_pred):
    true_positives = tf.keras.backend.sum(tf.keras.backend.round(tf.keras.backend.clip(y_true * y_pred, 0, 1)))
    possible_positives = tf.keras.backend.sum(tf.keras.backend.round(tf.keras.backend.clip(y_true, 0, 1)))
    predicted_positives = tf.keras.backend.sum(tf.keras.backend.round(tf.keras.backend.clip(y_pred, 0, 1)))
    precision = true_positives / (predicted_positives + tf.keras.backend.epsilon())
    recall = true_positives / (possible_positives + tf.keras.backend.epsilon())
    f1_val = 2 * (precision * recall) / (precision + recall + tf.keras.backend.epsilon())
    return f1_val


def CnnCrispr(embedding_weights,test_ds,resampled_steps_per_epoch,resampled_ds,xtrain, ytrain, xtest, ytest,  num_classes, batch_size, epochs, callbacks, saved_prefix, retrain=False):
    VOCAB_SIZE = 16
    EMBED_SIZE = 100
    maxlen = 23
    if retrain or not os.path.exists('saved_model/'+'{}CnnCrispr.h5'.format(saved_prefix)):
        input = Input(shape=(23,))
        embedding = Embedding(VOCAB_SIZE, EMBED_SIZE, input_length=maxlen,
                            weights=[embedding_weights],
                            trainable=True)(input)
        bi_lstm = Bidirectional(LSTM(40, return_sequences=True))(embedding)
        bi_lstm_relu = Activation('relu')(bi_lstm)

        conv1 = Conv1D(10, (5))(bi_lstm_relu)
        conv1_relu = Activation('relu')(conv1)
        conv1_batch = BatchNormalization()(conv1_relu)

        conv2 = Conv1D(20, (5))(conv1_batch)
        conv2_relu = Activation('relu')(conv2)
        conv2_batch = BatchNormalization()(conv2_relu)

        conv3 = Conv1D(40, (5))(conv2_batch)
        conv3_relu = Activation('relu')(conv3)
        conv3_batch = BatchNormalization()(conv3_relu)

        conv4 = Conv1D(80, (5))(conv3_batch)
        conv4_relu = Activation('relu')(conv4)
        conv4_batch = BatchNormalization()(conv4_relu)

        conv5 = Conv1D(100, (5))(conv4_batch)
        conv5_relu = Activation('relu')(conv5)
        conv5_batch = BatchNormalization()(conv5_relu)

        flat = Flatten()(conv5_batch)
        drop = Dropout(0.3)(flat)
        dense = Dense(20)(drop)
        dense_relu = Activation('relu')(dense)
        prediction = Dense(2, activation='softmax', name='main_output')(dense_relu)
        model = Model(input, prediction)
        model.compile(tf.keras.optimizers.Adam(learning_rate=0.001), loss=tf.keras.losses.CategoricalCrossentropy(), metrics=['accuracy', f1_metric])
        history_model = model.fit(
            resampled_ds,
            batch_size=batch_size,
            epochs=epochs,
            verbose=1,
            validation_data=test_ds,
            steps_per_epoch=resampled_steps_per_epoch,
            callbacks=callbacks
        )
        model.save('saved_model/'+'{}CnnCrispr.h5'.format(saved_prefix))
    else:
        model = load_model('saved_model/'+'{}CnnCrispr.h5'.format(saved_prefix))
    return model


def cnn_std(test_ds,resampled_steps_per_epoch,resampled_ds,xtrain, ytrain, xtest, ytest, num_classes, batch_size, epochs, callbacks, saved_prefix, retrain=False):
    if retrain or not os.path.exists('saved_model/'+'{}cnn_std.h5'.format(saved_prefix)):
        input = Input(shape=(1, 23, 4))
        conv_1 = Conv2D(10, (1, 1), padding='same', activation='relu')(input)
        conv_2 = Conv2D(10, (1, 2), padding='same', activation='relu')(input)
        conv_3 = Conv2D(10, (1, 3), padding='same', activation='relu')(input)
        conv_4 = Conv2D(10, (1, 5), padding='same', activation='relu')(input)

        conv_output = tensorflow.keras.layers.concatenate([conv_1, conv_2, conv_3, conv_4])

        bn_output = BatchNormalization()(conv_output)

        pooling_output = MaxPool2D(pool_size=(1, 5), strides=None, padding='valid')(bn_output)

        flatten_output = Flatten()(pooling_output)

        x = Dense(100, activation='relu')(flatten_output)
        x = Dense(23, activation='relu')(x)
        x = Dropout(rate=0.15)(x)

        output = Dense(2, activation="softmax")(x)

        model = Model(inputs=[input], outputs=[output])
        model.compile(tf.keras.optimizers.Adam(learning_rate=0.001), loss=tf.keras.losses.BinaryCrossentropy(),
                      metrics=['accuracy', f1_metric])
        history_model = model.fit(
            resampled_ds,
            batch_size=batch_size,
            epochs=epochs,
            verbose=1,
            validation_data=test_ds,
            steps_per_epoch=resampled_steps_per_epoch,
            callbacks=callbacks
        )
        model.save('saved_model/'+'{}cnn_std.h5'.format(saved_prefix))
    else:
        model = load_model('saved_model/'+'{}cnn_std.h5'.format(saved_prefix))
    return model


def crispr_ip(test_ds,resampled_steps_per_epoch,resampled_ds,xtrain, ytrain, xtest, ytest, input_shape, num_classes, batch_size, epochs, callbacks, saved_prefix, retrain=False):
    if retrain or not os.path.exists('saved_model/'+'{}crispr_ip.h5'.format(saved_prefix)):
        initializer = VarianceScaling(mode='fan_avg', distribution='uniform')
        input_value = Input(shape=input_shape)
        conv_1_output = Conv2D(60, (1, input_shape[-1]), padding='valid', data_format='channels_first',
                               kernel_initializer=initializer)(input_value)
        conv_1_output_reshape = Reshape(tuple([x for x in conv_1_output.shape.as_list() if x != 1 and x is not None]))(
            conv_1_output)
        conv_1_output_reshape2 = tf.transpose(conv_1_output_reshape, perm=[0, 2, 1])
        conv_1_output_reshape_average = AveragePooling1D(data_format='channels_first')(conv_1_output_reshape2)
        conv_1_output_reshape_max = MaxPool1D(data_format='channels_first')(conv_1_output_reshape2)
        bidirectional_1_output = Bidirectional(
            LSTM(30, return_sequences=True, dropout=0.25, kernel_initializer=initializer))(
            Concatenate(axis=-1)([conv_1_output_reshape_average, conv_1_output_reshape_max]))
        attention_1_output = Attention()([bidirectional_1_output, bidirectional_1_output])
        average_1_output = GlobalAveragePooling1D(data_format='channels_last')(attention_1_output)
        max_1_output = GlobalMaxPool1D(data_format='channels_last')(attention_1_output)
        concat_output = Concatenate(axis=-1)([average_1_output, max_1_output])
        flatten_output = Flatten()(concat_output)
        linear_1_output = BatchNormalization()(
            Dense(200, activation='relu', kernel_initializer=initializer)(flatten_output))
        linear_2_output = Dense(100, activation='relu', kernel_initializer=initializer)(linear_1_output)
        linear_2_output_dropout = Dropout(0.9)(linear_2_output)
        linear_3_output = Dense(num_classes, activation='softmax', kernel_initializer=initializer)(
            linear_2_output_dropout)
        model = Model(input_value, linear_3_output)
        model.compile(tf.keras.optimizers.Adam(learning_rate=0.001), loss=tf.keras.losses.CategoricalCrossentropy(), metrics=['accuracy', f1_metric])
        history_model = model.fit(
            resampled_ds,
            batch_size=batch_size,
            epochs=epochs,
            verbose=1,
            validation_data=test_ds,
            steps_per_epoch=resampled_steps_per_epoch,
            callbacks=callbacks
        )
        model.save('saved_model/'+'{}crispr_ip.h5'.format(saved_prefix))
    else:
        model = load_model('saved_model/'+'{}crispr_ip.h5'.format(saved_prefix))
    return model


def CRISPR_Net_model(test_ds,resampled_steps_per_epoch,resampled_ds,xtrain, ytrain, xtest, ytest, input_shape, num_classes, batch_size, epochs, callbacks, saved_prefix, retrain=False):
    roc_auc = 0
    pr_auc = 0
    if retrain or not os.path.exists('saved_model/'+'{}CRISPR_Net.h5'.format(saved_prefix)):
        inputs = Input(shape=(1, 23, 7), name='main_input')
        branch_0 = conv2d_bn(inputs, 10, (1, 1))
        branch_1 = conv2d_bn(inputs, 10, (1, 2))
        branch_2 = conv2d_bn(inputs, 10, (1, 3))
        branch_3 = conv2d_bn(inputs, 10, (1, 5))
        branches = [inputs, branch_0, branch_1, branch_2, branch_3]

        mixed = Concatenate(axis=-1)(branches)
        mixed = Reshape((23, 47))(mixed)
        blstm_out = Bidirectional(LSTM(15, kernel_initializer=glorot_normal(seed=seed),return_sequences=True, input_shape=(23, 46), name="LSTM_out"))(mixed)
        blstm_out = Flatten()(blstm_out)
        x = Dense(80, kernel_initializer=glorot_normal(seed=seed),activation='relu')(blstm_out)
        x = Dense(20, kernel_initializer=glorot_normal(seed=seed),activation='relu')(x)
        x = Dropout(0.35,seed=seed)(x)
        prediction = Dense(2,kernel_initializer=glorot_normal(seed=seed), activation='softmax', name='main_output')(x)
        model = Model(inputs, prediction)
        model.compile(tf.keras.optimizers.Adam(learning_rate=0.001), loss=tf.keras.losses.CategoricalCrossentropy(), metrics=['accuracy', f1_metric])
        history_model = model.fit(
            resampled_ds,
            batch_size=batch_size,
            epochs=epochs,
            verbose=1,
            validation_data=test_ds,
            steps_per_epoch=resampled_steps_per_epoch,
            callbacks=callbacks
        )
        model.save('saved_model/'+'{}CRISPR_Net.h5'.format(saved_prefix))
    else:
        model = load_model('saved_model/'+'{}CRISPR_Net.h5'.format(saved_prefix))
    return model


def crisprDNT(test_ds,resampled_steps_per_epoch,resampled_ds,xtrain, ytrain, xtest, ytest, input_shape, num_classes, batch_size, epochs, callbacks, saved_prefix, retrain=False):
    if retrain or not os.path.exists('saved_model/'+'{}crisprDNT.h5'.format(saved_prefix)):
        initializer = VarianceScaling(mode='fan_avg', distribution='uniform')
        input_value = Input(shape=input_shape)
        conv_1_output = Conv2D(64, (1, input_shape[-1]), activation='relu', padding='valid', data_format='channels_first',
                               kernel_initializer=initializer)(input_value)
        conv_1_output = BatchNormalization()(conv_1_output)
        conv_1_output_reshape = Reshape(tuple([x for x in conv_1_output.shape.as_list() if x != 1 and x is not None]))(
            conv_1_output)
        conv_1_output_reshape2 = tf.transpose(conv_1_output_reshape, perm=[0, 2, 1])
        conv_1_output_reshape_average = AveragePooling1D(data_format='channels_first')(conv_1_output_reshape2)
        conv_1_output_reshape_max = MaxPool1D(data_format='channels_first')(conv_1_output_reshape2)
        input_value1 = Reshape((23,input_shape[-1]))(input_value)
        bidirectional_1_output = Bidirectional(
            LSTM(32, return_sequences=True, dropout=0.2,activation='relu',kernel_initializer=initializer))(
            Concatenate(axis=-1)([input_value1,conv_1_output_reshape_average, conv_1_output_reshape_max]))


        bidirectional_1_output_ln = LayerNormalization()(bidirectional_1_output)

        pos_embedding = PositionalEncoding(sequence_len=23, embedding_dim=64)(bidirectional_1_output_ln)

        attention_1_output = MultiHeadAttention(head_num=8)([pos_embedding, pos_embedding, pos_embedding])
        residual1 = attention_1_output + pos_embedding
        laynorm1 = LayerNormalization()(residual1)
        linear1 = Dense(512, activation='relu', kernel_initializer=initializer)(laynorm1)
        linear2 = Dense(64, activation='relu', kernel_initializer=initializer)(linear1)
        residual2 = laynorm1 + linear2
        laynorm2 = LayerNormalization()(residual2)
        attention_2_output = MultiHeadAttention(head_num=8)([laynorm2, laynorm2, laynorm2])
        residual3 = attention_2_output + laynorm2
        laynorm3 = LayerNormalization()(residual3)
        linear3 = Dense(512, activation='relu', kernel_initializer=initializer)(laynorm3)
        linear4 = Dense(64, activation='relu', kernel_initializer=initializer)(linear3)
        residual4 = laynorm3 + linear4
        laynorm4 = LayerNormalization()(residual4)
        flatten_output = Flatten()(laynorm4)
        linear_1_output = (Dense(256, activation='relu', kernel_initializer=initializer)(flatten_output))
        linear_2_output = Dense(64, activation='relu', kernel_initializer=initializer)(linear_1_output)
        linear_2_output_dropout = Dropout(0.25)(linear_2_output)
        linear_3_output = Dense(2, activation='softmax', kernel_initializer=initializer)(linear_2_output_dropout)
        model = Model(input_value, linear_3_output)
        model.compile(tf.keras.optimizers.Adam(learning_rate=0.001), loss=tf.keras.losses.CategoricalCrossentropy(), metrics=['accuracy', f1_metric])#Adam是0.001，SGD是0.01
        history_model = model.fit(
            resampled_ds,
            batch_size=batch_size,
            epochs=epochs,
            verbose=1,
            validation_data=test_ds,
            steps_per_epoch=resampled_steps_per_epoch,
            callbacks=callbacks
        )
        model.save('saved_model/'+'{}crisprDNT.h5'.format(saved_prefix))
    else:
        model = load_model('saved_model/'+'{}crisprDNT.h5'.format(saved_prefix),custom_objects={'PositionalEncoding': PositionalEncoding,'MultiHeadAttention':MultiHeadAttention})
    return model


