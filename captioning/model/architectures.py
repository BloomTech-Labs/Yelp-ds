import numpy as np
from keras.models import Model
from keras.layers import Input, Dense, LSTM, Embedding, Dropout, RepeatVector, Masking
from keras.layers.merge import concatenate, add
from keras.callbacks import ModelCheckpoint, EarlyStopping
from keras.optimizers import SGD
from keras.utils import to_categorical


VOCAB_SIZE = 30212
EMBED_SIZE = 300



def get_merge_add_model(embedding_matrix,trainable = True):
    inputs_photo = Input(shape = (4096,), name="Inputs-photo")
    dense = Dense(4096, activation = 'relu')(inputs_photo)
    drop1 = Dropout(0.5)(dense)
    dense1 = Dense(256, activation='relu')(drop1)
    
    inputs_caption = Input(shape=(15,), name = "Inputs-caption")
    embedding = Embedding(VOCAB_SIZE, EMBED_SIZE,
                          mask_zero = True, trainable = trainable,
                          weights=[embedding_matrix])(inputs_caption)
    drop2 = Dropout(0.5)(embedding)
    lstm1 = LSTM(256)(drop2)
    merged = add([dense1, lstm1])
    dense2 = Dense(256, activation='relu')(merged)
    outputs = Dense(VOCAB_SIZE, activation='softmax')(dense2)
    model = Model(inputs=[inputs_photo, inputs_caption], outputs=outputs)
    sgd = SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True)
    model.compile(loss='sparse_categorical_crossentropy', optimizer=sgd)
    return(model)

def get_inject_model(embedding_matrix, trainable = True):
    inputs_photo = Input(shape = (4096,), name="Inputs-photo")
    drop1 = Dropout(0.5)(inputs_photo)
    dense1 = Dense(EMBED_SIZE, activation='relu')(drop1)
    cnn_feats = Masking()(RepeatVector(1)(dense1))

    inputs_caption = Input(shape=(15,), name = "Inputs-caption")
    embedding = Embedding(VOCAB_SIZE, EMBED_SIZE,
                    mask_zero = True, trainable = False,
                    weights=[embedding_matrix])(inputs_caption)
    drop2 = Dropout(0.5)(embedding)
    merged = concatenate([cnn_feats, drop2], axis=1)
    lstm_layer = LSTM(units=EMBED_SIZE,
                      input_shape=(15 + 1, EMBED_SIZE),  
                      return_sequences=False)(merged)
    drop3 = Dropout(0.5)(lstm_layer)

    outputs = Dense(units=VOCAB_SIZE,activation='softmax')(drop3)

    model = Model(inputs=[inputs_photo, inputs_caption], outputs=outputs)
    sgd = SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True)
    model.compile(loss='sparse_categorical_crossentropy', optimizer=sgd)
    return(model)
