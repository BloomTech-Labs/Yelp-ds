import numpy as np
import pandas as pd
import argparse
import datetime
import pickle
import gc
from keras.models import Model
from keras.layers import Input, Dense, LSTM, Embedding, Dropout
from keras.layers.merge import add
from keras.callbacks import ModelCheckpoint, EarlyStopping
from keras.optimizers import SGD
from keras.utils import to_categorical
from architectures import get_inject_model, get_merge_model, get_merge_add_model
model_factories = {"inject":get_inject_model,
                            "merge":get_merge_model,
                            "merge_add":get_merge_add_model}


argparser = argparse.ArgumentParser()
argparser.add_argument("--epochs", dest = "epochs", default= 20, type = int,
                 help="Indicate how many training epochs you want")
argparser.add_argument("--lr", dest = "lr", default= .01, type = float,
                 help="Learning rate for training")
argparser.add_argument("--datadir", dest = "datadir", type = str,
                 help = "Relative path where data binary files are stored ")
argparser.add_argument("--embeddingdir", dest = "embeddingdir", type = str,
                 help = "Relative path where the embedding matrix binary file is stored.")
argparser.add_argument("--modeldir", dest = "modeldir", type = str,
                 help = "Relative path where pickled models should be saved.")
argparser.add_argument("--historydir", dest = "historydir", type = str,
                 help = "Relative path where training hitory should be saved.")
argparser.add_argument("--patience", dest = "patience", type = int, default = 2,
                 help = "Early stopping patience.")
argparser.add_argument("--arch", dest = "arch", type = str, default = "inject",
                 help = "Batch size.")
argparser.add_argument("--trainable", dest = "trainable", type =str, default = "False",
                 help = "Whether to train word embeddings. First letter capitalized (python boolean)")

args = argparser.parse_args()

epochs = args.epochs

lr = args.lr

datadir = args.datadir

embeddingdir = args.embeddingdir

modeldir = args.modeldir

historydir = args.historydir

patience = args.patience

architecture = args.arch

model_factory = model_factories[architecture]

trainable = args.trainable == "True"


VOCAB_SIZE = 30212
EMBED_SIZE = 300



def load_npy(path):
    with open(path, "rb") as handle:
        arr = np.load(path)
    handle.close()
    return(arr)



today = datetime.datetime.now()
model_path = modeldir + 'model_%s-date_%d-%d-%d-%d-ep{epoch:03d}-loss{loss:.3f}_lr-%f_patience-%d.h5' % (
    architecture,today.month, today.day, today.hour, today.minute, lr, patience)

checkpoint = ModelCheckpoint(model_path, monitor='val_loss', verbose=1, save_best_only=False)

early_stopping = EarlyStopping(patience=patience)

print("Loading model and embedding matrix...")
embedding_matrix = load_npy(embeddingdir + "embedding_matrix.npy")

model = model_factory(embedding_matrix, trainable = trainable)
del embedding_matrix
gc.collect()
print("done.")
print()
print("Loading training data...")

y_train = load_npy(datadir + "y_train.npy")
y_train = y_train.reshape((-1,))
X_train_photos = load_npy(datadir + "X_train_photos.npy")
X_train_captions = load_npy(datadir + "X_train_captions.npy")
print("done.")
print()
print("Loading validation data...")
y_valid = load_npy(datadir + "y_valid.npy")
y_valid = y_valid.reshape((-1,))
X_valid_photos = load_npy(datadir + "X_valid_photos.npy")
X_valid_captions = load_npy(datadir + "X_valid_captions.npy")
NUM_EXAMPLES = X_train_photos.shape[0]

history = model.fit([X_train_photos, X_train_captions], y_train, epochs=epochs, verbose=1,
    callbacks=[checkpoint,early_stopping], 
    validation_data=([X_valid_photos, X_valid_captions], y_valid))


history_path = historydir + "history-%s-date_%d-%d-%d-%d.pkl" % (architecture, today.month, today.day, today.hour, today.minute)
with open(history_path, "wb") as handle:
    pickle.dump(history.history, handle)
handle.close()
