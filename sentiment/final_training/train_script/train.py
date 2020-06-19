import argparse
import os
import pickle

import ktrain
from ktrain import text
import tensorflow as tf
mirrored_strategy = tf.distribute.MirroredStrategy()

def _load_training_data(base_dir):
    trn = pickle.load(open(os.path.join(base_dir, 'trn.p'), "rb" ))
    return trn

def _load_testing_data(base_dir):
    val = pickle.load(open(os.path.join(base_dir, 'val.p'), "rb" ))
    return val

def _load_preproc(base_dir):
    preproc = pickle.load(open(os.path.join(base_dir, 'preproc.p'), "rb" ))
    return preproc

def _parse_args():

    parser = argparse.ArgumentParser()

    # hyperparameters sent by the client are passed as command-line arguments to the script.
    parser.add_argument('--epochs', type=int, default=3)
    parser.add_argument('--batch_size', type=int, default=128)
    parser.add_argument('--learning_rate', type=float, default=2e-5)
    parser.add_argument('--model_name', type=str, default='distilbert')

    # input data and model directories
    
    # checkpoints
    # s3 path
    parser.add_argument('--model_dir', type=str)
    
    # saved model
    # local
    parser.add_argument('--sm-model-dir', type=str, default=os.environ.get('SM_MODEL_DIR'))
    parser.add_argument('--train', type=str, default=os.environ.get('SM_CHANNEL_TRAIN'))
    parser.add_argument('--test', type=str, default=os.environ.get('SM_CHANNEL_TEST'))
    parser.add_argument('--preproc', type=str, default=os.environ.get('SM_CHANNEL_PREPROC'))
    

    return parser.parse_known_args()

if __name__ =='__main__':

    args, unknown = _parse_args()
    
    trn = _load_training_data(args.train)
    val = _load_testing_data(args.test)
    preproc = _load_preproc(args.preproc)
    
    with mirrored_strategy.scope():
        model = text.text_regression_model(args.model_name, train_data=trn, preproc=preproc)
    
    learner = ktrain.get_learner(model,
                                 train_data=trn, 
                                 val_data=val, 
                                 batch_size=args.batch_size)
    
    learner.autofit(args.learning_rate, args.epochs, checkpoint_folder=args.model_dir)
    
    # learner.validate(val_data=val)
    
    predictor = ktrain.get_predictor(learner.model, preproc)
    
    predictor.save(args.sm_model_dir)