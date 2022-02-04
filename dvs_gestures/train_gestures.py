#!/bin/python
#-----------------------------------------------------------------------------
#-----------------------------------------------------------------------------
import sys
# insert at 1, 0 is the script path (or '' in REPL)
sys.path.insert(1, '../model')
sys.path.insert(1, '../')
from train_hybrid_vae_guided_base import Guide, HybridGuidedVAETrainer
import matplotlib
matplotlib.use('Agg')
from hybrid_beta_vae import Reshape, VAE
from decolle.utils import parse_args, train, test, accuracy, save_checkpoint, load_model_from_checkpoint, prepare_experiment, write_stats, cross_entropy_one_hot
#from utils import save_checkpoint, load_model_from_checkpoint
import datetime, os, socket, tqdm
import numpy as np
import torch
from torch import nn
import importlib
from itertools import chain
import matplotlib.pyplot as plt
from torchvision.utils import make_grid
from decolle.utils import MultiOpt
from torchneuromorphic import transforms
from tqdm import tqdm
import math
import sys
from utils import generate_process_target
import argparse

epsilon = sys.float_info.epsilon
np.set_printoptions(precision=4)


if __name__=="__main__":
    # parse args for params file, and dataset_path, that should take care of pretty much everything I think...
    # I might need to add something for lights, but I'll figure it out

    parser = argparse.ArgumentParser('HGVAE')
    
    parser.add_argument('--params-file', default = '../parameters/params_hybridvae_dvsgestures-guidedbeta-noaug-Copy1.yml', type=str, help='Path to the parameter config file.') 
    parser.add_argument('--data-file', default = '/home/kennetms/Documents/data/dvs_gestures.hdf5', type=str, help='Path to the file the data is in, should be hdf5 compatible with torchneuromorphic.')
    args = parser.parse_args()
    
    param_file = args.params_file #'parameters/params_hybridvae_dvsgestures-guidedbeta-noaug-Copy1.yml'
    dataset_path = args.data_file #'/home/kennetms/Documents/data/dvs_gestures.hdf5'
    
    HGVAE = HybridGuidedVAETrainer(param_file, dataset_path)
    
    HGVAE.train_eval_plot_loop()