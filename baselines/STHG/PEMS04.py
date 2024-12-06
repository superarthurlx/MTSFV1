import os
import sys
import torch
from easydict import EasyDict
sys.path.append(os.path.abspath(__file__ + '/../../..'))

from basicts.metrics import masked_mae, masked_mape, masked_rmse
from basicts.data import TimeSeriesForecastingDataset
from basicts.runners import SimpleTimeSeriesForecastingRunner
from basicts.scaler import ZScoreScaler
from basicts.utils import get_regular_settings, load_adj

<<<<<<<< HEAD:baselines/STHG/PEMS04.py
from .arch import STHG
from .runner import STHGRunner
========
from .arch import BigSTPreprocess
from .runner import BigSTPreprocessRunner
>>>>>>>> b85da4e198a180c714e70cdd299d566e5ba481f9:baselines/BigST/PreprocessPEMS08.py

############################## Hot Parameters ##############################
# Dataset & Metrics configuration
DATA_NAME = 'PEMS08'  # Dataset name
regular_settings = get_regular_settings(DATA_NAME)
INPUT_LEN = 2016 
OUTPUT_LEN = 12
TRAIN_VAL_TEST_RATIO = regular_settings['TRAIN_VAL_TEST_RATIO']  # Train/Validation/Test split ratios
NORM_EACH_CHANNEL = regular_settings['NORM_EACH_CHANNEL'] # Whether to normalize each channel of the data
RESCALE = regular_settings['RESCALE'] # Whether to rescale the data
NULL_VAL = regular_settings['NULL_VAL'] # Null value in the data
# Model architecture and parameters
<<<<<<<< HEAD:baselines/STHG/PEMS04.py
MODEL_ARCH = STHG

MODEL_PARAM = {
    "num_nodes": 307,
    "input_len": INPUT_LEN,
    "input_dim": 1,
    "embed_dim": 32,
    "embed_sparsity": 0.5,
    "embed_std": 0.01,
    "constant": 10,
    "output_len": OUTPUT_LEN,
    "num_layer": 7,
    # -------node&time-----------
    "time_of_day_size": 288,
    "day_of_week_size": 7,
    # -------kernel-------
    "if_hgnn": True,
    "if_cycle": True,
    "if_revin": True,
    "if_temporal": True,
    "n_kernel": 20, 
    "topk": 2,
    # -------Bernstein appro-------
    "use_bern": True
========
MODEL_ARCH = BigSTPreprocess
adj_mx, _ = load_adj("datasets/" + DATA_NAME +
                     "/adj_mx.pkl", "doubletransition")
MODEL_PARAM = {
    "num_nodes": 170,
    "in_dim": 3,
    "dropout": 0.3,
    "input_length": INPUT_LEN,
    "output_length": OUTPUT_LEN,
    "nhid": 32,
    "tiny_batch_size": 64,

>>>>>>>> b85da4e198a180c714e70cdd299d566e5ba481f9:baselines/BigST/PreprocessPEMS08.py
}

NUM_EPOCHS = 100

############################## General Configuration ##############################
CFG = EasyDict()
# General settings
CFG.DESCRIPTION = 'An Example Config'
CFG.GPU_NUM = 1 # Number of GPUs to use (0 for CPU mode)
# Runner
<<<<<<<< HEAD:baselines/STHG/PEMS04.py
CFG.RUNNER = STHGRunner
========
CFG.RUNNER = BigSTPreprocessRunner
>>>>>>>> b85da4e198a180c714e70cdd299d566e5ba481f9:baselines/BigST/PreprocessPEMS08.py

############################## Environment Configuration ##############################
CFG.ENV = EasyDict() # Environment settings. Default: None
CFG.ENV.SEED = 0 # Random seed. Default: None

############################## Dataset Configuration ##############################
CFG.DATASET = EasyDict()
# Dataset settings
CFG.DATASET.NAME = DATA_NAME
CFG.DATASET.TYPE = TimeSeriesForecastingDataset
CFG.DATASET.PARAM = EasyDict({
    'dataset_name': DATA_NAME,
    'train_val_test_ratio': TRAIN_VAL_TEST_RATIO,
    'input_len': INPUT_LEN,
    'output_len': OUTPUT_LEN,
    # 'mode' is automatically set by the runner
})

############################## Scaler Configuration ##############################
CFG.SCALER = EasyDict()
# Scaler settings
CFG.SCALER.TYPE = ZScoreScaler # Scaler class
CFG.SCALER.PARAM = EasyDict({
    'dataset_name': DATA_NAME,
    'train_ratio': TRAIN_VAL_TEST_RATIO[0],
    'norm_each_channel': NORM_EACH_CHANNEL,
    'rescale': RESCALE,
})

############################## Model Configuration ##############################
CFG.MODEL = EasyDict()
# Model settings
CFG.MODEL.NAME = MODEL_ARCH.__name__
CFG.MODEL.ARCH = MODEL_ARCH
CFG.MODEL.PARAM = MODEL_PARAM
CFG.MODEL.FORWARD_FEATURES = [0, 1, 2]
CFG.MODEL.TARGET_FEATURES = [0]

############################## Metrics Configuration ##############################

CFG.METRICS = EasyDict()
# Metrics settings
CFG.METRICS.FUNCS = EasyDict({
                                'MAE': masked_mae,
                                'MAPE': masked_mape,
                                'RMSE': masked_rmse,
                            })
CFG.METRICS.TARGET = 'MAE'
CFG.METRICS.NULL_VAL = NULL_VAL

############################## Training Configuration ##############################
CFG.TRAIN = EasyDict()
CFG.TRAIN.NUM_EPOCHS = NUM_EPOCHS
CFG.TRAIN.CKPT_SAVE_DIR = os.path.join(
    'checkpoints',
    MODEL_ARCH.__name__,
    '_'.join([DATA_NAME, str(CFG.TRAIN.NUM_EPOCHS), str(INPUT_LEN), str(OUTPUT_LEN)])
)
CFG.TRAIN.LOSS = masked_mae
# Optimizer settings
CFG.TRAIN.OPTIM = EasyDict()
CFG.TRAIN.OPTIM.TYPE = "AdamW"
CFG.TRAIN.OPTIM.PARAM = {
    "lr": 0.005,
    "weight_decay": 0.01,
}
# Learning rate scheduler settings
CFG.TRAIN.LR_SCHEDULER = EasyDict()
CFG.TRAIN.LR_SCHEDULER.TYPE = "MultiStepLR"
CFG.TRAIN.LR_SCHEDULER.PARAM = {
    "milestones": [1, 30, 50, 80],
    "gamma": 0.5
}
# Train data loader settings
CFG.TRAIN.DATA = EasyDict()
CFG.TRAIN.DATA.BATCH_SIZE = 1
CFG.TRAIN.DATA.SHUFFLE = True
# Gradient clipping settings
CFG.TRAIN.CLIP_GRAD_PARAM = {
    "max_norm": 5.0
}

############################## Validation Configuration ##############################
CFG.VAL = EasyDict()
CFG.VAL.INTERVAL = 1
CFG.VAL.DATA = EasyDict()
CFG.VAL.DATA.BATCH_SIZE = 1

############################## Test Configuration ##############################
CFG.TEST = EasyDict()
CFG.TEST.INTERVAL = 1
CFG.TEST.DATA = EasyDict()
CFG.TEST.DATA.BATCH_SIZE = 1

############################## Evaluation Configuration ##############################

CFG.EVAL = EasyDict()

# Evaluation parameters
CFG.EVAL.HORIZONS = [3, 6, 12] # Prediction horizons for evaluation. Default: []
CFG.EVAL.USE_GPU = True # Whether to use GPU for evaluation. Default: True
