import argparse
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
import seaborn as sns
from tensorflow import keras
import os
#os.environ["CUDA_VISIBLE_DEVICES"] = "-1" # disables GPU
#from tqdm import tqdm

from ast import literal_eval
from tensorflow.keras import layers
import logomaker as lm
from sklearn.model_selection import StratifiedKFold
import sklearn
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from sklearn.model_selection import train_test_split

import util

parser = argparse.ArgumentParser(description='Train and evaluate one model on all kinds of post-translational modifications')

parser.add_argument('-l', '--logdir', type=str, default="./tensorboard_logs/", help='Logging directory for tensorboard.')
parser.add_argument('-o', '--output', type=str, default="./output/", help='Output directory for plots/metrics')
parser.add_argument('-p', '--ptm', nargs='+', default=[], help='List of ptms used for multi model')
parser.add_argument('-n', '--name', type=str, default='multi_class_model', help='Name tag for a particular run')
args = parser.parse_args()


SEED = 42
WINDOW_SIZE= 8
LOGDIR = args.logdir
OUTPUT_DIR = args.output
PTM_LIST = args.ptm
NAME = args.name

print('########################### DATA SETUP ##########################')
df = pd.read_csv("./data/ptm_data.csv", index_col=[0])

# splitting Methylation in Arg/Lys Methylation since they are quite different
# and splitting Proline/Lysine Methylation
ptm_mod_list = []
for index in df.index:
    ptm = df.loc[index].ptm
    mod_aa = df.loc[index].mod_aa
    if ptm == 'Methylation':
        if mod_aa == 'K':
            ptm_mod_list.append('Lys_Methylation')
        elif mod_aa == 'R':
            ptm_mod_list.append('Arg_Methylation')
        else:
            RaiseAttributeError('Mod_aa needs to be K or R')
    elif ptm == 'Hydroxylation':
        if mod_aa == 'K':
            ptm_mod_list.append('Lys_Hydroxylation')
        if mod_aa == 'P':
            ptm_mod_list.append('Pro_Hydroxylation')
    else:
        ptm_mod_list.append(ptm)
df['ptm'] = ptm_mod_list

# filtering using the PTM_LISt arg
df = df[df.ptm.isin(PTM_LIST)].reset_index()

# Creating labels for later, one time pos/neg examples for each modification
# and one time we pool negative examples by amino acid type for oversampling later
ptm_label_list = []
ptm_label_negPool_list = []
for index in df.index:
    ptm = df.loc[index].ptm
    label = df.loc[index].label
    mod_aa = df.loc[index].mod_aa
    if label == 0:
        ptm_label_negPool = 'unmodified'
        ptm_label = ptm + '_unmodified_' + mod_aa
    else:
        ptm_label_negPool = ptm
        ptm_label = ptm + '_' + mod_aa

    ptm_label_list.append(ptm_label)
    ptm_label_negPool_list.append(ptm_label_negPool)

label_dict = {}
for i, label in enumerate(np.unique(ptm_label_negPool_list)):
    label_dict[label] = i

ptm_label_int_list = []
for label in ptm_label_negPool_list:
    ptm_label_int_list.append(label_dict[label])


df['ptm_label'] = ptm_label_list
df['ptm_label_negPool'] = ptm_label_int_list

train_val, test = train_test_split(df, test_size=0.2, random_state=SEED, stratify=df.ptm_label.values)

shape_len = train_val.shape[0] #.reshape(shape_len,1)
X_train_val = util.return_struct_feat(train_val, WINDOW_SIZE, shape_len)
Z_train_val = train_val.ptm_label.values
Y_train_val = train_val.ptm_label_negPool.values
X_test = util.return_struct_feat(test, WINDOW_SIZE, shape_len=test.shape[0])
Y_test = test.ptm_label_negPool.values

log_config = NAME

print("########################### TRAINING #############################")
model_list, history_list = util.train_cross_validate_multi_class(
    X_train_val,
    Y_train_val,
    Z_train_val,
    util.create_multi_model,
    LOGDIR+log_config,
    sampling='over',
    warmup=True
)

print('############################# MODEL SAVING #########################')
for i, model in enumerate(model_list):
    model.save('./models/' + log_config + '_' + str(i))

print('########################### EVALUATION ###########################')
train_pred_list, val_pred_list, Y_train_list, Y_val_list = util.test_models_multi(
    X_train_val,
    Y_train_val,
    Z_train_val,
    model_list,
    OUTPUT_DIR+log_config
)

util.final_test_models_multi(X_test, Y_test, model_list, OUTPUT_DIR+log_config)

util.final_test_models_multi_per_ptm(test, model_list, OUTPUT_DIR+log_config)
