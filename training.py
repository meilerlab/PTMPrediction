import argparse
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
import seaborn as sns
from tensorflow import keras
import os

from ast import literal_eval
from tensorflow.keras import layers
import logomaker as lm
from sklearn.model_selection import StratifiedKFold
import sklearn
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from sklearn.model_selection import train_test_split
from imblearn.over_sampling import RandomOverSampler
from imblearn.under_sampling import RandomUnderSampler

import util

parser = argparse.ArgumentParser(description='Train and evaluate one model on all kinds of post-translational modifications')

parser.add_argument('-p', '--ptm', nargs='+', default=[])
parser.add_argument('-l', '--logdir', type=str, default="./tensorboard_logs/", help='Logging directory for tensorboard.')
parser.add_argument('-o', '--output', type=str, default="./output/", help='Output directory for plots/metrics')
parser.add_argument('-g', '--gpu', type=bool, default=False, help='Set flag to enable GPU')
parser.add_argument('-c', '--class_weight', type=bool, default=False, help='Use class_weights instead of over/undersamping')

args = parser.parse_args()

if not args.gpu:
    print('Disable GPU')
    os.environ['CUDA_VISIBLE_DEVICES'] = "-1"

SEED = 42
WINDOW_SIZE=8
LOGDIR = args.logdir
OUTPUT_DIR = args.output
PTM_LIST = args.ptm

print('########################### DATA SETUP ##########################')
df = pd.read_csv("./data/ptm_data.csv", index_col=[0])

# splitting Methylation in Arg/Lys Methylation since they are quite different
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
    else:
        ptm_mod_list.append(ptm)
df['ptm'] = ptm_mod_list

# preparing for train_test split and cross validation
ptm_label_list = []
for index in df.index:
    ptm = df.loc[index].ptm
    label = df.loc[index].label
    ptm_label_list.append(ptm+'_'+str(label))

df['ptm_label'] = ptm_label_list

train_val, test = train_test_split(df, test_size=0.2, random_state=SEED, stratify=df.ptm_label.values)

shape_len = train_val.shape[0]
X_train_val = util.return_struct_feat(train_val, WINDOW_SIZE, shape_len)
Y_train_val = train_val.label.values

# We take the ptm+label list for splitting our data, so that we later can fine-tune/train on
# single PTMs without having data leakage or no fair comparison
Z_train_val = train_val.ptm_label.values

# we only go for undersampling if the negative class has ~100x more samples
# and we have a reasonable amount of positive examples
# listing all here for clarity
sampling_type_dict = {
    'N-linkedGlycosylation':'over',
    'phosphorylation':'under',
    'Arg_Methylation':'over',
    'Lys_Methylation':'over',
    'Malonylation':'over',
    'Acetylation':'over',
    'O-linkedGlycosylation':'under',
    'Ubiquitination':'over',
    'Sumoylation':'over',
    'Glutathionylation':'over',
    'Succinylation':'over',
    'S-nitrosylation':'over',
    'Glutarylation':'over',
    'Hydroxylation':'over',
    'Formylation':'over',
    'Citrullination':'over',
    'Nitration':'over',
    'Gamma-carboxyglutamic-acid':'over',
    'Crotonylation':'over'
}

for filter_type in PTM_LIST:

    # Create seq logo
    util.create_seqLogo(df[df.ptm == filter_type])
    plt.savefig(OUTPUT_DIR+'sequence_logo_' + filter_type + '.png')

    if args.class_weight:
        print('Using class weights')
        sampling_type = 'class_weights'
    else:
        sampling_type = sampling_type_dict[filter_type]
    log_config = filter_type + '_LR_' + '1e-4_' + 'Simple-Model' + sampling_type

    print("########################### TRAINING:" + filter_type + " #############################")
    model_list, history_list = util.train_cross_validate(X_train_val, Y_train_val, Z_train_val, util.create_simple_model, LOGDIR+log_config, sampling=sampling_type, warmup=False, filter_type=filter_type)

    print('############################# MODEL SAVING:' + filter_type + '  #########################')
    for i, model in enumerate(model_list):
        model.save('./models/' + log_config + '_' + str(i))

    print('########################### EVALUATION:' + filter_type + '  ###########################')
    train_pred_list, val_pred_list, Y_train_list, Y_val_list = util.test_models(X_train_val, Y_train_val, Z_train_val, model_list, OUTPUT_DIR+log_config, filter_type=filter_type)

    # filter test set also by PTM type
    test = test[test.ptm == filter_type]
    X_test = util.return_struct_feat(test, WINDOW_SIZE, shape_len=test.shape[0])
    Y_test = test.label.values
    util.final_test_models(X_test, Y_test, model_list, OUTPUT_DIR+log_config, filter_type=filter_type)
