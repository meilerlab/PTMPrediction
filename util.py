import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
#import seaborn as sns
from tensorflow import keras
import math
from ast import literal_eval
from tensorflow.keras import layers
import logomaker as lm
from sklearn.model_selection import StratifiedKFold
import sklearn
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
#from sklearn.model_selection import train_test_split
from imblearn.over_sampling import RandomOverSampler
from imblearn.under_sampling import RandomUnderSampler
from sklearn.metrics import classification_report

SEED = 42
WINDOW_SIZE = 8
SEQ_FEAT_SHAPE = 9
STRUCT_FEAT_SHAPE = 10

plt.rcParams['figure.figsize'] = (20, 20)

early_stopping = tf.keras.callbacks.EarlyStopping(
    monitor='val_loss',
    verbose=2,
    patience=20,
    mode='min',
    restore_best_weights=True)

lr = 0.0001
opt = keras.optimizers.Adam(learning_rate=lr)


def genWord2Vec(alphabet):
    word2vec = dict(zip(alphabet, range(len(alphabet)+1)))
    return word2vec
WORD_DICT = genWord2Vec("ACDEFGHIKLMNPQRSTVWY")

def return_vector(df, window_size=20):
    sequence_list = []
    half = int(window_size/2)
    for seq in df.trunc_seq.values:
        sub_list = []
        for i, char in enumerate(seq[10-half:11+half]):
            sub_list.append(WORD_DICT[char])
        sequence_list.append(np.array(sub_list))
    return np.array(sequence_list)

def return_angles(df, name_angle):
    angle_list = []
    if name_angle == 'phi':
        for vals in df.phi.values:
            angle_list.append(literal_eval(vals)[5:8])#[1:11])
    elif name_angle == 'psi':
        for vals in df.psi.values:
            angle_list.append(literal_eval(vals)[5:8])#[1:11])
    else:
        print('Not a valid angle')
    return np.array(angle_list)

def return_struct_feat(train_val, window_size, shape_len):
    return np.concatenate((return_vector(train_val, window_size=window_size), return_angles(train_val, name_angle='phi'), return_angles(train_val, name_angle='psi'), train_val.sasa.values.reshape(shape_len,1), train_val.E.values.reshape(shape_len,1), train_val.H.values.reshape(shape_len,1), train_val.L.values.reshape(shape_len,1)), axis=-1)

def create_seqLogo(df):
    counts = lm.alignment_to_matrix(df[df.label == 0].trunc_seq.values)
    counts2 = lm.alignment_to_matrix(df[df.label == 1].trunc_seq.values)
    normalized_counts = lm.transform_matrix(counts2, normalize_values=True) - lm.transform_matrix(counts, normalize_values=True)
    lm.Logo(normalized_counts.fillna(0), color_scheme='dmslogo_funcgroup', figsize=(15, 5))
    plt.xlabel('Position')
    plt.ylabel('Probability')

def modify_labels(Y_labels):
    '''Pools all negative classes into one'''
    Y_labels_mod = []
    for label in Y_labels:
        if label%2 == 0:
            Y_labels_mod.append(0)
        else:
            Y_labels_mod.append(math.ceil(label/2))

    return np.array(Y_labels_mod)


def create_multi_model():
    inputs = layers.Input(shape=(SEQ_FEAT_SHAPE,))
    inputs_structure = layers.Input(shape=(STRUCT_FEAT_SHAPE))

    # Sequence block
    embedding_layer = TokenAndPositionEmbedding(SEQ_FEAT_SHAPE, 20, 8)
    x = embedding_layer(inputs)
    transformer_block = TransformerBlock(8, 3, 16) # embedding, heads, feed forward size
    x = transformer_block(x)
    x = tf.keras.layers.GlobalAveragePooling1D()(x)
    x = tf.keras.layers.Dense(32, activation='relu')(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.Dropout(0.2)(x)

    #y = normalizer(inputs_structure)
    y = layers.Dense(32, activation='relu')(inputs_structure)
    y = tf.keras.layers.BatchNormalization()(y)
    y = layers.Dropout(0.2)(y)

    y = layers.Dense(32, activation='relu')(y)
    y = tf.keras.layers.BatchNormalization()(y)
    y = layers.Dropout(0.2)(y)

    y = layers.Dense(32, activation='relu')(y)
    y = tf.keras.layers.BatchNormalization()(y)
    y = layers.Dropout(0.2)(y)

    # Connect the two tracks
    x = layers.Concatenate(axis=1)([x, y])

    outputs = layers.Dense(8, activation="softmax")(x)
    model = keras.Model(inputs=[inputs, inputs_structure], outputs=outputs)

    opt = keras.optimizers.Adam(learning_rate=1e-4)
    model.compile(optimizer=opt, loss="sparse_categorical_crossentropy") # need to exclude any metrics to be able to load it later


    return model

def train_cross_validate(X_train_val, Y_train_val, Z_train_val, create_model, log_config, sampling='over', seed=42, warmup=False, filter_type=''):

    sKfold = StratifiedKFold(n_splits=10, shuffle=True, random_state=seed)
    split = sKfold.split(X_train_val, Z_train_val)
    history_list = []
    model_list = []
    i = 0

    tensorboard_callback = keras.callbacks.TensorBoard(log_dir=log_config)

    for train_index, val_index in split:
        i += 1
        print('############# Split ' + str(i) + ' #####################################')

        X_train, X_val = X_train_val[train_index], X_train_val[val_index]
        Y_train, Y_val = Y_train_val[train_index], Y_train_val[val_index]
        Z_train, Z_val = Z_train_val[train_index], Z_train_val[val_index]

        if filter_type: # check that string is not empty
            mask_train = (Z_train == filter_type + '_0') | (Z_train == filter_type + '_1')
            mask_val = (Z_val == filter_type + '_0') | (Z_val == filter_type + '_1')

            X_train, X_val = X_train[mask_train], X_val[mask_val]
            Y_train, Y_val = Y_train[mask_train], Y_val[mask_val]

        if sampling == 'over':
            X_train, Y_train = RandomOverSampler(random_state=seed).fit_resample(X_train, Y_train) # oversampling
        elif sampling == 'under':
            X_train, Y_train = RandomUnderSampler(random_state=seed).fit_resample(X_train, Y_train) # undersampling
        elif sampling == 'class_weights':
            total, pos = len(Y_train), sum(Y_train)
            neg = total-pos

            weight_0 = (1/neg) * (total/2.0)
            weight_1 = (1/pos) * (total/2.0)

            class_weight = {0: weight_0 , 1: weight_1}
        else:
            raise ValueError('Sampling option not recognized')

        if i == 1: # log to tensorboard
            callbacks = [early_stopping, tensorboard_callback]
        else:
            callbacks = [early_stopping]

        if warmup:
            steps = int(len(X_train)/32) # 32 is batch size
            #print(steps)
            total_steps = steps*15 # where 50 is N epochs
            warmup_steps = int(0.10*total_steps)
            warmupLR = WarmupCosineDecay(total_steps=total_steps,
                                          warmup_steps=warmup_steps,
                                          hold=int(warmup_steps/2),
                                         start_lr=0.0,
                                         target_lr=1e-4)
            callbacks = callbacks+[warmupLR]

        model = create_model()

        if sampling == 'class_weights':
            history = model.fit(
                            x=[X_train[:, :WINDOW_SIZE+1], X_train[:, WINDOW_SIZE+1:]], y=Y_train,
                            validation_data=([X_val[:, :WINDOW_SIZE+1], X_val[:, WINDOW_SIZE+1:]], Y_val),
                            callbacks=callbacks,
                            epochs=200,
                            verbose=2,
                            class_weight=class_weight
            )

        else:

            history = model.fit(
                            x=[X_train[:, :WINDOW_SIZE+1], X_train[:, WINDOW_SIZE+1:]], y=Y_train,
                            validation_data=([X_val[:, :WINDOW_SIZE+1], X_val[:, WINDOW_SIZE+1:]], Y_val),
                            callbacks=callbacks,
                            epochs=200,
                            verbose=2
            )

        history_list.append(history)
        model_list.append(model)
    return model_list, history_list

def train_cross_validate_multi_class(X_train_val, Y_train_val, Z_train_val, create_model, log_config, sampling='over', seed=42, warmup=False):

    sKfold = StratifiedKFold(n_splits=10, shuffle=True, random_state=seed)
    split = sKfold.split(X_train_val, Z_train_val)
    history_list = []
    model_list = []
    i = 0
    batch_size = 32
    tensorboard_callback = keras.callbacks.TensorBoard(log_dir=log_config)

    for train_index, val_index in split:
        i += 1
        print('############# Split ' + str(i) + ' ###########################')

        X_train, X_val = X_train_val[train_index], X_train_val[val_index]
        Y_train, Y_val = Y_train_val[train_index], Y_train_val[val_index]
        Z_train = Z_train_val[train_index]

        X_train_Y_train = np.hstack((X_train,
                                     Y_train.reshape(X_train.shape[0], 1)))

        # print('Before', X_train.shape, Y_train.shape, sum(Y_train))
        if sampling == 'over':
            X_train_Y_train, Z_train = RandomOverSampler(random_state=seed).fit_resample(X_train_Y_train, Z_train) # oversampling
        elif sampling == 'under':
            X_train_Y_train, Z_train = RandomUnderSampler(random_state=seed).fit_resample(X_train_Y_train, Z_train) # undersampling
        else:
            raise ValueError('Sampling option not recognized')
        X_train = X_train_Y_train[:, :-1]
        Y_train = X_train_Y_train[:, -1]
        if i == 1:  # log to tensorboard
            callbacks = [early_stopping, tensorboard_callback]
        else:
            callbacks = [early_stopping]

        if warmup:
            steps = int(len(X_train)/batch_size) # 32 is batch size
            total_steps = steps*20 # where 25 is N epochs
            warmup_steps = int(0.10*total_steps)
            warmupLR = WarmupCosineDecay(total_steps=total_steps,
                                          warmup_steps=warmup_steps,
                                          hold=int(warmup_steps/2),
                                         start_lr=0.0,
                                         target_lr=1e-4)
            callbacks = callbacks+[warmupLR]

        model = create_model()

        history = model.fit(
                        x=[X_train[:, :WINDOW_SIZE+1], X_train[:, WINDOW_SIZE+1:]],
                        y=Y_train,
                        validation_data=([X_val[:, :WINDOW_SIZE+1], X_val[:, WINDOW_SIZE+1:]], Y_val),
                        callbacks=callbacks,
                        epochs=200,
                        verbose=2,
                        batch_size=batch_size
        )

        history_list.append(history)
        model_list.append(model)
    return model_list, history_list


def create_simple_model():
    inputs = layers.Input(shape=(SEQ_FEAT_SHAPE))
    inputs_structure = layers.Input(shape=(STRUCT_FEAT_SHAPE))

    # Sequence block
    x = tf.keras.layers.Embedding(20, 4)(inputs)
    x = tf.keras.layers.BatchNormalization()(x)
    x = layers.Dropout(0.1)(x)

    x = tf.keras.layers.GlobalAveragePooling1D()(x)
    x = tf.keras.layers.Dense(8, activation='relu')(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.Dropout(0.1)(x)

    # Structure block
    y = layers.Dense(8, activation='relu')(inputs_structure)
    y = tf.keras.layers.BatchNormalization()(y)
    y = layers.Dropout(0.1)(y)

    y = layers.Dense(8, activation='relu')(y)
    y = tf.keras.layers.BatchNormalization()(y)
    y = layers.Dropout(0.1)(y)

    # Connect the two tracks
    x = layers.Concatenate(axis=1)([x, y])

    outputs = layers.Dense(1, activation="sigmoid")(x)
    model = keras.Model(inputs=[inputs, inputs_structure], outputs=outputs)

    model.compile(optimizer=opt, loss="binary_crossentropy")

    return model

def train_cross_validate_finetune(X_train_val, Y_train_val, old_model_list, log_config, sampling='over'):

    sKfold = StratifiedKFold(n_splits=10, shuffle=True, random_state=SEED)
    split = sKfold.split(X_train_val, Y_train_val)
    history_list = []
    model_list = []
    i = 0

    tensorboard_callback = keras.callbacks.TensorBoard(log_dir=LOGDIR+log_config)

    for train_index, val_index in split:
        i += 1
        print('############################ Split ' + str(i) + ' ############################')

        X_train, X_val = X_train_val[train_index], X_train_val[val_index]
        Y_train, Y_val = Y_train_val[train_index], Y_train_val[val_index]

        if sampling == 'over':
            X_train, Y_train = RandomOverSampler(random_state=SEED).fit_resample(X_train, Y_train) # oversampling
        elif sampling == 'under':
            X_train, Y_train = RandomUnderSampler(random_state=SEED).fit_resample(X_train, Y_train) # undersampling
        else:
            raise ValueError('Sampling option not recognized')

        if i == 1: # log to tensorboard
            callbacks = [early_stopping, tensorboard_callback]
        else:
            callbacks = [early_stopping]

        model = old_model_list[i-1]
        model.trainable = True
        model.compile(optimizer=keras.optimizers.Adam(1e-5), loss="binary_crossentropy")#, metrics=HP_METRICS)

        history = model.fit(
                        x=[X_train[:, :WINDOW_SIZE+1], X_train[:, WINDOW_SIZE+1:]], y=Y_train,
                        validation_data=([X_val[:, :WINDOW_SIZE+1], X_val[:, WINDOW_SIZE+1:]], Y_val),
                        callbacks=callbacks,
                        epochs=200,
                        verbose=0
        )
        history_list.append(history)
        model_list.append(model)
    return model_list, history_list


def test_models(X_train_val, Y_train_val, Z_train_val, model_list, log_config, seed=42, filter_type=''):
    acc_train = []
    ROC_train = []
    PRC_train = []
    MCC_train = []
    precision_train = []
    recall_train = []

    acc_eval = []
    ROC_eval = []
    PRC_eval = []
    MCC_eval = []
    precision_eval = []
    recall_eval = []

    fp_rate_train = []
    fn_rate_train = []
    fp_rate_eval = []
    fn_rate_eval = []

    best_thresh = 0.5
    sKfold = StratifiedKFold(n_splits=10, shuffle=True, random_state=seed)
    split = sKfold.split(X_train_val, Z_train_val)
    index_list = list(split)

    train_pred_list = []
    val_pred_list = []
    Y_val_list = []
    Y_train_list = []
    i = 0
    for indices, model in zip(index_list, model_list):
        train_index = indices[0]
        val_index = indices[1]
        X_train, X_val = X_train_val[train_index], X_train_val[val_index]
        Y_train, Y_val = Y_train_val[train_index], Y_train_val[val_index]
        Z_train, Z_val = Z_train_val[train_index], Z_train_val[val_index]

        if filter_type: # check that string is not empty
            mask_train = (Z_train == filter_type + '_0') | (Z_train == filter_type + '_1')
            mask_val = (Z_val == filter_type + '_0') | (Z_val == filter_type + '_1')

            X_train, X_val = X_train[mask_train], X_val[mask_val]
            Y_train, Y_val = Y_train[mask_train], Y_val[mask_val]


        train_pred = model.predict([X_train[:, :WINDOW_SIZE+1], X_train[:, WINDOW_SIZE+1:]])
        val_pred = model.predict([X_val[:, :WINDOW_SIZE+1], X_val[:, WINDOW_SIZE+1:]])

        train_pred_list.append(train_pred)
        val_pred_list.append(val_pred)
        Y_val_list.append(Y_val)
        Y_train_list.append(Y_train)

        acc_train.append(sklearn.metrics.accuracy_score(Y_train, train_pred>best_thresh))
        acc_eval.append(sklearn.metrics.accuracy_score(Y_val, val_pred>best_thresh))
        ROC_train.append(sklearn.metrics.roc_auc_score(Y_train, train_pred))
        ROC_eval.append(sklearn.metrics.roc_auc_score(Y_val, val_pred))
        PRC_train.append(sklearn.metrics.average_precision_score(Y_train, train_pred))
        PRC_eval.append(sklearn.metrics.average_precision_score(Y_val, val_pred))
        MCC_train.append(sklearn.metrics.matthews_corrcoef(Y_train, train_pred >= best_thresh))
        MCC_eval.append(sklearn.metrics.matthews_corrcoef(Y_val, val_pred >= best_thresh))

        precision_eval.append(sklearn.metrics.precision_score(Y_val, val_pred >= best_thresh))
        recall_eval.append(sklearn.metrics.recall_score(Y_val, val_pred >= best_thresh))

        precision_train.append(sklearn.metrics.precision_score(Y_train, train_pred>best_thresh))
        recall_train.append(sklearn.metrics.recall_score(Y_train, train_pred>best_thresh))

        cm = confusion_matrix(Y_train, train_pred >= 0.5)
        fp_rate = cm[0][1]/(cm[0][0]+cm[0][1])
        fn_rate = cm[1][0]/(cm[1][0]+cm[1][1])
        fp_rate_train.append(fp_rate)
        fn_rate_train.append(fn_rate)

        disp = ConfusionMatrixDisplay(confusion_matrix=cm)
        disp.plot()
        plt.savefig(log_config+'_cm_train_'+str(i)+'.png')
        plt.close()

        cm = confusion_matrix(Y_val, val_pred >= 0.5)
        fp_rate = cm[0][1]/(cm[0][0]+cm[0][1])
        fn_rate = cm[1][0]/(cm[1][0]+cm[1][1])
        fp_rate_eval.append(fp_rate)
        fn_rate_eval.append(fn_rate)

        disp = ConfusionMatrixDisplay(confusion_matrix=cm)
        disp.plot()
        plt.savefig(log_config+'_cm_eval_'+str(i)+'.png')
        plt.close()

        i += 1

    print('')
    print('Avg Accuracy TRAIN: ', np.mean(acc_train).round(2), '+-', np.std(acc_train).round(2))
    print('Avg Accuracy EVAL: ', np.mean(acc_eval).round(2), '+- ', np.std(acc_eval).round(2))
    print('')
    print('ROC AUC TRAIN: ', np.mean(ROC_train).round(2), '+- ', np.std(ROC_train).round(2))
    print('ROC AUC EVAL: ', np.mean(ROC_eval).round(2), '+- ', np.std(ROC_eval).round(2))
    print('')
    print('Avg TRAIN PRC: ', np.mean(PRC_train).round(2), '+- ', np.std(PRC_train).round(2))
    print('Avg EVAL PRC: ', np.mean(PRC_eval).round(2), '+- ', np.std(PRC_eval).round(2))
    print('')
    print('MCC Train', np.mean(MCC_train).round(2), '+- ', np.std(MCC_train).round(2))
    print('MCC Val', np.mean(MCC_eval).round(2), '+- ', np.std(MCC_eval).round(2))
    print('')
    print('Precision EVAL', np.mean(precision_eval).round(2), '+- ', np.std(precision_eval).round(2))
    print('Recall EVAL', np.mean(recall_eval).round(2), '+- ', np.std(recall_eval).round(2))
    with open(log_config+'_metrics.txt', 'w') as f:
        f.write('PTM, Accuracy, ROC_AUC, PRC, MCC, precision, recall, fp_rate, fn_rate\n')
        f.write(filter_type+'_TRAIN'+','+str(np.mean(acc_train).round(2))+'±'+str(np.std(acc_train).round(2))
                +','+str(np.mean(ROC_train).round(2))+'±'+str(np.std(ROC_train).round(2))+','+str(np.mean(PRC_train).round(2))+'±'+
                str(np.std(PRC_train).round(2))+','+str(np.mean(MCC_train).round(2))+'±'+str(np.std(MCC_train).round(2))+','+str(np.mean(precision_train).round(2))+'±'+
                str(np.std(precision_train).round(2))+','+str(np.mean(recall_train).round(2))+'±'
                +str(np.std(recall_train).round(2))+','+str(np.mean(fp_rate_train).round(2))+
                '±'+str(np.std(fp_rate_train).round(2))+','+str(np.mean(fn_rate_train).round(2))+'±'+
                str(np.std(fn_rate_train).round(2))+'\n')

        f.write(filter_type+'_EVAL'+','+str(np.mean(acc_eval).round(2))+'±'+str(np.std(acc_eval).round(2))
                +','+str(np.mean(ROC_eval).round(2))+'±'+str(np.std(ROC_eval).round(2))+','+str(np.mean(PRC_eval).round(2))+'±'+
                str(np.std(PRC_eval).round(2))+','+str(np.mean(MCC_eval).round(2))+'±'+str(np.std(MCC_eval).round(2))+','+str(np.mean(precision_eval).round(2))+'±'+
                str(np.std(precision_eval).round(2))+','+str(np.mean(recall_eval).round(2))+'±'
                +str(np.std(recall_eval).round(2))+','+str(np.mean(fp_rate_eval).round(2))+
                '±'+str(np.std(fp_rate_eval).round(2))+','+str(np.mean(fn_rate_eval).round(2))+'±'+
                str(np.std(fn_rate_eval).round(2))+'\n')

    return train_pred_list, val_pred_list, Y_train_list, Y_val_list

def final_test_models(X_test, Y_test, model_list, log_config, filter_type=''):
    acc_test = []
    ROC_test = []
    PRC_test = []
    MCC_test = []
    precision_test = []
    recall_test = []
    fp_rate_test = []
    fn_rate_test = []

    best_thresh = 0.5

    for i, model in enumerate(model_list):
        test_pred = model.predict([X_test[:, :WINDOW_SIZE+1], X_test[:, WINDOW_SIZE+1:]])

        acc_test.append(sklearn.metrics.accuracy_score(Y_test, test_pred>best_thresh))
        ROC_test.append(sklearn.metrics.roc_auc_score(Y_test, test_pred))
        PRC_test.append(sklearn.metrics.average_precision_score(Y_test, test_pred))
        MCC_test.append(sklearn.metrics.matthews_corrcoef(Y_test, test_pred >= best_thresh))
        precision_test.append(sklearn.metrics.precision_score(Y_test, test_pred >= best_thresh))
        recall_test.append(sklearn.metrics.recall_score(Y_test, test_pred >= best_thresh))

        cm = confusion_matrix(Y_test, test_pred >= 0.5)
        fp_rate = cm[0][1]/(cm[0][0]+cm[0][1])
        fn_rate = cm[1][0]/(cm[1][0]+cm[1][1])
        fp_rate_test.append(fp_rate)
        fn_rate_test.append(fn_rate)
        disp = ConfusionMatrixDisplay(confusion_matrix=cm)
        disp.plot()
        plt.savefig(log_config+'_cm_test_'+str(i)+'.png')
        plt.close()

    print('Accuracy TEST: ', np.mean(acc_test).round(2), '+-', np.std(acc_test).round(2), '\n')
    print('ROC AUC TEST: ', np.mean(ROC_test).round(2), '+- ', np.std(ROC_test).round(2), '\n')
    print('PRC TEST: ', np.mean(PRC_test).round(2), '+- ', np.std(PRC_test).round(2), '\n')
    print('MCC TEST', np.mean(MCC_test).round(2), '+- ', np.std(MCC_test).round(2), '\n')
    print('Precision TEST', np.mean(precision_test).round(2), '+- ', np.std(precision_test).round(2), '\n')
    print('Recall TEST', np.mean(recall_test).round(2), '+- ', np.std(recall_test).round(2), '\n')
    print('FP RATE TEST', np.mean(fp_rate_test).round(2), '+- ', np.std(fp_rate_test).round(2), '\n')
    print('FN RATE TEST', np.mean(fn_rate_test).round(2), '+- ', np.std(fn_rate_test).round(2), '\n')

    print("Best model by MCC:", np.argmax(MCC_test))

    with open(log_config+'_metrics_test_set.txt', 'w') as f:
        f.write('PTM, Accuracy, ROC_AUC, PRC, MCC, precision, recall, fp_rate, fn_rate\n')
        f.write(filter_type+'_TEST'+','+str(np.mean(acc_test).round(2))+
                +','+str(np.mean(ROC_test).round(2))+','+str(np.mean(PRC_test).round(2))+
                ','+str(np.mean(MCC_test).round(2))+','+str(np.mean(precision_test).round(2))+
                ','+str(np.mean(recall_test).round(2))+
                ','+str(np.mean(fp_rate_test).round(2))+
                ','+str(np.mean(fn_rate_test).round(2))+
                str(np.std(fn_rate_test).round(2))+'\n')

class TransformerBlock(layers.Layer):
    def __init__(self, embed_dim, num_heads, ff_dim, rate=0.1, **kwargs):
        super(TransformerBlock, self).__init__(**kwargs)
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.ff_dim = ff_dim
        self.rate = rate

        self.att = layers.MultiHeadAttention(num_heads=self.num_heads, key_dim=self.embed_dim, name='attention')
        self.ffn = keras.Sequential(
            [layers.Dense(self.ff_dim, activation="relu"), layers.Dense(self.embed_dim),]
        )
        self.layernorm1 = layers.LayerNormalization(epsilon=1e-6)
        self.layernorm2 = layers.LayerNormalization(epsilon=1e-6)
        self.dropout1 = layers.Dropout(self.rate)
        self.dropout2 = layers.Dropout(self.rate)
    def get_config(self):
        config = super(TransformerBlock, self).get_config()
        config.update({
            "embed_dim": self.embed_dim,
            "num_heads": self.num_heads,
            "ff_dim": self.ff_dim,
            "rate": self.rate
        })
        return config

    def call(self, inputs, training):
        attn_output = self.att(inputs, inputs)
        attn_output = self.dropout1(attn_output, training=training)
        out1 = self.layernorm1(inputs + attn_output)
        ffn_output = self.ffn(out1)
        ffn_output = self.dropout2(ffn_output, training=training)
        return self.layernorm2(out1 + ffn_output)



def test_models_multi(X_train_val, Y_train_val, Z_train_val, model_list, log_config, seed=42):

    acc_train = []
    ROC_train = []
    MCC_train = []
    precision_train = []
    recall_train = []

    acc_eval = []
    ROC_eval = []
    MCC_eval = []
    precision_eval = []
    recall_eval = []

    df_class_rep_train = pd.DataFrame()
    df_class_rep_val = pd.DataFrame()

    sKfold = StratifiedKFold(n_splits=10, shuffle=True, random_state=seed)
    split = sKfold.split(X_train_val, Z_train_val)
    index_list = list(split)

    train_pred_list = []
    val_pred_list = []
    Y_val_list = []
    Y_train_list = []
    i = 0
    for indices, model in zip(index_list, model_list):
        train_index = indices[0]
        val_index = indices[1]
        X_train, X_val = X_train_val[train_index], X_train_val[val_index]
        Y_train, Y_val = Y_train_val[train_index], Y_train_val[val_index]

        train_pred = model.predict([X_train[:, :WINDOW_SIZE+1], X_train[:, WINDOW_SIZE+1:]])
        val_pred = model.predict([X_val[:, :WINDOW_SIZE+1], X_val[:, WINDOW_SIZE+1:]])

        train_pred_list.append(train_pred)
        val_pred_list.append(val_pred)
        Y_val_list.append(Y_val)
        Y_train_list.append(Y_train)

        train_pred_arg = np.argmax(train_pred, axis=-1)
        val_pred_arg = np.argmax(val_pred, axis=-1)

        acc_train.append(sklearn.metrics.accuracy_score(Y_train, train_pred_arg))
        acc_eval.append(sklearn.metrics.accuracy_score(Y_val, val_pred_arg))
        ROC_train.append(sklearn.metrics.roc_auc_score(Y_train, train_pred, multi_class='ovr'))
        ROC_eval.append(sklearn.metrics.roc_auc_score(Y_val, val_pred, multi_class='ovr'))
        MCC_train.append(sklearn.metrics.matthews_corrcoef(Y_train, train_pred_arg))
        MCC_eval.append(sklearn.metrics.matthews_corrcoef(Y_val, val_pred_arg))

        precision_eval.append(sklearn.metrics.precision_score(Y_val, val_pred_arg, average='micro'))
        recall_eval.append(sklearn.metrics.recall_score(Y_val, val_pred_arg, average='micro'))

        precision_train.append(sklearn.metrics.precision_score(Y_train, train_pred_arg, average='micro'))
        recall_train.append(sklearn.metrics.recall_score(Y_train, train_pred_arg, average='micro'))

        cm = confusion_matrix(Y_train, train_pred_arg)

        disp = ConfusionMatrixDisplay(confusion_matrix=cm)
        disp.plot()
        plt.savefig(log_config+'_cm_train_'+str(i)+'.png')
        plt.close()

        cm = confusion_matrix(Y_val, val_pred_arg)
        disp = ConfusionMatrixDisplay(confusion_matrix=cm)
        disp.plot()
        plt.savefig(log_config+'_cm_eval_'+str(i)+'.png')
        plt.close()

        i += 1

        class_rep_train = classification_report(Y_train, train_pred_arg, output_dict=True)
        df_class_rep_train = pd.concat((df_class_rep_train, pd.DataFrame(data=class_rep_train).T))

        class_rep_val = classification_report(Y_val, val_pred_arg, output_dict=True)
        df_class_rep_val = pd.concat((df_class_rep_val, pd.DataFrame(data=class_rep_val).T))

    df_group_train = df_class_rep_train.groupby(df_class_rep_train.index)
    df_group_train.mean().to_csv(log_config+'_class_report_mean_train_set.csv')
    df_group_train.std().to_csv(log_config+'_class_report_std_train_set.csv')

    df_group_val = df_class_rep_val.groupby(df_class_rep_val.index)
    df_group_val.mean().to_csv(log_config+'_class_report_mean_val_set.csv')
    df_group_val.std().to_csv(log_config+'_class_report_std_val_set.csv')

    print('')
    print('Avg Accuracy TRAIN: ', np.mean(acc_train).round(2), '+-', np.std(acc_train).round(2))
    print('Avg Accuracy EVAL: ', np.mean(acc_eval).round(2), '+- ', np.std(acc_eval).round(2))
    print('')
    print('ROC AUC TRAIN: ', np.mean(ROC_train).round(2), '+- ', np.std(ROC_train).round(2))
    print('ROC AUC EVAL: ', np.mean(ROC_eval).round(2), '+- ', np.std(ROC_eval).round(2))
    print('')
    print('MCC Train', np.mean(MCC_train).round(2), '+- ', np.std(MCC_train).round(2))
    print('MCC Val', np.mean(MCC_eval).round(2), '+- ', np.std(MCC_eval).round(2))
    print('')
    print('Precision EVAL', np.mean(precision_eval).round(2), '+- ', np.std(precision_eval).round(2))
    print('Recall EVAL', np.mean(recall_eval).round(2), '+- ', np.std(recall_eval).round(2))
    with open(log_config+'_metrics.txt', 'w') as f:
        f.write('PTM, Accuracy, ROC_AUC, MCC, precision, recall, fp_rate, fn_rate\n')
        f.write('TRAIN'+','+str(np.mean(acc_train).round(2))+'±'+str(np.std(acc_train).round(2))
                +','+str(np.mean(ROC_train).round(2))+'±'+str(np.std(ROC_train).round(2))+','+str(np.mean(MCC_train).round(2))+'±'+str(np.std(MCC_train).round(2))+','+str(np.mean(precision_train).round(2))+'±'+
                str(np.std(precision_train).round(2))+','+str(np.mean(recall_train).round(2))+'±'
                +str(np.std(recall_train).round(2))+'\n')

        f.write('EVAL'+','+str(np.mean(acc_eval).round(2))+'±'+str(np.std(acc_eval).round(2))
                +','+str(np.mean(ROC_eval).round(2))+'±'+str(np.std(ROC_eval).round(2))+','+str(np.mean(MCC_eval).round(2))+'±'+str(np.std(MCC_eval).round(2))+','+str(np.mean(precision_eval).round(2))+'±'+
                str(np.std(precision_eval).round(2))+','+str(np.mean(recall_eval).round(2))+'±'
                +str(np.std(recall_eval).round(2))+'\n')

    return train_pred_list, val_pred_list, Y_train_list, Y_val_list


def final_test_models_multi(X_test, Y_test, model_list, log_config):

    acc_test = []
    ROC_test = []
    MCC_test = []
    precision_test = []
    recall_test = []

    df_class_rep = pd.DataFrame()

    for i, model in enumerate(model_list):
        test_pred = model.predict([X_test[:, :WINDOW_SIZE+1], X_test[:, WINDOW_SIZE+1:]])
        test_pred_arg = np.argmax(test_pred, axis=-1)

        acc_test.append(sklearn.metrics.accuracy_score(Y_test, test_pred_arg))
        ROC_test.append(sklearn.metrics.roc_auc_score(Y_test, test_pred, multi_class='ovr'))
        MCC_test.append(sklearn.metrics.matthews_corrcoef(Y_test, test_pred_arg))
        precision_test.append(sklearn.metrics.precision_score(Y_test, test_pred_arg, average='micro'))
        recall_test.append(sklearn.metrics.recall_score(Y_test, test_pred_arg, average='micro'))

        cm = confusion_matrix(Y_test, test_pred_arg)
        disp = ConfusionMatrixDisplay(confusion_matrix=cm)
        disp.plot()
        plt.savefig(log_config+'_cm_test_'+str(i)+'.png')
        plt.close()

        class_rep = classification_report(Y_test, test_pred_arg, output_dict=True)
        df_class_rep = pd.concat((df_class_rep, pd.DataFrame(data=class_rep).T))

    df_group = df_class_rep.groupby(df_class_rep.index)
    df_group.mean().to_csv(log_config+'_class_report_mean_test_set.csv')
    df_group.std().to_csv(log_config+'_class_report_std_test_set.csv')

    print('Accuracy TEST: ', np.mean(acc_test).round(2), '+-', np.std(acc_test).round(2), '\n')
    print('ROC AUC TEST: ', np.mean(ROC_test).round(2), '+- ', np.std(ROC_test).round(2), '\n')
    print('MCC TEST', np.mean(MCC_test).round(2), '+- ', np.std(MCC_test).round(2), '\n')
    print('Precision TEST', np.mean(precision_test).round(2), '+- ', np.std(precision_test).round(2), '\n')
    print('Recall TEST', np.mean(recall_test).round(2), '+- ', np.std(recall_test).round(2), '\n')

    with open(log_config+'_overall_metrics_test_set.txt', 'w') as f:
        f.write('PTM, Accuracy, ROC_AUC, PRC, MCC, precision, recall, fp_rate, fn_rate\n')
        f.write('TEST'+','+str(np.mean(acc_test).round(2))+'±'+str(np.std(acc_test).round(2))
                +','+str(np.mean(ROC_test).round(2))+'±'+str(np.std(ROC_test).round(2))+','+str(np.mean(MCC_test).round(2))+'±'+str(np.std(MCC_test).round(2))+','+str(np.mean(precision_test).round(2))+'±'+
                str(np.std(precision_test).round(2))+','+str(np.mean(recall_test).round(2))+'±'
                +str(np.std(recall_test).round(2))+'\n')


def final_test_models_multi_per_ptm(test, model_list, log_config):
    ''' Testing performance of each PTM by treating it as binary case'''

    with open(log_config+'_metrics_test_set.txt', 'w') as f:
        f.write('class, Accuracy, ROC_AUC, MCC, precision, recall, fp_rate, fn_rate\n')

    for ptm in np.unique(test.ptm.values):

        test_filtered = test[test.ptm == ptm]
        # currently re-calculating X_test, some form of masking would be faster
        X_test = return_struct_feat(test_filtered, WINDOW_SIZE, shape_len=test_filtered.shape[0])
        Y_test = test_filtered.ptm_label_negPool.values
        index = test_filtered[test_filtered.label == 1].ptm_label_negPool.values[0]
        for pos, val in enumerate(Y_test):
            if val == index:
                Y_test[pos] = 1
            else:
                Y_test[pos] = 0

        acc_test = []
        ROC_test = []
        PRC_test = []
        MCC_test = []
        precision_test = []
        recall_test = []
        fp_rate_test = []
        fn_rate_test = []

        for i, model in enumerate(model_list):
            test_pred = model.predict([X_test[:, :WINDOW_SIZE+1], X_test[:, WINDOW_SIZE+1:]])
            test_pred_arg = np.argmax(test_pred, axis=-1)
            test_pred_arg = (test_pred_arg == index).astype(np.int)
            # get only probability for particular PTM
            test_pred = test_pred[:, index]

            acc_test.append(sklearn.metrics.accuracy_score(Y_test, test_pred_arg))
            ROC_test.append(sklearn.metrics.roc_auc_score(Y_test, test_pred_arg))
            PRC_test.append(sklearn.metrics.average_precision_score(Y_test, test_pred))
            MCC_test.append(sklearn.metrics.matthews_corrcoef(Y_test, test_pred_arg))
            precision_test.append(sklearn.metrics.precision_score(Y_test, test_pred_arg))
            recall_test.append(sklearn.metrics.recall_score(Y_test, test_pred_arg))

            cm = confusion_matrix(Y_test, test_pred_arg)
            fp_rate = cm[0][1]/(cm[0][0]+cm[0][1])
            fn_rate = cm[1][0]/(cm[1][0]+cm[1][1])
            fp_rate_test.append(fp_rate)
            fn_rate_test.append(fn_rate)
            disp = ConfusionMatrixDisplay(confusion_matrix=cm)
            disp.plot()
            plt.savefig(log_config+'_'+ptm+'_cm_test_'+str(i)+'.png')
            plt.close()

        print("Metrics for only:", ptm)
        print('Accuracy TEST: ', np.mean(acc_test).round(2), '+-', np.std(acc_test).round(2), '\n')
        print('ROC AUC TEST: ', np.mean(ROC_test).round(2), '+- ', np.std(ROC_test).round(2), '\n')
        print('PRC TEST: ', np.mean(PRC_test).round(2), '+- ', np.std(PRC_test).round(2), '\n')
        print('MCC TEST', np.mean(MCC_test).round(2), '+- ', np.std(MCC_test).round(2), '\n')
        print('Precision TEST', np.mean(precision_test).round(2), '+- ', np.std(precision_test).round(2), '\n')
        print('Recall TEST', np.mean(recall_test).round(2), '+- ', np.std(recall_test).round(2), '\n')
        print('FP RATE TEST', np.mean(fp_rate_test).round(2), '+- ', np.std(fp_rate_test).round(2), '\n')
        print('FN RATE TEST', np.mean(fn_rate_test).round(2), '+- ', np.std(fn_rate_test).round(2), '\n')

        print('Best model by MCC:', np.argmax(MCC_test))

        with open(log_config+'_metrics_test_set.txt', 'a') as f:
            f.write(ptm+','+str(np.mean(acc_test).round(2))+
                    ','+str(np.mean(ROC_test).round(2))+','+str(np.mean(MCC_test).round(2))+','+str(np.mean(precision_test).round(2))+'±'+
                    ','+str(np.mean(recall_test).round(2))+
                    ','+str(np.mean(fp_rate_test).round(2))+
                    ','+str(np.mean(fn_rate_test).round(2))+
                    '\n')



class TokenAndPositionEmbedding(layers.Layer):
    def __init__(self, maxlen, vocab_size, embed_dim, **kwargs):
        super(TokenAndPositionEmbedding, self).__init__(**kwargs)
        self.maxlen = maxlen
        self.vocab_size = vocab_size
        self.embed_dim = embed_dim
        self.token_emb = layers.Embedding(input_dim=self.vocab_size, output_dim=self.embed_dim)
        self.pos_emb = layers.Embedding(input_dim=self.maxlen, output_dim=self.embed_dim)

    def get_config(self):
        config = super(TokenAndPositionEmbedding, self).get_config()
        config.update({
            "maxlen": self.maxlen,
            "vocab_size": self.vocab_size,
            "embed_dim": self.embed_dim
        })
        return config

    def call(self, x):
        maxlen = tf.shape(x)[-1]
        positions = tf.range(start=0, limit=self.maxlen, delta=1)
        positions = self.pos_emb(positions)
        x = self.token_emb(x)
        return x + positions


def lr_warmup_cosine_decay(global_step,
                           warmup_steps,
                           hold = 0,
                           total_steps=0,
                           start_lr=0.0,
                           target_lr=1e-3):
    # Cosine decay
    learning_rate = 0.5 * target_lr * (1 + np.cos(np.pi * (global_step - warmup_steps - hold) / float(total_steps - warmup_steps - hold)))
    warmup_lr = target_lr * (global_step / warmup_steps)
    if hold > 0:
        learning_rate = np.where(global_step > warmup_steps + hold,
                                 learning_rate, target_lr)

    learning_rate = np.where(global_step < warmup_steps, warmup_lr, learning_rate)
    return learning_rate


class WarmupCosineDecay(keras.callbacks.Callback):
    def __init__(self, total_steps=0, warmup_steps=0, start_lr=0.0, target_lr=1e-3, hold=0):

        super(WarmupCosineDecay, self).__init__()
        self.start_lr = start_lr
        self.hold = hold
        self.total_steps = total_steps
        self.global_step = 0
        self.target_lr = target_lr
        self.warmup_steps = warmup_steps
        self.lrs = []

    def on_batch_end(self, batch, logs=None):
        self.global_step = self.global_step + 1
        lr = self.model.optimizer.lr.numpy()
        self.lrs.append(lr)

    def on_batch_begin(self, batch, logs=None):
        lr = lr_warmup_cosine_decay(global_step=self.global_step,
                                    total_steps=self.total_steps,
                                    warmup_steps=self.warmup_steps,
                                    start_lr=self.start_lr,
                                    target_lr=self.target_lr,
                                    hold=self.hold)
        keras.backend.set_value(self.model.optimizer.lr, lr)


def get_lr_metric(optimizer):
    def lr(y_true, y_pred):
        return optimizer.lr
    return lr
