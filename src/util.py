import datetime
import itertools
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle,resample


def get_device():
    if torch.cuda.is_available():

        # Tell PyTorch to use the GPU.
        device = torch.device("cuda")

        print('There are %d GPU(s) available.' % torch.cuda.device_count())

        print('We will use the GPU:', torch.cuda.get_device_name(0))

    # If not...
    else:
        print('No GPU available, using the CPU instead.')
        device = torch.device("cpu")
    return device


def over_sampling(df,seed_val):
    """
    remove Missing value first, then output two balanced dataset (Undersampling and Oversampling)
    Input: X,y before pre-processing
    Output: dataframes after removing missing value, Undersampling and Oversampling
    """
    df_true = df[df['y'] == True]
    df_false = df[df['y'] == False]

    # Upsampling, for the class with less data, copy some data
    df_false_upsampled = resample(df_false, random_state=seed_val, n_samples=len(df_true), replace=True)
    df_upsampled = pd.concat([df_false_upsampled, df_true])
    df_upsampled = shuffle(df_upsampled)

    print('\nWe totally have {} training data after oversampling.'.format(len(df_upsampled)))
    return df_upsampled


def transform_df(df):
    # transform label to int
    df.loc[df['y'] == 'True', 'y'] = 1
    df.loc[df['y'] == 'False', 'y'] = 0
    df.y = df.y.astype(int)
    return df


def plot_confusion_matrix(cm,
                          target_names,
                          title='Confusion matrix',
                          cmap=None,
                          normalize=True):
    if cmap is None:
        cmap = plt.get_cmap('Blues')

    plt.figure(figsize=(8, 6))
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()

    if target_names is not None:
        tick_marks = np.arange(len(target_names))
        plt.xticks(tick_marks, target_names, rotation=45)
        plt.yticks(tick_marks, target_names)

    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]

    thresh = cm.max() / 1.5 if normalize else cm.max() / 2
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        if normalize:
            plt.text(j, i, "{:0.4f}".format(cm[i, j]),
                     horizontalalignment="center",
                     color="white" if cm[i, j] > thresh else "black")
        else:
            plt.text(j, i, "{:,}".format(cm[i, j]),
                     horizontalalignment="center",
                     color="black" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.show()
    return


def format_time(elapsed):
    '''
    Takes a time in seconds and returns a string hh:mm:ss
    '''
    # Round to the nearest second.
    elapsed_rounded = int(round((elapsed)))

    # Format as hh:mm:ss
    return str(datetime.timedelta(seconds=elapsed_rounded))


def train_val_test(train_ratio, validation_ratio, test_ratio, X, y,seed_val):
    """
    remove Missing value first, then output two balanced dataset (Undersampling and Oversampling)
    Input: X,y before pre-processing
    Output: dataframes after removing missing value, Undersampling and Oversampling
    """

    df = pd.DataFrame({'X': pd.Series(X), 'y': pd.Series(y)})


    df_train, df_test = train_test_split(df, test_size=1 - train_ratio, random_state=seed_val)
    df_val, df_test = train_test_split(df_test, test_size=test_ratio / (test_ratio + validation_ratio),
                                       random_state=seed_val)

    X_train, y_train = df_train['X'], df_train['y']
    X_val, y_val = df_val['X'], df_val['y']
    X_test, y_test = df_test['X'], df_test['y']

    print('[X training set shape, X validation set shape, X test set shape]:', y_train.shape, y_val.shape, y_test.shape)

    df_train = pd.DataFrame({'X': pd.Series(X_train), 'y': pd.Series(y_train)})
    df_train = transform_df(df_train)

    # transform testset to right form
    df_val = pd.DataFrame({'X': pd.Series(X_val), 'y': pd.Series(y_val)})
    df_val = transform_df(df_val)

    df_test = pd.DataFrame({'X': pd.Series(X_test), 'y': pd.Series(y_test)})
    df_test = transform_df(df_test)

    return df_train, df_val, df_test