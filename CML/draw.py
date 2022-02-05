import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

def draw(file_name, sava_name, sava_name2):
    df = pd.read_csv(file_name)
    df.head(2)

    hamming_test_list = np.array(df['hamming_test_list'])
    hamming_train_list = np.array(df['hamming_train_list'])
    f1_macro_test_list = np.array(df['f1_macro_test_list'])
    f1_macro_train_list = np.array(df['f1_macro_train_list'])
    f1_micro_test_list = np.array(df['f1_micro_test_list'])
    f1_micro_train_list = np.array(df['f1_micro_train_list'])
    acc_test_list = np.array(df['acc_test_list'])
    acc_train_list = np.array(df['acc_train_list'])

    x =range(len(hamming_test_list))

    plt.rcParams['font.size'] = 20.

    plt.rcParams['figure.figsize'] = (20, 14) #aA paper

    plt.subplots_adjust(left=None, bottom=None, right=None, top=None, wspace=0.3, hspace=0.5)

    plt.subplot(221)
    plt.plot(x, hamming_train_list)
    plt.plot(x, hamming_test_list)
    plt.title('hamming loss each epoch')
    plt.xlabel('epochs')
    plt.ylabel('hamming loss')
    plt.legend(['train', 'test'])

    plt.subplot(222)
    plt.plot(x, f1_macro_train_list)
    plt.plot(x, f1_macro_test_list)
    plt.title('f1 macro score each epoch')
    plt.xlabel('epochs')
    plt.ylabel('f1 macro score')
    plt.legend(['train', 'test'])

    plt.subplot(223)
    plt.plot(x, f1_micro_train_list)
    plt.plot(x, f1_micro_test_list)
    plt.title('f1 micro score each epoch')
    plt.xlabel('epochs')
    plt.ylabel('f1 micro score')
    plt.legend(['train', 'test'])

    plt.subplot(224)
    plt.plot(x, acc_train_list)
    plt.plot(x, acc_test_list)
    plt.title('accuracy each epoch')
    plt.xlabel('epochs')
    plt.ylabel('accuracy')
    plt.legend(['train', 'test'])

    plt.savefig(sava_name)
    plt.close()

    plt.rcParams['font.size'] = 20.

    plt.rcParams['figure.figsize'] = (12, 8) #aA paper

    plt.plot(x, acc_train_list)
    plt.title('train and test accuracy each epoch')
    plt.xlabel('epochs')
    plt.ylabel('accuracy')

    plt.plot(x, acc_test_list)
    plt.xlabel('epochs')
    plt.ylabel('accuracy')

    plt.legend(['train accuracy', 'test accuracy'])
    plt.savefig(sava_name2)

draw('result/Heud2000/info/2021_12_13_01_17_25epochs_20_lr0.001_traininfo.csv', 'result/Heud2000/info/2021_12_13_01_17_25epochs_20_lr0.001_train.png'
, 'result/Heud2000/info/2021_12_13_01_17_25epochs_20_lr0.001_acc.png')

