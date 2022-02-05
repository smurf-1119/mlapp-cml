from tqdm import tqdm
from utils import *
import time
from test import test
import pandas as pd

# 求目标函数l(Lambda|D)
def obj_func(DataSets, Labels, thegma, Lambda, randLabels):
    """
    :param q:标签集的维度
    :param DataSets:所有训练样本的特征集
    :param Labels:所有训练样本的标签集
    :param thegma:自己给定的参数值，2**-6,2**-5,2**-4,2**-3,2**-2,2**-1,2**1,2**2,2**3,2**4,2**5,2**6逐个取值，参数寻优
    :return:目标函数，以及待定参数Lambda
    """
    samples = len(DataSets)
    d = len(DataSets[0])
    q = len(Labels[0])
    temp_sum = 0
    for i in range(samples):
        fk = f_k(DataSets[i], Labels[i], d, q)
        z = Z(DataSets[i], d, q, Lambda, randLabels)
        temp_sum = temp_sum + (Lambda * fk).sum() - torch.log2(z)
        temp_div = (
            (Lambda * Lambda) / (2 * thegma ** 2)
        ).sum()  # temp_div=sum(Lambda**2/(2*thegma**2))

    l = -(temp_sum - temp_div)
    return l  # 求解l的最大值，可以转化为求-l的最小值问题


def Train(
    objfunc,
    train_iter,
    test_iter,
    num_epochs,
    optimizer,
    thegma,
    Lambda,
    randLabels,
    d,
    q,
):
    train_l_list = []
    hamming_train_list = []
    f1_macro_train_list = []
    f1_micro_train_list = []
    acc_train_list = []
    hamming_test_list = []
    f1_macro_test_list = []
    f1_micro_test_list = []
    acc_test_list = []

    for epoch in range(num_epochs):
        train_l_sum, n, start = 0.0, 0, time.time()
        
        for X, y in tqdm(train_iter, total=len(train_iter), ascii=True, desc="train"):

            loss = objfunc(X, y, thegma, Lambda, randLabels)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            train_l_sum += loss.cpu().data
            n += y.shape[0]

        hamming_train, f1_macro_train, f1_micro_train, acc_train = test(train_iter, Lambda, d, q, randLabels, 'train_test')

        hamming_train_list.append(hamming_train)
        f1_macro_train_list.append(f1_macro_train)
        f1_micro_train_list.append(f1_micro_train)
        acc_train_list.append(acc_train)
        train_l_list.append(train_l_sum / n)

        print(
            "TRAIN: epoch %d, time %.1f sec, train_loss %.2f, hamming_train %.2f, f1_macro_train %.2f, f1_micro_train %.2f, subset acc_train %.4f acc"
            % (
                epoch + 1,
                time.time() - start,
                train_l_sum / n,
                hamming_train,
                f1_macro_train,
                f1_micro_train,
                acc_train,
            )
        )

        hamming_test, f1_macro_test, f1_micro_test, acc_test = test(test_iter, Lambda, d, q, randLabels, 'test')

        hamming_test_list.append(hamming_test)
        f1_macro_test_list.append(f1_macro_test)
        f1_micro_test_list.append(f1_micro_test)
        acc_test_list.append(acc_test)

        print(
            "TEST: epoch %d, hamming_test %.2f, f1_macro_test %.2f, f1_micro_test %.2f, subset acc_test %.4f acc"
            % (
                epoch + 1,
                hamming_test,
                f1_macro_test,
                f1_micro_test,
                acc_test,
            )
        )
        print(Lambda)
        # try:
        #     acc_test_list[-1] - acc_test_list[-2] < -0.3
        #     print('early stop')
        #     break
        # except:
        #     pass
    
    

    train_info = {}
    train_info['hamming_train_list'] = hamming_train_list
    train_info['f1_macro_train_list'] = f1_macro_train_list
    train_info['f1_micro_train_list'] = f1_micro_train_list
    train_info['acc_train_list'] = acc_train_list
    train_info['hamming_test_list'] = hamming_test_list
    train_info['f1_macro_test_list'] = f1_macro_test_list
    train_info['f1_micro_test_list'] = f1_micro_test_list
    train_info['acc_test_list'] = acc_test_list
    train_info['train_l_list'] = train_l_list

    return Lambda, train_l_list, train_info
