from tqdm import tqdm
from sklearn.metrics import hamming_loss
from sklearn.metrics import f1_score
import numpy as np
import torch
from utils import *
from train import *
from test import *


def test(Test_iter, Lambda, d, q, randLabels, desc='test'):
    with torch.no_grad():
        preLabels = []
        targets = []
        for X, y in tqdm(Test_iter, total=len(Test_iter), ascii=True, desc=desc):
            # for i in range(len(X)):
            for xx, yy in zip(X, y):
                preLabels.append(Pred(xx, Lambda, d, q, randLabels))
                targets.append(list(yy))

        preLabels = np.array(preLabels)
        targets = np.array(targets)

        hamming = hamming_loss(targets, preLabels)  # 汉明损失，越低越好
        f1_macro = f1_score(targets, preLabels, average="macro")  # 0.6
        f1_micro = f1_score(targets, preLabels, average="micro")

        temp = preLabels == targets
        acc_list = []
        for data in temp:
            acc = 1
            for x in data:
                acc *= x
            acc_list.append(acc)
        acc = sum(acc_list) / len(acc_list)
    return hamming, f1_macro, f1_micro, acc


def Pred(test_data, Lambda, d, q, randLabels):
    bestLabels = None
    z = Z(test_data, d, q, Lambda, randLabels)
    bestP = -1.0
    for i in range(len(randLabels)):
        fk = f_k(test_data, randLabels[i], d, q)
        temp_P = torch.exp((Lambda * fk).sum()) / z
        if temp_P > bestP:
            bestP = temp_P
            bestLabels = randLabels[i]
    return np.array(bestLabels)
