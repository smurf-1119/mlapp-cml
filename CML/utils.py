import numpy as np
import torch.utils.data as Data
import torch
from sklearn.decomposition import PCA
import numpy as np
import torch.utils.data as Data
from scipy.special import comb  # 排列组合中的组合公式


def f_k(dataSet, Labels, d, q):
    """
    :param dataSet: 某一个样本的特征集
    :param Labels: 某一个样本的标签集
    :param d: 样本的维度，即一个样本含有的特征数
    :param q: 标签的维度，即标签集中标签的个数
    :return: 返回的是fk(x,y)
    """
    F_k = []
    for l in range(d):
        for j in range(q):
            if Labels[j] == 1:
                try:
                    F_k.append(float(dataSet[l]))
                except:
                    # print(dataSet)
                    # print(l, dataSet)
                    raise IndexError

            else:
                F_k.append(0.0)

    for j1 in range(q - 1):
        for j2 in range(j1 + 1, q):
            y_j1 = Labels[j1]
            y_j2 = Labels[j2]
            if y_j1 == 1 and y_j2 == 1:
                F_k.append(1.0)
            else:
                F_k.append(0.0)
            if y_j1 == 1 and y_j2 == 0:
                F_k.append(1.0)
            else:
                F_k.append(0.0)
            if y_j1 == 0 and y_j2 == 1:
                F_k.append(1.0)
            else:
                F_k.append(0.0)
            if y_j1 == 0 and y_j2 == 0:
                F_k.append(1.0)
            else:
                F_k.append(0.0)
    # print(len(F_k))
    return torch.tensor(F_k, requires_grad=True)


def basic_rand_labels(len):
    """
    变成辅助函数
    #关于这个函数的for循环的嵌套次数，Y标签集中，有几个标签就嵌套几层。（y1,y2,...,yq）
    :return: 返回的是q维的标签集的所有组合情况
    """
    """
    randLabels=[]
    for i in range(2):
        randLabels.append([i])
    return randLabels
    """
    randLabels = []
    for i in range(2 ** len):
        randLabel = np.zeros(shape=len)
        for j in range(len):
            randLabel[len - j - 1] = i % 2
            i = i // 2
            if i == 0:
                break
        print(randLabel)
        randLabels.append(randLabel)
    np.save("./basic_rand_Labels.npy", np.array(randLabels))


def supported_rand_labels(train_label):
    """
    这是个辅助函数，用来生成support_rand_Labels
    """
    """
    randLabels=[]
    for i in range(2):
        randLabels.append([i])
    return randLabels
    """
    # for _, y in train_iter:
    labels = train_label.tolist()
    label_set = []
    for label in labels:
        if label in label_set:
            continue
        else:
            label_set.append(label)
    randLables = np.array(label_set)
    print(label_set)
    np.save("./supported_rand_labels.npy", randLables)
    print("finish")


def generate_rand_Labels(mode):
    if mode == "supported":
        randLabels = np.load("./data/rand_labels/supported_rand_labels.npy")
        randLabels = randLabels.tolist()
        return randLabels
    elif mode == "basic":
        randLabels = np.load("./data/rand_labels/basic_rand_Labels.npy")
        randLabels = randLabels.tolist()
        return randLabels


def Z(dataSet, d, q, Lambda, randLabels):  # 对于某一个样本的Z
    """
    :param dataSet: 某一个样本的特征集
    :param d: 样本的维度，即特征的个数
    :param q: 标签集的个数
    :param Lambda: Lambda是一个1*K维向量
    :return: 归一化范数，以及所有标签集组合的对应f_k
    """
    Z = 0
    for i in range(len(randLabels)):
        fk = f_k(dataSet, randLabels[i], d, q)
        temp_sum = torch.exp((Lambda * fk).sum())
        Z = Z + temp_sum
    return Z


def load_data(train_data_path, train_label_path, test_data_path, test_label_path, batch_size):
    # 训练集的处理

    train_data = np.load(train_data_path)
    train_data = train_data[:, :]
    train_label = np.load(train_label_path)
    train_label = train_label[:, :]
    test_data = np.load(test_data_path)
    test_data = test_data[:, :]
    test_label = np.load(test_label_path)
    test_target = test_label[:, :]

    train_target = torch.tensor(train_label, dtype=torch.float, requires_grad=True)
    test_target = torch.tensor(test_label, dtype=torch.float, requires_grad=True)

    # 主成分降维
    pca = PCA(n_components=5)  # 保留5个主成分
    train_data = torch.tensor(pca.fit_transform(train_data), requires_grad=True)
    test_data = torch.tensor((pca.fit_transform(test_data)), requires_grad=True)

    d = len(train_data[0])
    q = len(train_target[0])
    K = int(d * q + 4 * comb(q, 2))
    thegma = 2 ** (1)  # 参数寻优，-6,-5,-4,-3,-2,-1,0,1,2,3,4,5,6

    # 训练数据集
    dataset = Data.TensorDataset(train_data, train_target)
    # 测试集的处理
    test_data = Data.TensorDataset(test_data, test_target)
    # test_data = Data.TensorDataset(torch.tensor(test_data, requires_grad=True))

    # 随机读取小批量
    train_iter = Data.DataLoader(dataset, batch_size, shuffle=True)
    test_iter = Data.DataLoader(test_data, batch_size, shuffle=False)

    return train_iter, test_iter, K, thegma, d, q
