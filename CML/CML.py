import torch
import argparse
from train import *
import time
import os
import matplotlib.pyplot as plt

def set_interact_args():
    """
    Sets up the training arguments.
    """
    parser = argparse.ArgumentParser()
    parser.add_argument( "--data_type", default="Reuters2000", type=str, required=False, help="输入数据类型")
    parser.add_argument("--rand_labels", default="basic", type=str, required=False, help="所有标签组合")
    parser.add_argument("--model", default='2021_12_12_23_04_43_Lambda.pt', type=str, required=False, help="载入参数")
    parser.add_argument("--train", default='true', type=str, required=False, help="是否训练")
    parser.add_argument("--load_para", default='true', type=str, required=False, help="是否训练")
    parser.add_argument("--lr", default=0.001, type=float, required=False, help="学习率")
    parser.add_argument("--epochs", default=20, type=int, required=False, help="epcoch数")
    parser.add_argument("--batch_size", default=32, type=int, required=False, help="batch大小")


    return parser.parse_args()

def draw_img(train_l_list, img_name, data_type):
    img_name = os.path.join('./result', data_type, 'imgs', img_name)
    plt.plot(range(len(train_l_list)), train_l_list)
    plt.title('Train_loss')
    plt.xlabel('Epochs')
    plt.ylabel('loss')
    plt.savefig(img_name)


def main():
    # --------------------------------------------------------------------------------
    # -----------------------------------载入超参--------------------------------------
    # --------------------------------------------------------------------------------
    # 获取数据路径
    args = set_interact_args()
    data_dir = os.path.join('./data', args.data_type)
    train_data_path = os.path.join(data_dir, "train", args.data_type + "_traindata.npy")
    train_label_path = os.path.join(data_dir, "train", args.data_type + "_trainlabel.npy")
    test_data_path = os.path.join(data_dir, "test", args.data_type + "_testdata.npy")
    test_label_path = os.path.join(data_dir, "test", args.data_type + "_testlabel.npy")

    # 获取所有标签组合
    randLabels = generate_rand_Labels(args.rand_labels)

    # 获取学习率
    lr = args.lr

    # 获取模型地址
    Lambda_path = os.path.join('./result', args.data_type, 'Lambda', args.model)

    # 是否训练
    mode = args.train

    # epoch数
    num_epochs = args.epochs

    # batch_size
    batch_size = args.batch_size

    # --------------------------------------------------------------------------------
    # --------------------------------------------------------------------------------

    # 载入数据
    train_iter, test_iter, K, thegma, d, q = load_data(
        train_data_path, train_label_path, test_data_path, test_label_path, batch_size
    )
    
    # 初始化参数
    if args.load_para == 'true':
        Lambda = torch.load(Lambda_path)
    else:
        Lambda = torch.zeros( K, requires_grad=True)  
    

    if mode == 'true':
        # 定义优化器
        optimizer = torch.optim.Adam([Lambda], lr=lr, betas=(0.9, 0.999), eps=1e-08)

        # 训练
        Lambda, train_l_list, train_info = Train(
            obj_func,
            train_iter,
            test_iter,
            num_epochs,
            optimizer,
            thegma,
            Lambda,
            randLabels,
            d,
            q,
        )

        print(Lambda)
        start = time.time()
        timeArray = time.localtime(start)
        timedate = time.strftime("%Y-%m-%d %H:%M:%S",timeArray)
        timedate = '{}'.format(timedate).replace(' ', '_').replace('-', '_').replace(':','_')

        draw_img(train_l_list, timedate+'_loss.png', args.data_type)
        pd.DataFrame(train_info).to_csv(os.path.join('./result', args.data_type, 'info', timedate+'epochs:{0}_lr{1}_traininfo.csv').format(num_epochs, lr))

        torch.save(Lambda, os.path.join("./result",args.data_type,"Lambda", timedate + "epochs:{0}_lr{1}__Lambda.pt".format(num_epochs, lr)))

    # ([ 1.3684e-03,  1.0080e-01,  2.1870e-03, -1.9966e-02,  9.5491e-03,
    #         -3.5565e-01,  5.6930e-02,  8.2445e-02, -9.0433e-03,  7.7105e-02,
    #          5.3024e-02, -5.3038e-02, -7.3672e-02,  1.1402e-01, -1.4662e-01,
    #          9.3514e-02, -4.7597e-02, -1.3051e-03,  2.2363e-02, -2.6729e-02,
    #         -1.4027e+01, -1.3943e+01, -1.4388e+01, -1.4196e+01, -1.4172e+01,
    #         -1.3829e+01, -1.4311e+01, -1.4092e+01, -1.4597e+01, -1.3747e+01,
    #         -1.4072e+01, -1.4278e+01, -1.4439e+01, -1.4082e+01, -1.4181e+01,
    #         -1.3875e+01, -1.4323e+01, -1.4137e+01, -1.4269e+01, -1.3813e+01,
    #         -1.4222e+01, -1.4327e+01, -1.4379e+01, -1.3603e+01],
    #        requires_grad=True)
    # 目前最好的参数

    # 测试
    hamming, f1_macro, f1_micro, acc = test(test_iter, Lambda, d, q, randLabels)
    print(
        "hamming %.3f, f1_macro %.3f, f1_micro %.3f, subset acc %.3f"
        % (hamming, f1_macro, f1_micro, acc)
    )


if __name__ == "__main__":
    main()
