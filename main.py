import numpy as np
import pandas as pd

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

import networkx as nx
import dgl
from tqdm import tqdm
import pickle
import argparse
import csv
import pandas

from sklearn.metrics import f1_score
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import label_binarize
from sklearn import metrics
import matplotlib.pyplot as plt
from sklearn.model_selection import StratifiedKFold
import matplotlib as mpl

from model import GAT_LSTM, LSTMbaseline
from model import GAT_Transformer
from model import GCN_LSTM
from model import GCN_Transformer
from model import GAT_dgl_LSTM
from model import GAT_dgl_Transformer
from model import GCN_baseline
import os

from torchcam.methods import GradCAM




if __name__ == '__main__':
    parser = argparse.ArgumentParser()


    os.environ['CUDA_VISIBLE_DEVICES'] = '7'
    device = torch.device("cuda:7" if torch.cuda.is_available() else "cpu")


    """data114   114sample, 116node"""
    """data705   582sample, 90node"""
    """data renji"""

    # label_df_path = "./../dataset/data114/window_37_step_10/label_df.pkl"
    # adj_graphs_path = "./../dataset/data114/window_37_step_10/FC_adj_graphs_df.pkl"
    # label_df_path = "./../dataset/data705/window_37_step_10/label_df.pkl"
    # adj_graphs_path = "./../dataset/data705/window_37_step_10/FC_adj_graphs_df.pkl"
    #
    label_df_path = "./../dataset/Renji processed func data/window_50_step_10/label_df.pkl"
    adj_graphs_path = "./../dataset/Renji processed func data/window_50_step_10/FC_adj_graphs_df.pkl"


    parser.add_argument('--period', type=int, default=17)  #  time=10   segment_num = 11    17

    parser.add_argument('--df_path', type=str, default=label_df_path)
    parser.add_argument('--graphs_path', type=str, default=adj_graphs_path)

    parser.add_argument('--sample_num', type=int, default=221)  # sample_num=114  90   221
    parser.add_argument('--gcn_in', type=int, default=100)  # node_num=116  90   100
    # parser.add_argument('--gcn_in', type=int, default=90)  # node_num=116  90

    parser.add_argument('--gcn_hid', type=int, default=32)  # default=32
    parser.add_argument('--gcn_out', type=int, default=16)  # default=16
    parser.add_argument('--gcn_layers', type=int, default=2)  # default=2
    parser.add_argument('--lstm_hid', type=int, default=16)  # default=16
    # parser.add_argument('--num_classes', type=int, default=2)  # default=2
    parser.add_argument('--num_classes', type=int, default=2)  # default=3


    parser.add_argument('--lstm_layers', type=int, default=2)  # default=2
    parser.add_argument('--activation', type=str, default='relu')  # default=0
    parser.add_argument('--model_drop', type=int, default=0.5)  # default=0\

    parser.add_argument('--head_num', type=str, default='1')
    parser.add_argument('--transform_layer', type=str, default='16')   #16
    parser.add_argument('--temporal_drop', type=float, default=0.5)
    parser.add_argument('--residual', type=bool, default=True)

    parser.add_argument('--gat_drop', type=float, default=0.5)  # can increase
    parser.add_argument('--gat_alpha', type=float, default=0.8)  # default=0.0001
    parser.add_argument('--gat_nheads', type=int, default=4)  # default=2

    parser.add_argument('--gat_dgl_num_layers', type=int, default=2)  # default=2
    parser.add_argument('--gat_dgl_heads', type=int, default=4)  # default=2
    parser.add_argument('--gat_activation', type=str, default='LeakyReLU')  # default=2
    parser.add_argument('--gat_dgl_feat_drop', type=float, default=0.2)  # default=2
    parser.add_argument('--gat_dgl_attn_drop', type=float, default=0.)  # default=2
    parser.add_argument('--gat_dgl_alpha', type=float, default=0.2)  # default=2
    parser.add_argument("--edge_feature_attn", action="store_true", default=False)
    parser.add_argument("--gat_dgl_residual", action="store_true", default=True)


    parser.add_argument('--epoch', type=int, default=10)  # can increase
    parser.add_argument('--lr', type=float, default=0.001)  # default=0.0001
    parser.add_argument('--alpha', type=float, default=0.5)  # default=2



    args = parser.parse_args()


    # load data
    df = pd.read_pickle(
        args.df_path)  # [10 row * 4 column]  label, action1 list, action2 list, action3 list, action4 list
    # if args.macro:
    #     macro = pd.read_pickle(
    #         args.macro_path).to_numpy()  # 10 * 4 array  macro_feat1, macro_feat2, macro_feat3, macro_feat4
    graphs_sep = pickle.load(open(args.graphs_path, 'rb'))  # {id: list of networkx graphs}  10 * 7graph


    window_size = 50     #renji 50    data114,705  37
    step = 10
    # window_path = "./../dataset/data114/window_" + str(window_size) + "_step_" + str(step)
    # window_path = "./../dataset/data705/window_" + str(window_size) + "_step_" + str(step)
    window_path = "./../dataset/Renji processed func data/window_" + str(window_size) + "_step_" + str(step)



    FC_dgls_path = window_path + "/FC_dgls.pkl"
    EC_dgls_path = window_path + "/EC_dgls.pkl"
    FC_inputs_path = window_path + "/FC_inputs.pkl"
    EC_inputs_path = window_path + "/EC_inputs.pkl"
    dgls = pd.read_pickle(FC_dgls_path)
    inputs = pd.read_pickle(FC_inputs_path)

    dgls_EC = pd.read_pickle(EC_dgls_path)
    inputs_EC = pd.read_pickle(EC_inputs_path)






    """data114 2class"""                         #60 0label NC       54 1label   MCI
    # zero_num = 0
    # one_num = 0
    # for line in range(len(df['label'].values)):
    #     if (df['label'].values[line].astype(int) == 1 or df['label'].values[line].astype(int) == 2):
    #         df['label'].values[line] = 0
    #         zero_num += 1
    #     else:
    #         df['label'].values[line] = 1
    #         one_num += 1
    #
    # print("zero_label_num:", zero_num)
    # print("one_label_num:", one_num)



    # """data705 2class"""
    # NC_index = []     #0
    # EMCI_index = []   #1
    # LMCI_index = []   #2
    #
    # EC_index_with_nan = [60, 227,290,317,329]     #61_5,61_6,228_9,228_10,291_7,291_8,318_1,318_2,318_3,318_4,330_1,330_2
    #
    # for line in range(len(df['label'].values)):
    #     if (df['label'].values[line].astype(int) == 0) and line not in EC_index_with_nan:
    #         NC_index.append(line)
    #     elif (df['label'].values[line].astype(int) == 1) and line not in EC_index_with_nan:
    #         EMCI_index.append(line)
    #     elif (df['label'].values[line].astype(int) == 2) and line not in EC_index_with_nan:
    #         LMCI_index.append(line)
    #         df['label'].values[line] = 1
    #     else:
    #         continue



    """dataRenji processed func data 2class"""                         #60 0label NC       54 1label   MCI
    zero_num = 0
    one_num = 0
    for line in range(len(df['label'].values)):
        if (df['label'].values[line].astype(int) == 1 or df['label'].values[line].astype(int) == 2):
            df['label'].values[line] = 0
            zero_num += 1
        else:
            df['label'].values[line] = 1
            one_num += 1

    print("zero_label_num:", zero_num)
    print("one_label_num:", one_num)









    Y_list = df['label']
    total_test_label = []
    total_predict_label = []
    total_predict_proba_label = np.empty(shape=(0, args.num_classes))
    total_predict_proba_label_save = np.empty(shape=(0, args.num_classes))

    test_acc_list = []
    total_test_proba_label = np.empty(shape=(0, args.num_classes))


    """five fold"""
    skf = StratifiedKFold(n_splits=10, random_state=8, shuffle=True)


    """renji split"""
    sample_list = [0 for i in range(197)]
    for train_index, test_index in skf.split(sample_list, sample_list):
    # for train_index, test_index in skf.split(Y_list, Y_list):
        print("length train_subject_index",len(train_index))
        print("train_subject_index",train_index)
        print("length val_subject_indx", len(test_index))
        print("val_subject_indx", test_index)


        """renji subject 1vs all scan"""
        dict_index_sub_scan_path = "./../dataset/Renji processed func data/dict_index_sub_scan.csv"
        dict = pd.read_csv(dict_index_sub_scan_path, header=None)

        train_index_new = []
        for i in train_index:
            for j in range(1, len(dict.iloc[:, i + 1])):
                if not pd.isna(dict.iloc[j, i + 1]):
                    train_index_new.append(int(dict.iloc[j, i + 1]))
        # print(train_index_new)

        test_index_new = []
        for i in test_index:
            for j in range(1, len(dict.iloc[:, i + 1])):
                if not pd.isna(dict.iloc[j, i + 1]):
                    test_index_new.append(int(dict.iloc[j, i + 1]))
        # print(test_index_new)

        train_index = train_index_new
        test_index = test_index_new

        print("length train_scan_index",len(train_index))
        print("train_scan_index",train_index)
        print("length val_scan_indx", len(test_index))
        print("val_scan_indx", test_index)






        # # train, test split
        # n = len(dgls)
        # index = np.arange(n)
        # # index = NC_index + LMCI_index
        #
        # n = len(index)
        # split = int(n * .9)
        #
        # np.random.seed(16)
        # np.random.shuffle(index)
        #
        #
        # train_index, test_index = index[:split], index[split:]
        #
        # print("total graph num:", n)
        # print("train_index:", train_index)
        # print("test_index:", test_index)


        train_labels, test_labels = Variable(torch.LongTensor((df['label'].values.astype(int))[train_index])), Variable(
            torch.LongTensor((df['label'].values.astype(int))[test_index]))


        print("train_labels", train_labels)
        print("test_label", test_labels)

        # prep temporal graph data
        k = args.period
        trainGs, testGs = [dgls[i] for i in train_index], [dgls[i] for i in test_index]  # trainGs 8 * 7
        print("trainGs", (len(trainGs), len(trainGs[0])))          #sample* time

        trainGs, testGs = [dgl.batch([u[i].int().to(device) for u in trainGs]) for i in range(k)], \
                          [dgl.batch([u[i].int().to(device) for u in testGs]) for i in range(k)]
        train_inputs, test_inputs = [inputs[i] for i in train_index], [inputs[i] for i in test_index]
        train_inputs, test_inputs = [torch.FloatTensor(np.concatenate([inp[i] for inp in train_inputs])).to(device) for i in range(k)], \
                                    [torch.FloatTensor(np.concatenate([inp[i] for inp in test_inputs])).to(device) for i in range(k)]

        print("trainGs input length", len(trainGs))   # time
        print("trainGs input size", trainGs[0])       # sample num graphs consist one big graph

        trainGs_EC, testGs_EC = [dgls_EC[i] for i in train_index], [dgls_EC[i] for i in test_index]  # trainGs 8 * 7
        print("trainGs_EC", (len(trainGs_EC), len(trainGs_EC[0])))

        trainGs_EC, testGs_EC = [dgl.batch([u[i].int().to(device) for u in trainGs_EC]) for i in range(k)], \
                                [dgl.batch([u[i].int().to(device) for u in testGs_EC]) for i in range(k)]
        train_inputs_EC, test_inputs_EC = [inputs_EC[i] for i in train_index], [inputs_EC[i] for i in test_index]
        train_inputs_EC, test_inputs_EC = [torch.FloatTensor(np.concatenate([inp[i] for inp in train_inputs_EC])).to(device) for i in
                                           range(k)], \
                                          [torch.FloatTensor(np.concatenate([inp[i] for inp in test_inputs_EC])).to(device) for i in
                                           range(k)]

        print("test_inputs", test_inputs[0])




        # nx_trainGs = [dgl.to_networkx(trainGs[i].cpu()) for i in range(len(trainGs))]
        # nx_testGs = [dgl.to_networkx(testGs[i].cpu()) for i in range(len(testGs))]
        # nx_trainGs_EC = [dgl.to_networkx(trainGs_EC[i].cpu()) for i in range(len(trainGs_EC))]
        # nx_testGs_EC = [dgl.to_networkx(testGs_EC[i].cpu()) for i in range(len(testGs_EC))]
        #
        # adj_matrix_nx_trainGs = np.array(nx.adjacency_matrix(nx_trainGs).todense())
        # print("adj_matrix_nx_trainGs.shape", adj_matrix_nx_trainGs.shape)
        # adj_matrix_nx_testGs = np.array(nx.adjacency_matrix(nx_testGs).todense())
        # print("adj_matrix_nx_testGs.shape", adj_matrix_nx_testGs.shape)
        # adj_matrix_nx_trainGs_EC = np.array(nx.adjacency_matrix(nx_trainGs_EC).todense())
        # print("adj_matrix_nx_trainGs_EC.shape", adj_matrix_nx_trainGs_EC.shape)
        # adj_matrix_nx_testGs_EC = np.array(nx.adjacency_matrix(nx_testGs_EC).todense())
        # print("adj_matrix_nx_testGs_EC.shape", adj_matrix_nx_testGs_EC.shape)
        #
        #
        # trainGs = torch.from_numpy(adj_matrix_nx_trainGs).to(device)
        # testGs = torch.from_numpy(adj_matrix_nx_testGs).to(device)
        # trainGs_EC = torch.from_numpy(adj_matrix_nx_trainGs_EC).to(device)
        # testGs_EC = torch.from_numpy(adj_matrix_nx_testGs_EC).to(device)
        #
        #

        if args.activation == 'relu':
            activation = F.relu

        # model = GCN_LSTM(args.gcn_in, args.gcn_hid, args.gcn_out, args.gcn_layers,
        #                  args.lstm_hid, args.num_classes, args.lstm_layers, activation, args.model_drop)


        model = GCN_Transformer(args.gcn_in, args.gcn_hid, args.gcn_out, args.gcn_layers, activation,
                         args.period, args.head_num, args.transform_layer, args.temporal_drop, args.residual, args.num_classes, args.model_drop)




        # model = GAT_dgl_LSTM(args.gcn_in, args.gat_dgl_num_layers, args.gcn_in, args.gcn_hid, args.gcn_out,
        #                      args.gat_dgl_heads, args.gat_activation, args.gcn_out, args.lstm_hid,
        #                      args.num_classes, args.lstm_layers, activation, args.model_drop)


        # model = GAT_dgl_Transformer(args.gcn_in, args.gat_dgl_num_layers, args.gcn_in, args.gcn_hid, args.gcn_out,
        #                      args.gat_dgl_heads, args.gat_activation, args.gcn_out,
        #                             args.period, args.head_num, args.transform_layer, args.temporal_drop, args.residual, args.num_classes, args.model_drop)



        # model = GCN_baseline(args.gcn_in, args.gcn_hid, args.gcn_in*args.gcn_in, args.gcn_layers,
        #                  args.num_classes, activation, args.model_drop, args.period)




        # model = LSTMbaseline(args.gcn_in, args.gcn_hid, args.gcn_out, args.gcn_layers,
        #                  args.lstm_hid, args.num_classes, args.lstm_layers, activation, args.model_drop)

        # if torch.cuda.device_count() > 1:
        #     print("Use GPUs: ", torch.cuda.device_count())
        #     model = nn.DataParallel(model)

        model.cuda()

        parameters = list(model.parameters())
        optimizer = torch.optim.Adam(parameters, lr=args.lr, weight_decay=0.5)    #transfomer 0.8 1.0
        # optimizer = torch.optim.Adam(parameters, lr=args.lr)    #0.8
        scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[50, 100], gamma=0.8)


        train_loss_list = []
        test_loss_list = []
        temporal_acc = 0.
        temporal_predict_label = []
        temporal_predict_proba_label = np.empty(shape=(0,args.num_classes))
        temporal_predict_proba_label_save = np.empty(shape=(0,args.num_classes))

        early_stop = 0

        iterator = -1
        for epoch in tqdm(range(args.epoch)):

            iterator += 1

            # train
            model.train()

            k = args.period
            out, ST_dependecny,  a_grad_1, a_grad_2, h_node_tensor_1, h_node_tensor_2 = model(trainGs, train_inputs, trainGs_EC, train_inputs_EC, k)

            # print("out", out)


            # logp = F.log_softmax(out.to(device), dim=1)
            logp = F.softmax(out.to(device), dim=1)


            # logp = F.softmax(out, dim=1)
            print("logp size", np.shape(logp))
            # print("logp", logp)
            # print("logp2", logp2)
            print("train_labels size", np.shape(train_labels))

            # regular_loss = 0
            # for param in model.parameters():
            #     regular_loss += torch.sum(abs(param))

            # print("logp", logp)
            # print(train_labels)

            # loss = F.nll_loss(logp.to(device), train_labels.to(device))
            loss = F.cross_entropy(logp.to(device), train_labels.to(device))

            # loss = loss + args.alpha * regular_loss


            # print(logp)
            # print(train_labels)
            # criterion = nn.BCEWithLogitsLoss()
            # loss = criterion(logp[:, 1], train_labels.float())

            logp = logp.cpu()

            f1 = f1_score(train_labels, torch.argmax(logp, 1).data.numpy(), average='macro')
            train_accuracy = accuracy_score(train_labels, torch.argmax(logp, 1).data.numpy())
            train_precision = metrics.precision_score(train_labels, torch.argmax(logp, 1).data.numpy(), average='macro')
            train_recall = metrics.recall_score(train_labels, torch.argmax(logp, 1).data.numpy(), average='macro')

            # eval
            model.eval()

            # cam = GradCAM(model)



            out, ST_dependecny,  a_grad_1, a_grad_2, h_node_tensor_1, h_node_tensor_2 = model(testGs, test_inputs, testGs_EC, test_inputs_EC, k)

            # activation_map = cam(class_idx=0, scores=out)
            #
            # print("activation_map",activation_map[0].squeeze(0).numpy())
            # print("activation_map shape",activation_map[0].squeeze(0).numpy().shape())

            print("test_labels", test_labels.size())
            print(test_labels)
            # print("ST_dependecny", ST_dependecny.size())
            # print(ST_dependecny)



            one_hot = np.zeros((1, out.size()[-1]), dtype=np.float32)
            one_hot[0][test_labels[0]] = 1
            one_hot = torch.from_numpy(one_hot).requires_grad_(True)
            # 获取目标类别的输出,该值带有梯度链接关系,可进行求导操作
            one_hot = torch.sum(one_hot.to(device) * out.to(device))
            # model.zero_grad()
            one_hot.backward(retain_graph=True)  # backward 求导
            # 获取对应特征层的梯度map
            # grads_val = model.extractor.get_gradients()[-1].cpu().data.numpy()

            """renji"""
            h_node_1 = torch.reshape(h_node_tensor_1, (100, -1, 16))
            h_node_2 = torch.reshape(h_node_tensor_2, (100, -1, 16))
            a_grad_tensor_1 = torch.reshape(a_grad_1, (100, -1, 16))
            a_grad_tensor_2 = torch.reshape(a_grad_2, (100, -1, 16))

            # h_node_1 = torch.reshape(h_node_tensor_1, (116, -1, 16))
            # h_node_2 = torch.reshape(h_node_tensor_2, (116, -1, 16))
            # a_grad_tensor_1 = torch.reshape(a_grad_1, (116, -1, 16))
            # a_grad_tensor_2 = torch.reshape(a_grad_2, (116, -1, 16))

            feature_map1 = torch.mean(h_node_1, dim=1)
            feature_map2 = torch.mean(h_node_2, dim=1)
            a_grad_weight_1 = torch.mean(a_grad_tensor_1, dim=1)
            a_grad_weight_2 = torch.mean(a_grad_tensor_2, dim=1)




            print("feature_map1", feature_map1.size())
            print("a_grad_weight", a_grad_weight_1.size())

            cam_1 = torch.mul(a_grad_weight_1,feature_map1)
            cam_2 = torch.mul(a_grad_weight_2,feature_map2)

            # print("cam size",cam_1.size())

            cam_1 = np.maximum(cam_1.cpu().detach().numpy(), 0)
            cam_2 = np.maximum(cam_2.cpu().detach().numpy(), 0)

            # print("cam 1", cam_1)
            # print("cam 2", cam_2)

            cam_1 = np.mean(cam_1, axis=1)
            cam_2 = np.mean(cam_2, axis=1)


            # print("cam 1", cam_1)
            # print("cam 2", cam_2)


            cam_1 = cam_1/np.sum(cam_1)
            cam_2 = cam_2 / np.sum(cam_2)


            # np.savetxt("./weight1.csv", cam_1, delimiter=",")
            # np.savetxt("./weight2.csv", cam_2, delimiter=",")


            np.savetxt("./weight1_renji.csv", cam_1, delimiter=",")
            np.savetxt("./weight2_renji.csv", cam_2, delimiter=",")

            #
            # print("grads_val",grads_val)
            # target = features[-1].cpu().data.numpy()[0, :]  # 获取目标特征输出
            # weights = np.mean(grads_val, axis=(2, 3))[0, :]  # 利用GAP操作, 获取特征权重
            # cam = weights.dot(target.reshape((nc, h * w)))
            # # relu操作,去除负值, 并缩放到原图尺寸
            # cam = np.maximum(cam, 0)
            # cam = cv2.resize(cam, input.shape[2:])
            # # 归一化操作
            # batch_cams = self._normalize(batch_cams)







            # print("ST_dependecny_final", ST_dependecny_final.size())
            # print(ST_dependecny_final)


            """renji"""
            for index in range(100):
                file_name = "./../dataset/Renji processed func data/ST_dependecny/ST_dependecny_node" + str(
                    index + 1) + ".csv"
                np.savetxt(file_name, ST_dependecny[index].cpu().detach().numpy(), fmt="%f", delimiter=",")


            # for index in range(116):
            #     file_name = "./../dataset/data114/ST_dependecny/ST_dependecny_node" + str(index+1) + ".csv"
            #     np.savetxt(file_name, ST_dependecny[index].cpu().detach().numpy(), fmt="%f", delimiter=",")





            # for i_index in range(len(test_labels)):
            #     file_name = "./../dataset/data114/ST_dependecny/ST_dependecny_"+ str(test_index[i_index]) + "_" + str(test_labels[i_index].cpu()) + ".csv"
            #
            #     print(file_name)
            #     print(ST_dependecny[test_labels[i_index]].cpu())
            #     # np.savetxt(file_name, ST_dependecny[test_labels[i_index]], fmt="%f",
            #     #            delimiter=",")



            # print("eval out", out)

            # test_logp = F.log_softmax(out.to(device), 1)
            test_logp = F.softmax(out.to(device), 1)

            test_logp_save = F.softmax(out.to(device), 1)




            # print("test logp", test_logp)

            # test_logp = F.softmax(out, 1)

            test_logp = test_logp.cpu()



            # test_loss = F.nll_loss(test_logp.to(device), test_labels.to(device))
            test_loss = F.cross_entropy(test_logp.to(device), test_labels.to(device))


            # test_regular_loss = 0
            # for param in model.parameters():
            #     test_regular_loss += torch.sum(abs(param))

            # test_loss = test_loss + args.alpha * test_regular_loss


            test_f1 = f1_score(test_labels, torch.argmax(test_logp, 1).data.numpy(), average='macro')
            test_accuracy = accuracy_score(test_labels, torch.argmax(test_logp, 1).data.numpy())
            test_precision = metrics.precision_score(test_labels, torch.argmax(test_logp, 1).data.numpy(), average='macro')
            test_recall = metrics.recall_score(test_labels, torch.argmax(test_logp, 1).data.numpy(), average='macro')

            test_confusion_matrix = metrics.confusion_matrix(test_labels, torch.argmax(test_logp, 1).data.numpy())
            # TP = test_confusion_matrix[1, 1]
            # TN = test_confusion_matrix[0, 0]
            # FP = test_confusion_matrix[0, 1]
            # FN = test_confusion_matrix[1, 0]
            #
            # test_specificity = TN / float(TN + FP)

            # test_AUC = metrics.roc_auc_score(test_labels, torch.argmax(test_logp, 1).data.numpy())

            # back propagation
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            # scheduler.step()

            # for p in model.parameters():
            #     print(p)

            if test_accuracy > temporal_acc:
                temporal_predict_label = list(torch.argmax(test_logp, 1).data.numpy())
                # temporal_predict_proba_label = test_logp.data.numpy()
                temporal_predict_proba_label = F.softmax(test_logp, 1).data.numpy()
                temporal_acc = test_accuracy
                early_stop = iterator

                temporal_predict_proba_label_save = test_logp_save.cpu().data.numpy()



            if iterator== args.epoch-1:
                print("epoch stop", early_stop)
                print("single fold acc", temporal_acc)
                print("single fold predict label list", temporal_predict_label)
                print("single fold test label list", list(test_labels.data.numpy()))
                total_predict_label.extend(temporal_predict_label)
                total_test_label.extend(list(test_labels.data.numpy()))
                test_acc_list.append(temporal_acc)


                total_predict_proba_label = np.concatenate((total_predict_proba_label, temporal_predict_proba_label), axis=0)
                print("single fold predict label probability list", total_predict_proba_label)

                total_predict_proba_label_save = np.concatenate((total_predict_proba_label_save, temporal_predict_proba_label_save), axis=0)
                print("single fold predict label probability_save list", total_predict_proba_label_save)


                # test_one_hot = label_binarize(test_labels.data.numpy(), np.arange(args.num_classes))
                # total_test_proba_label = np.concatenate((total_test_proba_label, test_one_hot), axis=0)
                # print("single fold test label probability list", total_test_proba_label)




            # if iterator== args.epoch-1:
            #     print("list(torch.argmax(test_logp, 1).data.numpy()) type", type(list(torch.argmax(test_logp, 1).data.numpy())))
            #     print("list(torch.argmax(test_logp, 1).data.numpy()) length",
            #           len(list(torch.argmax(test_logp, 1).data.numpy())))
            #     total_predict_label.extend(list(torch.argmax(test_logp, 1).data.numpy()))
            #     print("total_predict_label ", total_predict_label)
            #     print("test_labels", type(test_labels))
            #     total_test_label.extend(list(test_labels.data.numpy()))
            #     print("total_test_label", total_test_label)
            #     test_acc_list.append(float(test_accuracy))


            # train_loss_list.append(loss.item())
            # test_loss_list.append(test_loss.item())
            # # print(test_loss_list)
            #
            #




            # print("predict_label_prob")
            # print(test_logp)
            # print("softmax")
            # print(F.softmax(test_logp, 1).data.numpy())
            # print("predict_label")
            # print(torch.argmax(test_logp, 1).data.numpy())
            # print("test_label")
            # print(test_labels)






            # # print("log")
            # # print(logp)
            # # print("length", logp.size(0))
            # # for i in range(logp.size(0)):
            # #     if logp[i][0] <1:
            # #         continue
            # #     else:
            # #         print("nan", i)
            # #         print(train_index[i])
            #
            # # print('Epoch %d | Train Loss: %.4f | Train F1: %.4f | Train acc: %.4f | Train pre: %.4f | Train rec: %.4f '
            # #       '| Test F1: %.4f | Test acc: %.4f | Test pre: %.4f | Test rec: %.4f | Test specificity: %.4f | Test AUC: %.4f' % (
            # #           epoch, loss.item(), f1, train_accuracy,
            # #           train_precision, train_recall,
            # #           test_f1, test_accuracy, test_precision, test_recall, test_specificity, test_AUC))
            #
            # print("test confusion matrix")
            # print(test_confusion_matrix)

            print('Epoch %d | Train Loss: %.4f | Test Loss: %.4f| Train F1: %.4f | Train acc: %.4f | Train pre: %.4f | Train rec: %.4f '
                  '| Test F1: %.4f | Test acc: %.4f | Test pre: %.4f | Test rec: %.4f' % (
                      epoch, loss.item(), test_loss.item(), f1, train_accuracy,
                      train_precision, train_recall,
                      test_f1, test_accuracy, test_precision, test_recall))
        #
        # plt.plot(range(args.epoch), train_loss_list)
        # plt.xlabel('epoch')
        # plt.ylabel('loss')
        # plt.show()
        #
        # plt.plot(range(args.epoch), test_loss_list)
        # plt.xlabel('epoch')
        # plt.ylabel('loss')
        # plt.show()




    total_test_accuracy = accuracy_score(total_test_label, total_predict_label)
    total_test_precision = metrics.precision_score(total_test_label, total_predict_label, average='macro')
    total_test_recall = metrics.recall_score(total_test_label, total_predict_label, average='macro')
    total_test_f1 = f1_score(total_test_label, total_predict_label, average='macro')

    test_confusion_matrix = metrics.confusion_matrix(total_test_label, total_predict_label)
    TP = test_confusion_matrix[1, 1]
    TN = test_confusion_matrix[0, 0]
    FP = test_confusion_matrix[0, 1]
    FN = test_confusion_matrix[1, 0]

    test_specificity = TN / float(TN + FP)

    test_AUC = metrics.roc_auc_score(total_test_label, total_predict_proba_label[:,1].T)


    # test_AUC_prob = metrics.roc_auc_score(total_test_proba_label, total_predict_proba_label)

    fpr, tpr, thresholds = metrics.roc_curve(total_test_label, total_predict_proba_label[:,1].T)
    print("fpr", fpr)
    print("tpr", tpr)
    auc_prob = metrics.auc(fpr, tpr)
    print('auc_prob：', auc_prob)
    print('test_AUC：', test_AUC)





    print(
    'Test ACC : %.4f | Test Precision: %.4f | Test Recall(Sensitivity): %.4f | Test Specificity: %.4f  | F-score: %.4f '
    '| Test AUC: %.4f' % (
        total_test_accuracy, total_test_precision, total_test_recall, test_specificity, total_test_f1, test_AUC))

    print("confusion matrix")
    print(test_confusion_matrix)

    print("test acc list", test_acc_list)

    #
    # result_matrix = np.stack((np.array(total_test_label), np.array(total_predict_label)), axis=0)
    # # print("result", result_matrix.shape)
    # result_matrix = np.concatenate((result_matrix, np.array(total_predict_proba_label[:, 1].T).reshape(1,args.sample_num)), axis=0)
    # # print("result", result_matrix.shape)
    #
    #
    # result_matrix_save = np.concatenate((result_matrix, np.array(total_predict_proba_label_save[:, 1].T).reshape(1,args.sample_num)), axis=0)
    #
    #
    # fpr_tpr_matrix = np.concatenate((np.array(fpr), np.array(tpr)), axis=0)
    #
    #
    # np.savetxt("./../dataset/data114/fpr_tpr_GCN_trans_0228.csv", fpr_tpr_matrix, fmt="%f", delimiter=",")
    #
    # np.savetxt("./../dataset/data114/result_GCN_trans_0228.csv", result_matrix, fmt="%f", delimiter=",")
    #
    # np.savetxt("./../dataset/data114/result_GCN_trans_0228.csv", result_matrix_save, fmt="%f", delimiter=",")


    # # draw roc
    #
    # # FPR axis,TPR yxis
    # plt.plot(fpr, tpr, c='r', lw=2, alpha=0.7, label=u'AUC=%.3f' % test_AUC)
    # plt.plot((0, 1), (0, 1), c='grey', lw=1, ls='--', alpha=0.7)
    # plt.xlim((-0.01, 1.02))
    # plt.ylim((-0.01, 1.02))
    # plt.xticks(np.arange(0, 1.1, 0.1))
    # plt.yticks(np.arange(0, 1.1, 0.1))
    # plt.xlabel('False Positive Rate', fontsize=13)
    # plt.ylabel('True Positive Rate', fontsize=13)
    # plt.grid(b=True, ls=':')
    # plt.legend(loc='lower right', fancybox=True, framealpha=0.8, fontsize=12)
    # plt.title(u'ROC and AUC', fontsize=17)
    # plt.show()














#
# """single fold"""
#
# import numpy as np
# import pandas as pd
#
# import torch
# import torch.nn as nn
# import torch.nn.functional as F
# from torch.autograd import Variable
#
# import networkx as nx
# import dgl
# from tqdm import tqdm
# import pickle
# import argparse
#
# from sklearn.metrics import f1_score
# from sklearn.metrics import accuracy_score
# from sklearn import metrics
# import matplotlib.pyplot as plt
#
# from model import GCN_LSTM
# from model import GCN_Transformer
# import os
#
#
#
#
# if __name__ == '__main__':
#     parser = argparse.ArgumentParser()
#
#     os.environ['CUDA_VISIBLE_DEVICES'] = '0'
#     device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
#
#
#     """data114   114sample, 116node"""
#     """data705   582sample, 90node"""
#     label_df_path = "./../dataset/data114/window_37_step_10/label_df.pkl"
#     adj_graphs_path = "./../dataset/data114/window_37_step_10/FC_adj_graphs_df.pkl"
#     # label_df_path = "./../dataset/data705/window_37_step_10/label_df.pkl"
#     # adj_graphs_path = "./../dataset/data705/window_37_step_10/FC_adj_graphs_df.pkl"
#
#
#     parser.add_argument('--period', type=int, default=10)  #  time=10   segment_num = 11
#
#     parser.add_argument('--df_path', type=str, default=label_df_path)
#     parser.add_argument('--graphs_path', type=str, default=adj_graphs_path)
#
#     parser.add_argument('--gcn_in', type=int, default=116)  # node_num=116  90
#     # parser.add_argument('--gcn_in', type=int, default=90)  # node_num=116  90
#
#     parser.add_argument('--gcn_hid', type=int, default=32)  # default=32
#     parser.add_argument('--gcn_out', type=int, default=16)  # default=16
#     parser.add_argument('--gcn_layers', type=int, default=2)  # default=2
#     parser.add_argument('--lstm_hid', type=int, default=16)  # default=16
#     # parser.add_argument('--num_classes', type=int, default=2)  # default=2
#     parser.add_argument('--num_classes', type=int, default=2)  # default=3
#
#
#     parser.add_argument('--lstm_layers', type=int, default=2)  # default=2
#     parser.add_argument('--activation', type=str, default='relu')  # default=0
#     parser.add_argument('--model_drop', type=int, default=0.5)  # default=0\
#
#     parser.add_argument('--head_num', type=str, default='4')
#     parser.add_argument('--transform_layer', type=str, default='16')   #16
#     parser.add_argument('--temporal_drop', type=float, default=0.5)
#     parser.add_argument('--residual', type=bool, default=True)
#
#
#     parser.add_argument('--epoch', type=int, default=1000)  # can increase
#     parser.add_argument('--lr', type=float, default=0.0001)  # default=0.0001
#     parser.add_argument('--alpha', type=float, default=0.5)  # default=2
#
#     args = parser.parse_args()
#
#
#     # load data
#     df = pd.read_pickle(
#         args.df_path)  # [10 row * 4 column]  label, action1 list, action2 list, action3 list, action4 list
#     # if args.macro:
#     #     macro = pd.read_pickle(
#     #         args.macro_path).to_numpy()  # 10 * 4 array  macro_feat1, macro_feat2, macro_feat3, macro_feat4
#     graphs_sep = pickle.load(open(args.graphs_path, 'rb'))  # {id: list of networkx graphs}  10 * 7graph
#
#
#     window_size = 37
#     step = 10
#     window_path = "./../dataset/data114/window_" + str(window_size) + "_step_" + str(step)
#     # window_path = "./../dataset/data705/window_" + str(window_size) + "_step_" + str(step)
#
#     FC_dgls_path = window_path + "/FC_dgls.pkl"
#     EC_dgls_path = window_path + "/EC_dgls.pkl"
#     FC_inputs_path = window_path + "/FC_inputs.pkl"
#     EC_inputs_path = window_path + "/EC_inputs.pkl"
#     dgls = pd.read_pickle(FC_dgls_path)
#     inputs = pd.read_pickle(FC_inputs_path)
#
#     dgls_EC = pd.read_pickle(EC_dgls_path)
#     inputs_EC = pd.read_pickle(EC_inputs_path)
#
#
#
#
#
#
#     """data114 2class"""
#     zero_num = 0
#     one_num = 0
#     for line in range(len(df['label'].values)):
#         if (df['label'].values[line].astype(int) == 1 or df['label'].values[line].astype(int) == 2):
#             df['label'].values[line] = 0
#             zero_num += 1
#         else:
#             df['label'].values[line] = 1
#             one_num += 1
#
#     print("zero_label_num:", zero_num)
#     print("one_label_num:", one_num)
#
#
#
#     # # """data705 2class"""
#     # NC_index = []     #0
#     # EMCI_index = []   #1
#     # LMCI_index = []   #2
#     #
#     # EC_index_with_nan = [60, 227,290,317,329]     #61_5,61_6,228_9,228_10,291_7,291_8,318_1,318_2,318_3,318_4,330_1,330_2
#     #
#     # for line in range(len(df['label'].values)):
#     #     if (df['label'].values[line].astype(int) == 0) and line not in EC_index_with_nan:
#     #         NC_index.append(line)
#     #     elif (df['label'].values[line].astype(int) == 1) and line not in EC_index_with_nan:
#     #         EMCI_index.append(line)
#     #     elif (df['label'].values[line].astype(int) == 2) and line not in EC_index_with_nan:
#     #         LMCI_index.append(line)
#     #         df['label'].values[line] = 1
#     #     else:
#     #         continue
#     #
#     # print("NC_num:", len(NC_index))
#     # print("EMCI_num:", len(EMCI_index))
#     # print("LMCI_num:", len(LMCI_index))
#
#
#
#
#     # train, test split
#     n = len(dgls)
#     index = np.arange(n)
#     # index = NC_index + LMCI_index
#
#     n = len(index)
#     split = int(n * .9)
#
#     np.random.seed(16)
#     np.random.shuffle(index)
#
#
#     train_index, test_index = index[:split], index[split:]
#
#     print("total graph num:", n)
#     print("train_index:", train_index)
#     print("test_index:", test_index)
#
#
#     train_labels, test_labels = Variable(torch.LongTensor((df['label'].values.astype(int))[train_index])), Variable(
#         torch.LongTensor((df['label'].values.astype(int))[test_index]))
#
#
#     print("train_labels", train_labels)
#     print("test_label", test_labels)
#
#     # prep temporal graph data
#     k = args.period
#     trainGs, testGs = [dgls[i] for i in train_index], [dgls[i] for i in test_index]  # trainGs 8 * 7
#     print("trainGs", (len(trainGs), len(trainGs[0])))
#
#     trainGs, testGs = [dgl.batch([u[i] for u in trainGs]) for i in range(k)], \
#                       [dgl.batch([u[i] for u in testGs]) for i in range(k)]
#     train_inputs, test_inputs = [inputs[i] for i in train_index], [inputs[i] for i in test_index]
#     train_inputs, test_inputs = [torch.FloatTensor(np.concatenate([inp[i] for inp in train_inputs])) for i in range(k)], \
#                                 [torch.FloatTensor(np.concatenate([inp[i] for inp in test_inputs])) for i in range(k)]
#
#
#     trainGs_EC, testGs_EC = [dgls_EC[i] for i in train_index], [dgls_EC[i] for i in test_index]  # trainGs 8 * 7
#     print("trainGs_EC", (len(trainGs_EC), len(trainGs_EC[0])))
#
#     trainGs_EC, testGs_EC = [dgl.batch([u[i] for u in trainGs_EC]) for i in range(k)], \
#                             [dgl.batch([u[i] for u in testGs_EC]) for i in range(k)]
#     train_inputs_EC, test_inputs_EC = [inputs_EC[i] for i in train_index], [inputs_EC[i] for i in test_index]
#     train_inputs_EC, test_inputs_EC = [torch.FloatTensor(np.concatenate([inp[i] for inp in train_inputs_EC])) for i in
#                                        range(k)], \
#                                       [torch.FloatTensor(np.concatenate([inp[i] for inp in test_inputs_EC])) for i in
#                                        range(k)]
#
#     print("test_inputs", test_inputs[0])
#
#     if args.activation == 'relu':
#         activation = F.relu
#
#     # model = GCN_LSTM(args.gcn_in, args.gcn_hid, args.gcn_out, args.gcn_layers,
#     #                  args.lstm_hid, args.num_classes, args.lstm_layers, activation, args.model_drop)
#
#
#     model = GCN_Transformer(args.gcn_in, args.gcn_hid, args.gcn_out, args.gcn_layers, activation,
#                      args.period, args.head_num, args.transform_layer, args.temporal_drop, args.residual, args.num_classes, args.model_drop)
#
#
#
#     # if torch.cuda.device_count() > 1:
#     #     print("Use GPUs: ", torch.cuda.device_count())
#     #     model = nn.DataParallel(model)
#
#     model.to(device)
#
#     parameters = list(model.parameters())
#     optimizer = torch.optim.Adam(parameters, lr=args.lr, weight_decay=0.8)    #0.8
#     scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[50, 100], gamma=0.9)
#
#
#
#     train_loss_list = []
#     test_loss_list = []
#     for epoch in tqdm(range(args.epoch)):
#
#         # train
#         model.train()
#
#         k = args.period
#         out = model(trainGs, train_inputs, trainGs_EC, train_inputs_EC, k)
#
#         # print("out", out)
#
#
#         logp = F.log_softmax(out.to(device), dim=1)
#         # logp = F.softmax(out.to(device), dim=1)
#
#
#         # logp = F.softmax(out, dim=1)
#         print("logp size", np.shape(logp))
#         print("train_labels size", np.shape(train_labels))
#
#         regular_loss = 0
#         for param in model.parameters():
#             regular_loss += torch.sum(abs(param))
#
#         # print("logp", logp)
#         # print(train_labels)
#
#         loss = F.nll_loss(logp.to(device), train_labels.to(device))
#
#         # loss = loss + args.alpha * regular_loss
#
#
#         # print(logp)
#         # print(train_labels)
#         # criterion = nn.BCEWithLogitsLoss()
#         # loss = criterion(logp[:, 1], train_labels.float())
#
#         logp = logp.cpu()
#
#         f1 = f1_score(train_labels, torch.argmax(logp, 1).data.numpy(), average='macro')
#         train_accuracy = accuracy_score(train_labels, torch.argmax(logp, 1).data.numpy())
#         train_precision = metrics.precision_score(train_labels, torch.argmax(logp, 1).data.numpy(), average='macro')
#         train_recall = metrics.recall_score(train_labels, torch.argmax(logp, 1).data.numpy(), average='macro')
#
#         # eval
#         model.eval()
#
#         out = model(testGs, test_inputs, testGs_EC, test_inputs_EC, k)
#
#
#         # print("eval out", out)
#
#         test_logp = F.log_softmax(out.to(device), 1)
#         # test_logp = F.softmax(out.to(device), 1)
#
#
#         # print("test logp", test_logp)
#
#         # test_logp = F.softmax(out, 1)
#
#         test_logp = test_logp.cpu()
#
#
#
#         test_loss = F.nll_loss(test_logp.to(device), test_labels.to(device))
#
#
#         # test_regular_loss = 0
#         # for param in model.parameters():
#         #     test_regular_loss += torch.sum(abs(param))
#
#         # test_loss = test_loss + args.alpha * test_regular_loss
#
#
#         test_f1 = f1_score(test_labels, torch.argmax(test_logp, 1).data.numpy(), average='macro')
#         test_accuracy = accuracy_score(test_labels, torch.argmax(test_logp, 1).data.numpy())
#         test_precision = metrics.precision_score(test_labels, torch.argmax(test_logp, 1).data.numpy(), average='macro')
#         test_recall = metrics.recall_score(test_labels, torch.argmax(test_logp, 1).data.numpy(), average='macro')
#
#         test_confusion_matrix = metrics.confusion_matrix(test_labels, torch.argmax(test_logp, 1).data.numpy())
#         # TP = test_confusion_matrix[1, 1]
#         # TN = test_confusion_matrix[0, 0]
#         # FP = test_confusion_matrix[0, 1]
#         # FN = test_confusion_matrix[1, 0]
#         #
#         # test_specificity = TN / float(TN + FP)
#
#         # test_AUC = metrics.roc_auc_score(test_labels, torch.argmax(test_logp, 1).data.numpy())
#
#         # back propagation
#         optimizer.zero_grad()
#         loss.backward()
#         optimizer.step()
#         scheduler.step()
#
#         # for p in model.parameters():
#         #     print(p)
#
#         train_loss_list.append(loss.item())
#         test_loss_list.append(test_loss.item())
#         # print(test_loss_list)
#
#
#         print("predict_label_prob")
#         print(test_logp)
#         print("predict_label")
#         print(torch.argmax(test_logp, 1).data.numpy())
#         print("test_label")
#         print(test_labels)
#         # print("log")
#         # print(logp)
#         # print("length", logp.size(0))
#         # for i in range(logp.size(0)):
#         #     if logp[i][0] <1:
#         #         continue
#         #     else:
#         #         print("nan", i)
#         #         print(train_index[i])
#
#         # print('Epoch %d | Train Loss: %.4f | Train F1: %.4f | Train acc: %.4f | Train pre: %.4f | Train rec: %.4f '
#         #       '| Test F1: %.4f | Test acc: %.4f | Test pre: %.4f | Test rec: %.4f | Test specificity: %.4f | Test AUC: %.4f' % (
#         #           epoch, loss.item(), f1, train_accuracy,
#         #           train_precision, train_recall,
#         #           test_f1, test_accuracy, test_precision, test_recall, test_specificity, test_AUC))
#
#         print("test confusion matrix")
#         print(test_confusion_matrix)
#
#         print('Epoch %d | Train Loss: %.4f | Test Loss: %.4f| Train F1: %.4f | Train acc: %.4f | Train pre: %.4f | Train rec: %.4f '
#               '| Test F1: %.4f | Test acc: %.4f | Test pre: %.4f | Test rec: %.4f' % (
#                   epoch, loss.item(), test_loss.item(), f1, train_accuracy,
#                   train_precision, train_recall,
#                   test_f1, test_accuracy, test_precision, test_recall))
#
#     plt.plot(range(args.epoch), train_loss_list)
#     plt.xlabel('epoch')
#     plt.ylabel('loss')
#     plt.show()
#
#     plt.plot(range(args.epoch), test_loss_list)
#     plt.xlabel('epoch')
#     plt.ylabel('loss')
#     plt.show()

