# encoding:utf-8
import numpy as np
import csv
from scipy.io import loadmat
import xlrd
from collections import defaultdict

from scipy.stats import pearsonr
import tqdm
from tqdm import tqdm
import os
import pyinform
#label 1,2 seen as 1
#label 3,4 seen as 2
import pandas as pd
import pickle
import dgl
import networkx as nx
import torch
from PyIF import te_compute as te
from sklearn import preprocessing

# np.set_printoptions(suppress=True)
#
# m = loadmat("./../dataset/data114.mat")
#
# data = m.get('All_Bold')
#
# label = m.get('All_Group')





"""renji data """
def all_subjects_scans_label(filename_group, searchworkbook, NCI_dict_scan_subject_csv, aMCI_dict_scan_subject_csv, naMCI_dict_scan_subject_csv):

    workbook = xlrd.open_workbook(filename_group)
    sheet = workbook.sheet_by_index(0)

    print(sheet)

    row_count = sheet.nrows
    col_count = sheet.ncols

    print("row_count",row_count)
    print("col_count",col_count)


    """scan list"""
    NCI = [x for x in sheet.col_values(0, start_rowx=1, end_rowx=None) if x!= '']       #96
    aMCI = [x for x in sheet.col_values(1, start_rowx=1, end_rowx=None) if x!= '']     #64
    naMCI = [x for x in sheet.col_values(2, start_rowx=1, end_rowx=None) if x!= '']    #61

    all_scans = NCI + aMCI + naMCI

    scans_count = len(all_scans)           #221


    search_workbook = xlrd.open_workbook(searchworkbook)
    search_sheet = search_workbook.sheet_by_index(0)

    NCI_dict_scan_subject = defaultdict(list)
    aMCI_dict_scan_subject = defaultdict(list)
    naMCI_dict_scan_subject = defaultdict(list)

    all_dict_scan_subject = defaultdict(list)


    for i in range(1,609):
        key_subject_name = search_sheet.cell_value(i,2)
        value_scans = search_sheet.cell_value(i,0) + '.mat'

        # print(key_subject_name)
        # print(value_scans)

        if value_scans in NCI:
            NCI_dict_scan_subject[key_subject_name].append(value_scans)
        elif value_scans in aMCI:
            aMCI_dict_scan_subject[key_subject_name].append(value_scans)
        elif value_scans in naMCI:
            naMCI_dict_scan_subject[key_subject_name].append(value_scans)


        if value_scans in all_scans:
            all_dict_scan_subject[key_subject_name].append(value_scans)


    """subject num(differen sub in different class)"""
    print(len(NCI_dict_scan_subject))     #81
    print(len(aMCI_dict_scan_subject))    #59
    print(len(naMCI_dict_scan_subject))   #57


    NCI_dict_scan_subject_pd = pd.DataFrame(dict([(k, pd.Series(v)) for k, v in NCI_dict_scan_subject.items()]))
    aMCI_dict_scan_subject_pd = pd.DataFrame(dict([(k, pd.Series(v)) for k, v in aMCI_dict_scan_subject.items()]))
    naMCI_dict_scan_subject_pd = pd.DataFrame(dict([(k, pd.Series(v)) for k, v in naMCI_dict_scan_subject.items()]))

    NCI_dict_scan_subject_pd.to_csv(NCI_dict_scan_subject_csv)
    aMCI_dict_scan_subject_pd.to_csv(aMCI_dict_scan_subject_csv)
    naMCI_dict_scan_subject_pd.to_csv(naMCI_dict_scan_subject_csv)

    """read"""
    # pf = pd.read_csv(NCI_dict_scan_subject_csv, header=None)




"""renji data"""
def mat_to_csv_renji(mat_path, NCI_dict_scan_subject_csv, norm_bold_matrix_path):

    NCI_dict_scan_subject = pd.read_csv(NCI_dict_scan_subject_csv, header=None)
    NCI_subject_num = NCI_dict_scan_subject.shape[1] - 1
    print("subject_count", NCI_subject_num)

    scan_count = 0
    for j in range(1, len(NCI_dict_scan_subject.iloc[1, :])):
        for i in range(1, len(NCI_dict_scan_subject.iloc[:, j])):
            if not pd.isna(NCI_dict_scan_subject.iloc[i,j]):
                NCI_scan_name = NCI_dict_scan_subject.iloc[i,j]
                print(NCI_scan_name)
                scan_count += 1
                NCI_scan_path = mat_path + NCI_scan_name

                m = loadmat(NCI_scan_path)
                data = m.get('ROI_ts')

                data_matrix = data.T    #100*220
                np.savetxt(norm_bold_matrix_path + NCI_scan_name.replace("mat","csv"), data_matrix, fmt="%f", delimiter=",")

    print("scan_count", scan_count)


def mat_to_csv_renji_index(mat_path, NCI_dict_scan_subject_csv, aMCI_dict_scan_subject_csv, naMCI_dict_scan_subject_csv, norm_bold_matrix_path,
                           dict_index_sub_scan_path, two_label_path, three_label_path):
    NCI_dict_scan_subject = pd.read_csv(NCI_dict_scan_subject_csv, header=None)
    NCI_subject_num = NCI_dict_scan_subject.shape[1] - 1
    print("subject_count", NCI_subject_num)

    scan_index = 0
    sub_index = 0
    two_label_list = []
    three_label_list = []
    dict_index_sub_scan = defaultdict(list)


    for j in range(1, len(NCI_dict_scan_subject.iloc[1, :])):
        for i in range(1, len(NCI_dict_scan_subject.iloc[:, j])):
            if not pd.isna(NCI_dict_scan_subject.iloc[i,j]):

                dict_index_sub_scan[sub_index].append(scan_index)
                scan_index += 1

                two_label_list.append(0)
                three_label_list.append(0)

                NCI_scan_name = NCI_dict_scan_subject.iloc[i,j]
                print(NCI_scan_name)

                NCI_scan_path = mat_path + NCI_scan_name

                m = loadmat(NCI_scan_path)
                data = m.get('ROI_ts')

                data_matrix = data.T    #100*220
                np.savetxt(norm_bold_matrix_path + str(scan_index) + ".csv", data_matrix, fmt="%f", delimiter=",")
        sub_index += 1


    aMCI_dict_scan_subject = pd.read_csv(aMCI_dict_scan_subject_csv, header=None)
    for j in range(1, len(aMCI_dict_scan_subject.iloc[1, :])):
        for i in range(1, len(aMCI_dict_scan_subject.iloc[:, j])):
            if not pd.isna(aMCI_dict_scan_subject.iloc[i,j]):

                dict_index_sub_scan[sub_index].append(scan_index)
                scan_index += 1

                two_label_list.append(1)
                three_label_list.append(1)

                aNCI_scan_name = aMCI_dict_scan_subject.iloc[i,j]
                print(aNCI_scan_name)

                aNCI_scan_path = mat_path + aNCI_scan_name

                m = loadmat(aNCI_scan_path)
                data = m.get('ROI_ts')

                data_matrix = data.T    #100*220
                np.savetxt(norm_bold_matrix_path + str(scan_index) + ".csv", data_matrix, fmt="%f", delimiter=",")
        sub_index += 1


    naMCI_dict_scan_subject = pd.read_csv(naMCI_dict_scan_subject_csv, header=None)
    for j in range(1, len(naMCI_dict_scan_subject.iloc[1, :])):
        for i in range(1, len(naMCI_dict_scan_subject.iloc[:, j])):
            if not pd.isna(naMCI_dict_scan_subject.iloc[i,j]):

                dict_index_sub_scan[sub_index].append(scan_index)
                scan_index += 1

                two_label_list.append(1)
                three_label_list.append(2)

                naNCI_scan_name = naMCI_dict_scan_subject.iloc[i,j]
                print(naNCI_scan_name)

                naNCI_scan_path = mat_path + naNCI_scan_name

                m = loadmat(naNCI_scan_path)
                data = m.get('ROI_ts')

                data_matrix = data.T    #100*220
                np.savetxt(norm_bold_matrix_path + str(scan_index) + ".csv", data_matrix, fmt="%f", delimiter=",")
        sub_index += 1

    dict_index_sub_scan_pd = pd.DataFrame(dict([(k, pd.Series(v)) for k, v in dict_index_sub_scan.items()]))
    dict_index_sub_scan_pd.to_csv(dict_index_sub_scan_path)

    np.savetxt(two_label_path, two_label_list, fmt="%f", delimiter=",")
    np.savetxt(three_label_path, three_label_list, fmt="%f", delimiter=",")


"""data114"""
def mat_to_csv(mat_path, original_label_csv_path, norm_bold_matrix_path):

    m = loadmat(mat_path)
    data = m.get('All_Bold')
    label = m.get('All_Group')


    """original label csv"""
    for i in range(114):
        matrix = label
        np.savetxt(original_label_csv_path, matrix, fmt="%d", delimiter=",")

    """normalized bold signal matrix csv"""  # cut out last 137 dimension
    for i in range(114):
        data_matrix = data[i][0].T

        #np.savetxt(all_bold_path + str(i + 1) + ".csv", data_matrix, fmt="%f", delimiter=",")

        """last 137"""
        norm_bold_matrix = data_matrix[:,-137:]
        print("save norm_bold_matrix shape", np.shape(norm_bold_matrix))

        np.savetxt(norm_bold_matrix_path + str(i + 1) + ".csv", norm_bold_matrix, fmt="%f", delimiter=",")



"""data705"""
def mat_to_scv_705(csv_path, mat_path, original_label_csv_path, norm_bold_matrix_path):
    sample_num = 0
    EMCI_num = 0    #label 1
    LMCI_num = 0    #label 2
    NC = 0          #label 0

    # AD = 0          #label 3
    # MCI = 0         #label 4

    label_list = []

    label_file = pd.read_csv(csv_path, header=None, usecols=[2, 12])
    for i in range(1, len(label_file.values)):
        if (label_file.values[i][0] == 'EMCI'):
            EMCI_num += 1
            sample_num += 1
            label_list.append(1)
            mat_file_name = label_file.values[i][1]
            file_path = mat_path + "/" + str(mat_file_name) + "_result.mat"
            print(file_path)
            m = loadmat(file_path)
            data = m.get('features')
            """last 137"""
            norm_bold_matrix = data[:,-137:]
            print("save norm_bold_matrix shape", np.shape(norm_bold_matrix))
            np.savetxt(norm_bold_matrix_path + str(sample_num) + ".csv", norm_bold_matrix, fmt="%f", delimiter=",")
        elif (label_file.values[i][0] == 'LMCI'):
            LMCI_num += 1
            sample_num += 1
            label_list.append(2)
            mat_file_name = label_file.values[i][1]
            file_path = mat_path + "/" + str(mat_file_name) + "_result.mat"
            print(file_path)
            m = loadmat(file_path)
            data = m.get('features')
            """last 137"""
            norm_bold_matrix = data[:,-137:]
            print("save norm_bold_matrix shape", np.shape(norm_bold_matrix))
            np.savetxt(norm_bold_matrix_path + str(sample_num) + ".csv", norm_bold_matrix, fmt="%f", delimiter=",")
        elif (label_file.values[i][0] == 'CN'):
            NC += 1
            sample_num += 1
            label_list.append(0)
            mat_file_name = label_file.values[i][1]
            file_path = mat_path + "/" + str(mat_file_name) + "_result.mat"
            print(file_path)
            m = loadmat(file_path)
            data = m.get('features')
            """last 137"""
            norm_bold_matrix = data[:,-137:]
            print("save norm_bold_matrix shape", np.shape(norm_bold_matrix))
            np.savetxt(norm_bold_matrix_path + str(sample_num) + ".csv", norm_bold_matrix, fmt="%f", delimiter=",")
        else:
            continue

    print("NC, EMCI, LMCI", NC, EMCI_num, LMCI_num)
    print(len(label_list))
    label_matrix = np.array(label_list)
    np.savetxt(original_label_csv_path, label_matrix.T, fmt="%d", delimiter=",")





    # for file in os.listdir(mat_path):
    #     graph_num += 1
    #     file_path = mat_path + "/" + file
    #     print(file_path)
    #     m = loadmat(mat_path)
    #     data = m.get('features')
    #
    #     """last 137"""
    #     norm_bold_matrix = data[:,-137:]
    #     print("save norm_bold_matrix shape", np.shape(norm_bold_matrix))
    #
    #     np.savetxt(norm_bold_matrix_path + str(graph_num) + ".csv", norm_bold_matrix, fmt="%f", delimiter=",")





"""FC adj matrix & fea matrix"""
"""EC adj matrix"""
def Sliding_window_to_csv(window_size, step, norm_bold_matrix_path, window_path, sample_num):
    norm_bold_matrix = norm_bold_matrix_path
    for i in range(sample_num):
        norm_bold_matrix = np.loadtxt(norm_bold_matrix_path + str(i + 1) + ".csv", delimiter=",")
        print("read norm_bold_matrix shape", np.shape(norm_bold_matrix))

        for j in tqdm(range(int((len(norm_bold_matrix[0]) - window_size) / step) + 1)):

            """Fc fea_matrix"""
            fea_matrix = norm_bold_matrix[:, step * j:window_size + step * j]

            Fc_fea_matrix_path = window_path + "/FC/fea_matrix"

            if (os.path.exists(Fc_fea_matrix_path) == False):
                os.makedirs(Fc_fea_matrix_path)
            np.savetxt(Fc_fea_matrix_path + "/FC_fea_matrix_sample_" + str(i + 1) + "_segment_" + str(j + 1) + ".csv", fea_matrix, fmt="%f", delimiter=",")

            """Fc adj_matrix"""
            adj_matrix = np.zeros((len(fea_matrix), len(fea_matrix)), dtype=float)
            for p in range(len(fea_matrix)):
                for q in range(len(fea_matrix)):
                    pccs = np.corrcoef(fea_matrix[p, :], fea_matrix[q,
                                                         :])  # 2*2 matrix, principal diagonal=1, pccs = value in counter diagonal
                    # pccs = pearsonr(fea_matrix[p,:], fea_matrix[q,:])

                    adj_matrix[p][q] = np.abs(pccs[0][1])  # absolute

                    # if pccs[0][1]<0:
                    #     pccs == 0
                    # adj_matrix[p][q] = pccs[0][1]   #norm to zero



            Fc_adj_matrix_path = window_path + "/FC/adj_matrix"
            if (os.path.exists(Fc_adj_matrix_path) == False):
                os.makedirs(Fc_adj_matrix_path)
            np.savetxt(Fc_adj_matrix_path + "/FC_adj_matrix_sample_" + str(i + 1) + "_segment_" + str(j + 1) + ".csv", adj_matrix, fmt="%f", delimiter=",")


            """Ec adj_matrix"""
            if ((window_size + step * (j + 1)) <= len(norm_bold_matrix[0])):
                adj_matrix_Ec = np.zeros((len(fea_matrix), len(fea_matrix)), dtype=float)

                fea_matrix_t1 = norm_bold_matrix[:, (step * j):(window_size + step * j)]
                fea_matrix_t2 = norm_bold_matrix[:, (step * (j + 1)):(window_size + step * (j + 1))]

                # print(len(list(fea_matrix_t1[0, :])))
                # print(len(list(fea_matrix_t2[0, :])))


                """tansfer entropy  the feature value can be too large"""
                for pp in range(len(fea_matrix)):
                    for qq in range(len(fea_matrix)):
                        # t1_sum = fea_matrix_t1[pp, :].sum()
                        # t1_mean = t1_sum / len(fea_matrix_t1[pp, :])
                        # pp_max = fea_matrix_t1[pp, :].max()
                        # pp_min = fea_matrix_t1[pp, :].min()
                        # # pp_value = (fea_matrix_t1[pp, :]-pp_mean)/pp_std
                        # pp_value = (fea_matrix_t1[pp, :])
                        #
                        #
                        # t2_sum = fea_matrix_t2[qq, :].sum()
                        # t2_mean = t2_sum / len(fea_matrix_t2[qq, :])
                        # qq_max = fea_matrix_t2[qq, :].max()
                        # qq_min = fea_matrix_t2[qq, :].min()
                        #
                        # # qq_value = (fea_matrix_t2[qq, :]-qq_mean)/qq_std
                        # qq_value = (fea_matrix_t2[qq, :])



                        # mutual_info = pyinform.transferentropy.transfer_entropy(np.array(pp_value),
                        #                                                         np.array(qq_value), k=1)

                        TE = te.te_compute(np.array(fea_matrix_t1[pp, :]), np.array(fea_matrix_t2[qq, :]), k=1, embedding=1, safetyCheck=True, GPU=False)

                        if np.isnan(TE):
                            print("nan in EC_sample_%_%",(i+1,j+1))
                            TE = 0



                        adj_matrix_Ec[pp][qq] = np.abs(TE)

                for ppp in range(len(fea_matrix)):
                    for qqq in range(len(fea_matrix)):
                        if adj_matrix_Ec[ppp][qqq] < adj_matrix_Ec[qqq][ppp]:
                            adj_matrix_Ec[ppp][qqq] = 0


                Ec_adj_matrix_path = window_path + "/EC/adj_matrix"
                if (os.path.exists(Ec_adj_matrix_path) == False):
                    os.makedirs(Ec_adj_matrix_path)

                np.savetxt(Ec_adj_matrix_path + "/EC_adj_matrix_sample_" + str(i + 1) + "_segment_" + str(j + 1) + ".csv", adj_matrix_Ec, fmt="%f", delimiter=",")



"""original label matrix to original label pkl"""
def label_matrix_to_pkl(original_label_csv_path, label_df_path):
    matrix = pd.read_csv(original_label_csv_path, header=None)
    print("read label_matrix", matrix)
    matrix.columns = ['label']

    with open(label_df_path, "wb") as f:
        pickle.dump(matrix, f)

    df = pd.read_pickle(label_df_path)
    print(df.index)
    print(df['label'].values)





def adj_matrix_to_pkl(adj_matrix_path, adj_graphs_path, sample_num, matrix_type="EC"):
    sample_dic = {}
    for i in tqdm(range(sample_num)):
        networkx_list = []
        adj_matrix_path = adj_matrix_path
        if(matrix_type=="EC"):
            segment_num = 10                           #renji 17    data114,705   10
        else:
            segment_num = 11
        for j in range(segment_num):
            if(matrix_type == "FC"):
                adj_matrix = adj_matrix_path + "/FC_adj_matrix_sample_" + str(i + 1) + "_segment_" + str(
                    j + 1) + ".csv"
            else:
                adj_matrix = adj_matrix_path + "/EC_adj_matrix_sample_" + str(i + 1) + "_segment_" + str(
                    j + 1) + ".csv"
            pd_adj_matrix = pd.read_csv(adj_matrix, header=None)
            adj_matrix = pd_adj_matrix.values
            graph = nx.from_numpy_matrix(adj_matrix)
            networkx_list.append(graph)
        sample_dic[i] = networkx_list

    with open(adj_graphs_path, "wb") as f:
        pickle.dump(sample_dic, f)

    graphs_sep = pickle.load(open(adj_graphs_path, 'rb'))
    print(graphs_sep)
    print(graphs_sep[0])
    print(len(graphs_sep))



def pkl_to_dgl(label_df_path, adj_graphs_path, dgls_path, inputs_path):
    # load data
    df = pd.read_pickle(label_df_path)  # [10 row * 4 column]  label, action1 list, action2 list, action3 list, action4 list
    # if args.macro:
    #     macro = pd.read_pickle(
    #         args.macro_path).to_numpy()  # 10 * 4 array  macro_feat1, macro_feat2, macro_feat3, macro_feat4


    graphs_sep = pickle.load(open(adj_graphs_path, 'rb'))  # {id: list of networkx graphs}  10 * 7graph


    # create graph input
    dgls, inputs, xav, eye, dim = [], [], [], [], 20
    for gid in tqdm(df.index):
        g_list = graphs_sep[gid]  # g_list 1*7 graph list
        temp_g, temp_adj, temp_xav = [], [], []
        for g in g_list:
            G = nx_to_dgl(g)
            temp_g.append(G)
            temp_adj.append(np.array(nx.adj_matrix(g).todense()))
            # temp_xav.append(nn.init.xavier_uniform_(torch.zeros([12, dim])))
        dgls.append(temp_g)
        inputs.append(temp_adj)
        # xav.append(temp_xav)


    with open(dgls_path, "wb") as f:
        pickle.dump(dgls, f)

    with open(inputs_path, "wb") as f:
        pickle.dump(inputs, f)



def nx_to_dgl(g):
    G = dgl.DGLGraph()
    d = {n: i for i, n in enumerate(list(g.nodes()))}
    G.add_nodes(g.number_of_nodes())
    for e in list(g.edges()):
        G.add_edge(d[e[0]], d[e[1]], {'w': torch.FloatTensor([[g[e[0]][e[1]]['weight']]])})
    return G



if __name__ == '__main__':

    window_size = 37  #data114,data705
    # window_size = 50    #ren ji
    step = 10


    """data 114"""
    # sample_num = 114           # NC 1     EMCI 3     LMCI  4
    # mat_path = "./../dataset/data114.mat"
    # original_label_csv_path = "./../dataset/data114/4_label_matrix.csv"
    # # all_bold_path = "./../dataset/data114/all_bold/all_bold_sample_"
    # norm_bold_matrix_path = "./../dataset/data114/norm_bold_matrix/norm_bold_matrix_sample_"
    #
    # window_path = "./../dataset/data114/window_" + str(window_size) + "_step_" + str(step)
    #
    # label_df_path = "./../dataset/data114/window_" + str(window_size) + "_step_" + str(step) + "/label_df.pkl"
    #
    #
    #
    # FC_adj_matrix_path = window_path + "/FC/adj_matrix"
    # EC_adj_matrix_path = window_path + "/EC/adj_matrix"
    # FC_adj_graphs_path = window_path + "/FC_adj_graphs_df.pkl"
    # EC_adj_graphs_path = window_path + "/EC_adj_graphs_df.pkl"
    #
    #
    # FC_dgls_path = window_path + "/FC_dgls.pkl"
    # EC_dgls_path = window_path + "/EC_dgls.pkl"
    # FC_inputs_path = window_path + "/FC_inputs.pkl"
    # EC_inputs_path = window_path + "/EC_inputs.pkl"
    #
    # # mat_to_csv(mat_path, original_label_csv_path, norm_bold_matrix_path)
    # # print("mat_to_csv success!")
    #
    # # Sliding_window_to_csv(window_size, step, norm_bold_matrix_path, window_path, sample_num)
    # # print("Sliding_window_to_csv success!")
    #
    # label_matrix_to_pkl(original_label_csv_path, label_df_path)
    # print("label_matrix_to_pkl success!")
    #
    # adj_matrix_to_pkl(FC_adj_matrix_path, FC_adj_graphs_path, sample_num, matrix_type="FC")
    # print("FC_adj_matrix_to_pkl success!")
    #
    # adj_matrix_to_pkl(EC_adj_matrix_path, EC_adj_graphs_path, sample_num, matrix_type="EC")
    # print("EC_adj_matrix_to_pkl success!")
    #
    # pkl_to_dgl(label_df_path, FC_adj_graphs_path, FC_dgls_path, FC_inputs_path)
    # print("FC_pkl_to_dgl success!")
    #
    # pkl_to_dgl(label_df_path, EC_adj_graphs_path, EC_dgls_path, EC_inputs_path)
    # print("EC_pkl_to_dgl success!")


    """data705"""
    # sample_num = 582   # NC 193 label_0, EMCI 240 label_1, LMCI 149 label_2  total 508        AD 105       MCI  18
    # mat_path = "./../dataset/ADNI_705_origin/mat"
    # csv_path = "./../dataset/ADNI_705_origin/fMRI_Collection_8_23_2018_final.csv"
    # norm_bold_matrix_path = "./../dataset/data705/norm_bold_matrix/norm_bold_matrix_sample_"
    # original_label_csv_path = "./../dataset/data705/5_label_matrix.csv"
    #
    #
    # window_path = "./../dataset/data705/window_" + str(window_size) + "_step_" + str(step)
    #
    # label_df_path = "./../dataset/data705/window_" + str(window_size) + "_step_" + str(step) + "/label_df.pkl"
    #
    # FC_adj_matrix_path = window_path + "/FC/adj_matrix"
    # EC_adj_matrix_path = window_path + "/EC/adj_matrix"
    # FC_adj_graphs_path = window_path + "/FC_adj_graphs_df.pkl"
    # EC_adj_graphs_path = window_path + "/EC_adj_graphs_df.pkl"
    #
    #
    # FC_dgls_path = window_path + "/FC_dgls.pkl"
    # EC_dgls_path = window_path + "/EC_dgls.pkl"
    # FC_inputs_path = window_path + "/FC_inputs.pkl"
    # EC_inputs_path = window_path + "/EC_inputs.pkl"
    #
    #
    # # mat_to_scv_705(csv_path, mat_path, original_label_csv_path, norm_bold_matrix_path)
    # # print("mat_to_csv success!")
    #
    # # Sliding_window_to_csv(window_size, step, norm_bold_matrix_path, window_path, sample_num)
    # # print("Sliding_window_to_csv success!")
    #
    # # label_matrix_to_pkl(original_label_csv_path, label_df_path)
    # # print("label_matrix_to_pkl success!")
    #
    # # adj_matrix_to_pkl(FC_adj_matrix_path, FC_adj_graphs_path, sample_num, matrix_type="FC")
    # # print("FC_adj_matrix_to_pkl success!")
    #
    # # adj_matrix_to_pkl(EC_adj_matrix_path, EC_adj_graphs_path, sample_num, matrix_type="EC")
    # # print("EC_adj_matrix_to_pkl success!")
    # #
    # # pkl_to_dgl(label_df_path, FC_adj_graphs_path, FC_dgls_path, FC_inputs_path)
    # # print("FC_pkl_to_dgl success!")
    # #
    # # pkl_to_dgl(label_df_path, EC_adj_graphs_path, EC_dgls_path, EC_inputs_path)
    # # print("EC_pkl_to_dgl success!")



    """renji data"""
    # filename_group = "./../dataset/Renji processed func data/filename_group.xlsx"
    # searchworkbook = "./../dataset/Renji processed func data/ToProfessorShen.xlsx"
    #
    # NCI_dict_scan_subject_csv = "./../dataset/Renji processed func data/NCI_dict_scan_subject.csv"
    # aMCI_dict_scan_subject_csv = "./../dataset/Renji processed func data/aMCI_dict_scan_subject.csv"
    # naMCI_dict_scan_subject_csv = "./../dataset/Renji processed func data/naMCI_dict_scan_subject.csv"
    # # all_subjects_scans_label(filename_group, searchworkbook, NCI_dict_scan_subject_csv, aMCI_dict_scan_subject_csv, naMCI_dict_scan_subject_csv)
    #
    # mat_path = "./../dataset/Renji processed func data/bold_signal_100ROI/"
    # norm_bold_matrix_path = "./../dataset/Renji processed func data/norm_bold_matrix/norm_bold_matrix_sample_"
    # # # mat_to_csv_renji(mat_path, NCI_dict_scan_subject_csv, norm_bold_matrix_path)
    # # # mat_to_csv_renji(mat_path, aMCI_dict_scan_subject_csv, norm_bold_matrix_path)
    # # # mat_to_csv_renji(mat_path, naMCI_dict_scan_subject_csv, norm_bold_matrix_path)
    #
    # dict_index_sub_scan_path = "./../dataset/Renji processed func data/dict_index_sub_scan.csv"
    # two_label_path = "./../dataset/Renji processed func data/2_label_matrix.csv"
    # three_label_path = "./../dataset/Renji processed func data/3_label_matrix.csv"
    #
    # # mat_to_csv_renji_index(mat_path, NCI_dict_scan_subject_csv, aMCI_dict_scan_subject_csv, naMCI_dict_scan_subject_csv, norm_bold_matrix_path,
    # #                        dict_index_sub_scan_path, two_label_path, three_label_path)
    #
    #
    # window_path = "./../dataset/Renji processed func data/window_" + str(window_size) + "_step_" + str(step)
    # scan_num =221
    #
    # # Sliding_window_to_csv(window_size, step, norm_bold_matrix_path, window_path, scan_num)
    # # print("Sliding_window_to_csv success!")
    #
    # label_df_path = "./../dataset/Renji processed func data/window_" + str(window_size) + "_step_" + str(step) + "/label_df.pkl"
    #
    # # label_matrix_to_pkl(two_label_path, label_df_path)
    # # print("label_matrix_to_pkl success!")
    #
    #
    #
    #
    # FC_adj_matrix_path = window_path + "/FC/adj_matrix"
    # EC_adj_matrix_path = window_path + "/EC/adj_matrix"
    # FC_adj_graphs_path = window_path + "/FC_adj_graphs_df.pkl"
    # EC_adj_graphs_path = window_path + "/EC_adj_graphs_df.pkl"
    #
    # FC_dgls_path = window_path + "/FC_dgls.pkl"
    # EC_dgls_path = window_path + "/EC_dgls.pkl"
    # FC_inputs_path = window_path + "/FC_inputs.pkl"
    # EC_inputs_path = window_path + "/EC_inputs.pkl"
    #
    # # adj_matrix_to_pkl(FC_adj_matrix_path, FC_adj_graphs_path, scan_num, matrix_type="FC")
    # # print("FC_adj_matrix_to_pkl success!")
    #
    # # adj_matrix_to_pkl(EC_adj_matrix_path, EC_adj_graphs_path, scan_num, matrix_type="EC")
    # # print("EC_adj_matrix_to_pkl success!")
    #
    # # pkl_to_dgl(label_df_path, FC_adj_graphs_path, FC_dgls_path, FC_inputs_path)
    # # print("FC_pkl_to_dgl success!")
    #
    # pkl_to_dgl(label_df_path, EC_adj_graphs_path, EC_dgls_path, EC_inputs_path)
    # print("EC_pkl_to_dgl success!")



    """ADHD"""   #NC 105   ADHD 57    min 77

    sample_num = 162

    min_point = 77

    csv_path = "./../dataset/ADNI_705_origin/fMRI_Collection_8_23_2018_final.csv"
    norm_bold_matrix_path = "./../dataset/data705/norm_bold_matrix/norm_bold_matrix_sample_"
    original_label_csv_path = "./../dataset/data705/5_label_matrix.csv"


    window_path = "./../dataset/data705/window_" + str(window_size) + "_step_" + str(step)

    label_df_path = "./../dataset/data705/window_" + str(window_size) + "_step_" + str(step) + "/label_df.pkl"

    # FC_adj_matrix_path = window_path + "/FC/adj_matrix"
    # EC_adj_matrix_path = window_path + "/EC/adj_matrix"
    # FC_adj_graphs_path = window_path + "/FC_adj_graphs_df.pkl"
    # EC_adj_graphs_path = window_path + "/EC_adj_graphs_df.pkl"
    #
    #
    # FC_dgls_path = window_path + "/FC_dgls.pkl"
    # EC_dgls_path = window_path + "/EC_dgls.pkl"
    # FC_inputs_path = window_path + "/FC_inputs.pkl"
    # EC_inputs_path = window_path + "/EC_inputs.pkl"
    #
    #
    # # mat_to_scv_705(csv_path, mat_path, original_label_csv_path, norm_bold_matrix_path)
    # # print("mat_to_csv success!")
    #
    # # Sliding_window_to_csv(window_size, step, norm_bold_matrix_path, window_path, sample_num)
    # # print("Sliding_window_to_csv success!")
    #
    # # label_matrix_to_pkl(original_label_csv_path, label_df_path)
    # # print("label_matrix_to_pkl success!")
    #
    # # adj_matrix_to_pkl(FC_adj_matrix_path, FC_adj_graphs_path, sample_num, matrix_type="FC")
    # # print("FC_adj_matrix_to_pkl success!")
    #
    # # adj_matrix_to_pkl(EC_adj_matrix_path, EC_adj_graphs_path, sample_num, matrix_type="EC")
    # # print("EC_adj_matrix_to_pkl success!")
    # #
    # # pkl_to_dgl(label_df_path, FC_adj_graphs_path, FC_dgls_path, FC_inputs_path)
    # # print("FC_pkl_to_dgl success!")
    # #
    # # pkl_to_dgl(label_df_path, EC_adj_graphs_path, EC_dgls_path, EC_inputs_path)
    # # print("EC_pkl_to_dgl success!")





    """ABIDE"""   #NC 427   ABIDE 406    min 78