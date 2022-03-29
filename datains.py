from typing import final
import word2vec
import dataload
import sklearn
from sklearn.model_selection import train_test_split
import numpy as np
# ans = word2vec.get_vec()
ans = {
    'L': [0.40705007314682007, 0.7434920072555542],
    'F': [0.47620540857315063, 0.49013853073120117],
    'Y': [0.4271319508552551, 0.3426735997200012],
    'A': [-0.23547989130020142, 1.0666087865829468],
    'I': [0.277849942445755, 0.7182607650756836],
    'X': [2.051089286804199, -0.6742867231369019],
    'K': [0.4145559072494507, 0.7840175628662109],
    'D': [0.19488804042339325, 0.8006709218025208],
    'Q': [-0.31024861335754395, 0.930740475654602],
    'V': [0.43926331400871277, 0.7616387009620667],
    'W': [0.02974635548889637, 0.16806700825691223],
    'P': [-0.26110294461250305, 0.998232901096344],
    'N': [-0.2566228210926056, 0.8910999894142151],
    'C': [0.015114320442080498, 0.7300181984901428],
    'G': [0.061138175427913666, 0.7566549181938171],
    'T': [-0.6935868859291077, 1.035444974899292],
    'H': [-0.6106293797492981, 1.0271210670471191],
    'R': [0.425143301486969, 0.7110256552696228],
    'E': [0.4655556082725525, 0.8371189832687378],
    'S': [-0.4302646815776825, 1.0112868547439575],
    'M': [0.2677305340766907, 0.8106550574302673]
}

pos_train_data = dataload.get_pos_train_data()
neg_train_data = dataload.get_neg_train_data()
pos_test_data = dataload.get_pos_test_data()
neg_train_data = dataload.get_neg_test_data()


def get_data_vec_ins(data, label):
    temp = [[] for _ in range(len(data))]
    for i in range(len(data)):
        con_temp = []
        for j in data[i].split(' '):
            con_temp.append(ans[str(j)])
        temp[i].append([con_temp,label])
        
    return temp


def get_train_data():
    train_data = []
    for i in get_data_vec_ins(pos_train_data, 1):
        train_data += i
    print(train_data[0])
    print("************************")
    train_data = np.array(train_data,dtype=object)
    train_data_x, train_data_y = train_data[:,-1], train_data[:,0:-1]
    train_data_x, train_data_y, test_data_x, test_data_y = train_test_split(train_data_x, train_data_y, test_size=0.3)
    return train_data_x, train_data_y, test_data_x, test_data_y



