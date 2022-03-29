from typing import final
import word2vec
import dataload
import sklearn
from sklearn.model_selection import train_test_split
import numpy as np
ans = word2vec.get_vec()

pos_train_data = dataload.get_pos_train_data()
neg_train_data = dataload.get_neg_train_data()
pos_test_data = dataload.get_pos_test_data()
neg_test_data = dataload.get_neg_test_data()


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
    for i in get_data_vec_ins(pos_train_data, 1)+get_data_vec_ins(neg_train_data, 0):
        train_data += i
    print(train_data[0])
    print("************************")
    train_data = np.array(train_data,dtype=object)
    train_data_x, train_data_y = train_data[:,-1], train_data[:,0:-1]
    train_data_x, train_data_y, test_data_x, test_data_y = train_test_split(train_data_x, train_data_y, test_size=0.3)
    return train_data_x, train_data_y, test_data_x, test_data_y

def get_test_data():
    train_data = []
    for i in get_data_vec_ins(pos_test_data, 1)+get_data_vec_ins(neg_test_data, 0):
        train_data += i
    print(train_data[0])
    print("************************")
    test_data = np.array(test_data,dtype=object)
    test_data_x, test_data_y = train_data[:,-1], train_data[:,0:-1]
    return test_data_x, test_data_y



