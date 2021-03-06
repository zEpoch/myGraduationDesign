from typing import final
import word2vec
import dataload
import sklearn
from sklearn.model_selection import train_test_split
import numpy as np
import torch
import torch.utils.data as Data
from sklearn.model_selection import train_test_split
from imblearn.over_sampling import SMOTE
ans = word2vec.get_vec()

pos_train_data = dataload.get_pos_train_data()
neg_train_data = dataload.get_neg_train_data()
pos_test_data = dataload.get_pos_test_data()
neg_test_data = dataload.get_neg_test_data()

# 定义GetLoader类，继承Dataset方法，并重写__getitem__()和__len__()方法
class GetLoader(torch.utils.data.Dataset):
	# 初始化函数，得到数据
    def __init__(self, data_root, data_label):
        self.data = data_root
        self.label = data_label
    # index是根据batchsize划分数据后得到的索引，最后将data和对应的labels进行一起返回
    def __getitem__(self, index):
        data = self.data[index]
        labels = self.label[index]
        return data, labels
    # 该函数返回数据大小长度，目的是DataLoader方便划分，如果不知道大小，DataLoader会一脸懵逼
    def __len__(self):
        return len(self.data)

def get_data_vec_ins(data, label):
    temp = [[] for _ in range(len(data))]
    for i in range(len(data)):
        con_temp = []
        for j in data[i].split(' '):
            for k in ans[str(j)]:
                con_temp.append(k)
                temp[i].append(k)
        temp[i].append(label)
        
    return temp

def plus_dim(data):
    new_data = []
    for i in data:
        result = i.reshape((27, 2))
        new_data.append(result)
    return np.array(new_data)

def get_train_data(BATCH_SIZE):
    train_data = []
    for i in get_data_vec_ins(pos_train_data, 1)+get_data_vec_ins(neg_train_data, 0):
        train_data.append(i)
    train_data,test_data = train_test_split(train_data,test_size=0.4)
    train_data,test_data = np.array(train_data),np.array(test_data)
    sm = SMOTE(random_state=42) #实例化
    
    train_data_x, train_data_y = train_data[:,0:-1],train_data[:,-1]
    test_data_x, test_data_y = test_data[:,0:-1],test_data[:,-1]
    train_data_x, train_data_y = sm.fit_resample(train_data_x, train_data_y)
    test_data_x, test_data_y = sm.fit_resample(test_data_x, test_data_y)
    train_data_x,test_data_x = plus_dim(train_data_x),plus_dim(test_data_x)
    
    # print(train_data_x.shape,train_data_y.shape)
    torch_train_data = GetLoader(np.array(train_data_x), np.array(train_data_y))
    torch_test_data = GetLoader(np.array(test_data_x), np.array(test_data_y))
    train_data_loader = Data.DataLoader(
        dataset=torch_train_data,      # torch TensorDataset format
        batch_size=BATCH_SIZE,      # mini batch size
        shuffle=True,               # 要不要打乱数据 (打乱比较好)
        drop_last=True,
        num_workers=0,              # 多线程来读数据，不要用多线程，会变得不幸
    )
    test_data_loader = Data.DataLoader(
        dataset=torch_test_data,      # torch TensorDataset format
        batch_size=BATCH_SIZE,      # mini batch size
        shuffle=True,               # 要不要打乱数据 (打乱比较好)
        drop_last=True,
        num_workers=0,              # 多线程来读数据，不要用多线程，会变得不幸
    )
    return train_data_loader,test_data_loader

def get_test_data(BATCH_SIZE):
    test_data = []
    for i in get_data_vec_ins(pos_test_data, 1)+get_data_vec_ins(neg_test_data, 0):
        test_data += i
    test_data = np.array(test_data,dtype=object)
    test_data_x, test_data_y = test_data[:,0:-1], test_data[:,-1]
    torch_test_data = GetLoader(np.array([i[0] for i in test_data_x]), np.array(test_data_y))
    test_loader = Data.DataLoader(
        dataset=torch_test_data,      # torch TensorDataset format
        batch_size=BATCH_SIZE,      # mini batch size
        shuffle=True,               # 要不要打乱数据 (打乱比较好)
        drop_last=True,
        num_workers=0,              # 多线程来读数据，不要用多线程，会变得不幸
    )
    return test_loader



