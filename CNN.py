import torch 
import torch.nn as nn
from torch.autograd import Variable
import torch.utils.data as data
import matplotlib.pyplot as plt
import torchvision
from datains import get_train_data
torch.manual_seed(1) #reproducible

#Hyper Parameters
EPOCH = 1
BATCH_SIZE = 50
LR = 0.001
train_data_x, train_data_y = get_train_data()

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

torch_train_data = GetLoader(train_data_x, train_data_y)

loader = data.DataLoader(
    dataset=torch_train_data,      # torch TensorDataset format
    batch_size=BATCH_SIZE,      # mini batch size
    shuffle=True,               # 要不要打乱数据 (打乱比较好)
    drop_last=False,
    num_workers=0,              # 多线程来读数据，不要用多线程，会变得不幸
)
for i,data__ in enumerate(loader):
    print(i,data__)

class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        self.conv1 = nn.Sequential( #input shape (1,28,28)
            nn.Conv2d(in_channels=1, #input height
                      out_channels=1, #n_filter
                     kernel_size=2, #filter size
                     stride=1, #filter step
                     ), #output shape (16,28,28)
            nn.ReLU(),

        )
        self.out = nn.Linear(32*7*7,10)

    def forward(self, x):
        x = self.conv1(x)
        x = x.view(x.size(0), -1) #flat (batch_size, 32*7*7)
        output = self.out(x)
        return output

cnn = CNN()
print(cnn)
#optimizer
optimizer = torch.optim.Adam(cnn.parameters(), lr=LR)

#loss_fun
loss_func = nn.CrossEntropyLoss()

#training loop
for epoch in range(EPOCH):
    for i, (x, y) in enumerate():
        batch_x = Variable(x)
        batch_y = Variable(y)
        #输入训练数据
        output = cnn(batch_x)
        #计算误差
        loss = loss_func(output, batch_y)
        #清空上一次梯度
        optimizer.zero_grad()
        #误差反向传递
        loss.backward()
        #优化器参数更新
        optimizer.step()
