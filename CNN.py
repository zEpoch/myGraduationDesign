import torch 
import torch.nn as nn
from torch.autograd import Variable
import matplotlib.pyplot as plt
import torchvision
from datains import get_train_data
from datains import get_test_data
torch.manual_seed(1) #reproducible

#Hyper Parameters
EPOCH = 10
BATCH_SIZE = 1
LR = 0.001

class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        self.conv1 = nn.Sequential( #input shape (1,27,2)
            nn.Conv1d(in_channels=27, #input height
                      out_channels=27, #n_filter
                     kernel_size=2, #filter size
                     stride=1, #filter step
                     ), #output shape (1,27,1)
            nn.ReLU(),

        )
        self.out = nn.Linear(27,2)

    def forward(self, x):
        x = self.conv1(x.to(torch.float32))
        x = torch.flatten(x,1)
        output = self.out(x)
        return output

cnn = CNN()
#optimizer
optimizer = torch.optim.Adam(cnn.parameters(), lr=LR)

#loss_fun
loss_func = nn.CrossEntropyLoss()


train_loader = get_train_data(BATCH_SIZE)
#training loop
for epoch in range(EPOCH):
    for i, (x, y) in enumerate(train_loader):
        batch_x = Variable(x)
        batch_y = Variable(y)
        #输入训练数据
        output = cnn(batch_x)
        #计算误差
        loss = loss_func(output, batch_y)
        #清空上一次梯度
        if((i+1)%1000==0):
            print('loss',loss)
        optimizer.zero_grad()
        #误差反向传递
        loss.backward()
        #优化器参数更新
        optimizer.step()


y_score = []
test_loader = get_test_data(BATCH_SIZE)
from sklearn.metrics import roc_auc_score
prob_all = []
label_all = []
for i, (x,y) in enumerate(train_loader):
    prob = cnn(x) #表示模型的预测输出
    print([abs(abs(max(prob[0]))-1).detach().numpy()],y)
    prob_all.extend([abs(abs(max(prob[0]))-1).detach().numpy()]) #prob[:,1]返回每一行第二列的数，根据该函数的参数可知，y_score表示的较大标签类的分数，因此就是最大索引对应的那个值，而不是最大索引值
    label_all.extend(y)
print(len(prob_all),len(label_all))
print("AUC:{:.4f}".format(roc_auc_score(label_all,prob_all)))
