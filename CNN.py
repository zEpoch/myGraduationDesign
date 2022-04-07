import torch 
import torch.nn as nn
from torch.autograd import Variable
import matplotlib.pyplot as plt
# from ProtBert import get_train_data
from datains import get_train_data
# from datains import get_test_data
from sklearn.metrics import roc_auc_score
from sklearn.metrics import roc_curve
from sklearn.metrics import auc
from sklearn.metrics import accuracy_score
from sklearn.metrics import recall_score
import numpy as np
torch.manual_seed(1) #reproducible
def sigmoid(x):
    return 1.0/(1+np.exp(-x-10))
#Hyper Parameters
EPOCH = 5
BATCH_SIZE = 80
LR = 0.002

class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        self.conv1 = nn.Sequential( #input shape (n,27,2)
            nn.Conv1d(in_channels=27, #input height
                      out_channels=27, #n_filter
                      kernel_size=2, #filter size
                      stride=1, #filter step
                    ), #output shape (n,27,1)
            # nn.AvgPool1d(2,stride=1),
            # nn.MaxPool1d(1,stride=1),
            nn.ReLU(),

        )
        self.layer1 = nn.Linear(27,10)
        self.layer2 = nn.Linear(10,5)
        self.out = nn.Linear(5,2)
    def forward(self, x):
        x = self.conv1(x.to(torch.float32))
        x = torch.flatten(x,1)
        x = self.layer1(x)
        x = self.layer2(x)
        output = self.out(x)
        return output

cnn = CNN()
#optimizer
optimizer = torch.optim.Adam(cnn.parameters(), lr=LR)
# optimizer = torch.optim.SGD(cnn.parameters(), lr=0.01)
#loss_fun
loss_func = nn.CrossEntropyLoss()


train_loader,test_loader = get_train_data(BATCH_SIZE)

#training loop

Loss = []
for epoch in range(EPOCH):
    for i, (x, y) in enumerate(train_loader):
        batch_x = Variable(x).to(torch.float32)
        batch_y = Variable(y)
        #输入训练数据
        output = cnn(batch_x)
        #计算误差
        loss = loss_func(output, batch_y.long())
        print(loss)
        #清空上一次梯度
        Loss.append(loss.detach().numpy())
        if((i+1)%1000==0):
            print('loss',loss)
        optimizer.zero_grad()
        #误差反向传递
        loss.backward()
        #优化器参数更新
        optimizer.step()
        
x = range(len(Loss))
y = Loss
plt.plot(x, y, label='train loss', linewidth=2, color='r', marker='o', markerfacecolor='r', markersize=5)
plt.xlabel('Epoch')
plt.ylabel('Loss Value')
plt.legend()
plt.show()
train_prob_all = []
train_label_all = []
for i, (x, y) in enumerate(train_loader):  
    output = cnn(x)
    train_prob_all.extend(output[:,1].detach().numpy())
    train_label_all.extend(y)

# test_loader = get_test_data(BATCH_SIZE)

prob_all = []
label_all = []
for i, (x,y) in enumerate(test_loader):
    prob = cnn(x) #表示模型的预测输出
    prob_all.extend(prob[:,1].detach().numpy()) #prob[:,1]返回每一行第二列的数，根据该函数的参数可知，y_score表示的较大标签类的分数，因此就是最大索引对应的那个值，而不是最大索引值
    label_all.extend(y)
    
prob_all_auc = []
train_prob_auc = []
for i in prob_all:
    prob_all_auc.append(sigmoid(i))
for i in train_prob_all:
    train_prob_auc.append(sigmoid(i))

train_fpr, train_tpr, _ = roc_curve(train_label_all,train_prob_auc,pos_label=1)
fpr, tpr, _ = roc_curve(label_all,prob_all_auc,pos_label=1)
train_roc_auc = auc(train_fpr, train_tpr)
roc_auc = auc(fpr, tpr)

prob_all_new = np.around(prob_all,0).astype(int)
prob_all = []
for i in prob_all_new:
    if i>0:
        prob_all.append(1)
    else:
        prob_all.append(0)

print("accuracy：",accuracy_score(label_all,prob_all))
print("recall：",recall_score(label_all,prob_all, average='micro'))
plt.figure()
lw = 2
plt.plot(
    fpr,
    tpr,
    color="darkorange",
    lw=lw,
    label="test_ROC curve (area = %0.2f)" % roc_auc,
)
plt.plot(
    train_fpr,
    train_tpr,
    color="pink",
    lw=lw,
    label="train_ROC curve (area = %0.2f)" % train_roc_auc,
)
plt.plot([0, 1], [0, 1], color="navy", lw=lw, linestyle="--")
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("Receiver operating characteristic example")
plt.legend(loc="lower right")
plt.show()