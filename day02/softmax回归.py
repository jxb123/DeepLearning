import torch
import torchvision
import numpy as np
import sys
sys.path.append('..')
import d2lzh_pytorch as d2l
#改用新的方法，加载数据集模块
def load_mnist(path, kind='train'):
    """ load自己手动下载的数据集 """
    import os
    import gzip
    import numpy as np

    """Load MNIST data from `path`"""
    labels_path = os.path.join(path,
                               '%s-labels-idx1-ubyte.gz'
                               % kind)
    images_path = os.path.join(path,
                               '%s-images-idx3-ubyte.gz'
                               % kind)

    with gzip.open(labels_path, 'rb') as lbpath:
        labels = np.frombuffer(lbpath.read(), dtype=np.uint8,
                               offset=8)

    with gzip.open(images_path, 'rb') as imgpath:
        images = np.frombuffer(imgpath.read(), dtype=np.uint8,
                               offset=16).reshape(len(labels), 784)

    return images, labels
#读取数据集
X_train, y_train = load_mnist('E:\DLData\FashionMNIST', kind='train')
X_test, y_test = load_mnist('E:\DLData\FashionMNIST', kind='t10k')
X_train_tensor = torch.from_numpy(X_train).to(torch.float32).view(-1, 1, 28, 28) * (1 / 255.0)
X_test_tensor = torch.from_numpy(X_test).to(torch.float32).view(-1, 1, 28, 28) * (1 / 255.0)
y_train_tensor = torch.from_numpy(y_train).to(torch.int64).view(-1, 1)
y_test_tensor = torch.from_numpy(y_test).to(torch.int64).view(-1, 1)

import torch.utils.data as Data
mnist_train = Data.TensorDataset(X_train_tensor, y_train_tensor)
mnist_test = Data.TensorDataset(X_test_tensor, y_test_tensor)
feature,label = mnist_train[0]
X,y = [],[]
for i in range(10):
    X.append(mnist_train[i][0])
    y.append(mnist_test[i][1])
d2l.show_fashion_mnist(X,d2l.get_fashion_mnist_labels(y))
#%%
batch_size = 256
if sys.platform.startswith('win'):
    num_workers = 0  #表示不用额外的进程来加速读取数据
else:
    num_workers = 4
train_iter = torch.utils.data.DataLoader(mnist_train,batch_size = batch_size,shuffle = True,num_workers=num_workers)
test_iter = torch.utils.data.DataLoader(mnist_test,batch_size = batch_size,shuffle = False,num_workers=num_workers)
#%%
num_inputs = 784
num_outputs = 10

W = torch.tensor(np.random.normal(0,0.01,(num_inputs,num_outputs)),dtype=torch.float)
b = torch.zeros(num_outputs,dtype=torch.float)
W.requires_grad_(requires_grad = True)
b.requires_grad_(requires_grad = True)
#%%# 小例子测试：如何对多维Tensor按维度操作
X = torch.tensor([[1,2,3],[4,5,6]])
print(X.sum(dim=0,keepdim=True))
print(X.sum(dim=1,keepdim=True))
#%%
def softmax(X):
    X_exp = X.exp()
    partition = X_exp.sum(dim=1,keepdim=True)
    return X_exp/partition
X = torch.rand((2,5))
X_prob = softmax(X)
print(X_prob,X_prob.sum(dim=1))
#%%定义模型
def net(X):
    return softmax(torch.mm(X.view((-1,num_inputs)),W)+b)
#%%定义损失函数
y_hat = torch.tensor([[0.1,0.3,0.6],[0.3,0.2,0.5]])
y = torch.LongTensor([0,2])
y_hat.gather(1,y.view(-1,1))
def cross_entropy(y_hat,y):
    return - torch.log(y_hat.gather(1,y.view(-1,1)))
#%%计算分类准确率
def accuracy(y_hat,y):
    return (y_hat.argmax(dim=1) == y).float().mean().item()
print(accuracy(y_hat,y))
print(d2l.evaluate_accuracy(test_iter,net))
#%%训练模型
num_epochs,lr = 5,0.1
d2l.train_ch3(net,train_iter,test_iter,cross_entropy,num_epochs,batch_size,[W,b],lr)
#%%预测
X,y = iter(test_iter).next()
true_labels = d2l.get_fashion_mnist_labels(y.numpy())
pred_labels = d2l.get_fashion_mnist_labels(net(X).argmax(dim=1).numpy())
titles = [true + '\n' + pred for true,pred in zip(true_labels,pred_labels)]
d2l.show_fashion_mnist(X[0:9],titles[0:9])
