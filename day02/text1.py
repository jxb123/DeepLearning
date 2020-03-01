import torch
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import time
import sys
sys.path.append("..")
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


print(type(mnist_train))
print(len(mnist_train),len(mnist_test))
#%%
feature,label = mnist_train[0]
print(feature.shape,label)
#%%
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
start = time.time()
for X,y in train_iter:
    continue
print('%.2f sec'%(time.time() - start))