import torch
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import time
import sys
sys.path.append("..")
import d2lzh_pytorch as d2l
mnist_train = torchvision.datasets.FashionMNIST(root='E:/DLData',train=True,download=False,transform=transforms.ToTensor())
mnist_test = torchvision.datasets.FashionMNIST(root='E:/DLData',train=False,download=False,transform=transforms.ToTensor())
