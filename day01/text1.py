import torch
#%%
print(torch.__version__)
#%%
x = torch.empty(5,3)
print(x)
#%%
x = torch.rand(5,3)
print(x)