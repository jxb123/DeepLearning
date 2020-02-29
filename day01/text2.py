import torch
#%%
x = torch.ones(2,2,requires_grad=True)
print(x)
print(x.grad_fn)
#%%
y = x + 2
print(y)
print(y.grad_fn)
#%%
print(x.is_leaf,y.is_leaf)
#%%
z = y * y * 3
out = z.mean()
print(z,out)
#%%
a = torch.randn(2,2)
a = ((a * 3) / (a - 1))
print(a.requires_grad)
a.requires_grad_(True)
print(a.requires_grad)
b = (a * a).sum()
print(b.grad_fn)
#%%
out.backward()
print(x.grad)
#%%
out2 = x.sum()
out2.backward()
print(x.grad)
out3 = x.sum()
x.grad.data.zero_()
out3.backward()
print(x.grad)
