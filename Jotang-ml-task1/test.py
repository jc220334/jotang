import torch
x=torch.arange(10)
print(x)
print(x.shape) #张量的形状
print(x.numel()) #张量的总数
x=x.reshape(2,-1) #张量的形状改变,-1表示自动计算
print(x)
y=torch.zeros((2,3,4))#创建一个全为0的张量
z=torch.ones((2,3,4))#创建一个全为1的张量
m=torch.rand((2,3,4))#创建一个随机张量
print(y)
print(z)
print(m)
print(torch.tensor([2, 1, 4, 3]))
#创建给定值的张量
h=torch.tensor([2, 1, 4, 3])
t=torch.tensor([1, 2, 3, 4])
print(h+t)
print(h-t)
print(h*t)
print(h/t)
print(h**t)#求幂运算
print(torch.exp(h))
X = torch.arange(12, dtype=torch.float32).reshape((3,4))
Y = torch.tensor([[2.0, 1, 4, 3], [1, 2, 3, 4], [4, 3, 2, 1]])
print(torch.cat((X, Y), dim=0))
print(torch.cat((X, Y), dim=1))
print(X==Y) 
print(X.sum()) #对所有张量元素求和，得到单张量
a = torch.arange(3).reshape((3, 1))
b = torch.arange(2).reshape((1, 2))
print(a+b) #广播机制/自动复制行列
