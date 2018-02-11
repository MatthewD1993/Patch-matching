from network.utils import CompareLoss
import torch

from torch.autograd.variable import Variable
a = Variable(torch.FloatTensor([0.3, 0.6]), requires_grad=True)
b = Variable(torch.FloatTensor([0.05, 0.4]), requires_grad=True)

compare = CompareLoss()

l = compare(a, b)
print(l)
l.backward()
print(a.grad.data)
print(b.grad.data)

a = Variable(torch.FloatTensor([0.3, 0.6]), requires_grad=True)
b = Variable(torch.FloatTensor([0.05, 0.71]), requires_grad=True)

l = compare(a, b)
print(l)
l.backward()
print(a.grad.data)
print(b.grad.data)