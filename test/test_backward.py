from network.utils import new_hinge_loss, compare_loss, accuracy
import torch
from torch.autograd import Variable, gradcheck


def to_np(x):
    return x.data.cpu().numpy()

# Testing my hinge loss, test backward.
# a_m = torch.autograd.Variable(torch.FloatTensor([0.1]), requires_grad=True)
#
# for i in range(100):
#     a = torch.Tensor([[0.2], [0.3]])
#
#     a_v = torch.autograd.Variable(a, requires_grad=False)
#
#     b = torch.FloatTensor([1, -1])
#     b_v = torch.autograd.Variable(b, requires_grad=False)
#
#     c = torch.matmul(a_v, a_m)
#     l_v = new_hinge_loss(c, b_v)
#
#     l_v.backward(retain_graph=True)
#     a_m.data -= a_m.grad.data*0.1
#     print('Iteration ', i)
#     print('loss', to_np(l_v))
#     print('a_m', to_np(a_m))
#     print('gradient', to_np(a_m.grad))
#     a_m.grad.zero_()


# # Test "compare loss" grad==========================================

p = Variable(torch.FloatTensor([0.5, 0.8]), requires_grad=True)
n = Variable(torch.FloatTensor([0.3, 0.7]), requires_grad=True)

test = gradcheck(compare_loss, (p, n), eps=1e-4, atol=1e-4)
print(test)

loss = compare_loss(p,n)
loss.backward()
print('p gradient: ', p.grad.data)
print('n gradient: ', n.grad.data)