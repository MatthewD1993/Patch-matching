import torch
from network.utils import new_hinge_loss

def to_np(x):
    return x.data.cpu().numpy()


a_m = torch.autograd.Variable(torch.FloatTensor([0.1]), requires_grad=True)


for i in range(100):
    a = torch.Tensor([[0.2], [0.3]])

    a_v = torch.autograd.Variable(a, requires_grad=False)

    b = torch.FloatTensor([1, -1])
    b_v = torch.autograd.Variable(b, requires_grad=False)

    c = torch.matmul(a_v, a_m)
    l_v = new_hinge_loss(c, b_v)

    l_v.backward(retain_graph=True)
    a_m.data -= a_m.grad.data*0.1
    print('Iteration ', i)
    print('loss', to_np(l_v))
    print('a_m', to_np(a_m))
    print('gradient', to_np(a_m.grad))
    a_m.grad.zero_()
