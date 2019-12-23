from mymodules import *

r0 = RandomMaskedConv(2,3,4)

x = torch.rand([4,2,4,4]).requires_grad_()

y = r0(x)

z = y.abs().sum()
z.backward()
print(x.grad.shape)
