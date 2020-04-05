#!/home/sunhanbo/software/anaconda3/bin/python
#-*-coding:utf-8-*-
import pdb
import torch
import torch.nn as nn
from torch.autograd import Function
import torch.nn.functional as F
import math

class MyConv2dFunction(Function):
    @staticmethod
    def forward(ctx, input, weight, bias):
        N = input.shape[0]
        Cout = weight.shape[0]
        Hin = input.shape[2]
        Win = input.shape[3]
        Kh = weight.shape[2]
        Kw = weight.shape[3]
        Hout = Hin - Kh + 1
        Wout = Win - Kw + 1
        assert Kh == Kw
        K = Kh
        L = Hout * Wout
        output = torch.mm(F.unfold(input, K).transpose(1, 2).reshape((N*L,-1)),
                          weight.reshape((Cout,-1)).t()).reshape((N,L,Cout)).transpose(1, 2).reshape((N,Cout,Hout,Wout))
        output = output + bias.reshape((1,-1,1,1))
        ctx.save_for_backward(input, weight, bias)
        return output
    @staticmethod
    def backward(ctx, grad_output):
        input, weight, bias = ctx.saved_tensors
        N = grad_output.shape[0]
        Cout = grad_output.shape[1]
        Hout = grad_output.shape[2]
        Wout = grad_output.shape[3]
        Kh = weight.shape[2]
        Kw = weight.shape[3]
        Hin = Hout + Kh - 1
        Win = Wout + Kw - 1
        assert Kh == Kw
        K = Kh
        L = Hout * Wout
        tmp_grad_output = grad_output.reshape((N,Cout,-1)).transpose(1, 2).reshape((N*L,-1))
        grad_weight = torch.mm(F.unfold(input, K).transpose(1, 2).reshape((N*L,-1)).t(), tmp_grad_output).transpose(0,1).reshape((Cout,-1,K,K))
        grad_input = F.fold(torch.mm(tmp_grad_output, weight.reshape((Cout,-1))).reshape((N,L,-1)).transpose(1, 2), (Hin, Win), K)
        grad_bias = torch.sum(grad_output, (0, 2, 3))
        return grad_input, grad_weight, grad_bias
MyConv2d = MyConv2dFunction.apply

class MyConv2dModule(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size):
        super(MyConv2dModule, self).__init__()
        self.weight = nn.Parameter(torch.zeros((out_channels, in_channels, kernel_size, kernel_size)))
        self.bias = nn.Parameter(torch.zeros(out_channels))
        self.reset_parameters()
    def reset_parameters(self):
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.weight)
        bound = 1 / math.sqrt(fan_in)
        nn.init.uniform_(self.bias, -bound, bound)
    def forward(self, input):
        return MyConv2d(input, self.weight, self.bias)

# l = MyConv2dModule(2, 2, 3)
# l = nn.Conv2d(2, 2, 3)
# l.weight.data.copy_(torch.arange(36.).reshape((2,2,3,3)))
# l.bias.data.copy_(torch.arange(2.))
# x = torch.arange(36.).reshape((2,2,3,3)).requires_grad_()
# s = l(x)
# print(s)
# s.sum().backward()
# print(x.grad)
# print(l.weight.grad)
# print(l.bias.grad)

class MyLinearFunction(Function):
    @staticmethod
    def forward(ctx, input, weight, bias):
        output = torch.mm(input, weight.t())
        output = output + bias.reshape((1,-1))
        ctx.save_for_backward(input, weight, bias)
        return output
    @staticmethod
    def backward(ctx, grad_output):
        input, weight, bias = ctx.saved_tensors
        grad_input = torch.mm(grad_output, weight)
        grad_weight = torch.mm(grad_output.t(), input)
        grad_bias = torch.sum(grad_output, 0)
        return grad_input, grad_weight, grad_bias
MyLinear = MyLinearFunction.apply

class MyLinearModule(nn.Module):
    def __init__(self, in_features, out_features):
        super(MyLinearModule, self).__init__()
        self.weight = nn.Parameter(torch.zeros((out_features, in_features)))
        self.bias = nn.Parameter(torch.zeros(out_features))
        self.reset_parameters()
    def reset_parameters(self):
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.weight)
        bound = 1 / math.sqrt(fan_in)
        nn.init.uniform_(self.bias, -bound, bound)
    def forward(self, input):
        return MyLinear(input, self.weight, self.bias)

# l = MyLinearModule(3, 4)
# l = nn.Linear(3, 4)
# l.weight.data.copy_(torch.arange(12).reshape((4,3)))
# l.bias.data.copy_(torch.arange(4))
# x = torch.arange(6.).reshape((2,3)).requires_grad_()
# s = l(x)
# print(s)
# s.sum().backward()
# print(x.grad)
# print(l.weight.grad)
# print(l.bias.grad)

class MyBatchNormFunction(Function):
    @staticmethod
    def forward(ctx, input, running_mean, running_var, weight, bias, training_state, momentum, eps):
        if training_state:
            tmp_mean = torch.mean(input, (0,2,3))
            tmp_var = torch.mean((input - tmp_mean.reshape((1,-1,1,1))).pow(2), (0,2,3))
            N = input.shape[0] * input.shape[2] * input.shape[3]
            running_mean.copy_((1 - momentum) * running_mean + momentum * tmp_mean)
            running_var.copy_((1 - momentum) * running_var + momentum * tmp_var * N / (N - 1))
        else:
            tmp_mean = running_mean
            tmp_var = running_var
        inter_var = (tmp_var + eps).pow(0.5).reshape((1,-1,1,1))
        inter_mean = input - tmp_mean.reshape((1,-1,1,1))
        inter_output = torch.div(inter_mean, inter_var)
        inter_info = torch.tensor(training_state, device = input.device)
        ctx.save_for_backward(inter_output, inter_mean, inter_var, weight, inter_info)
        output = torch.mul(inter_output, weight.reshape((1,-1,1,1))) + bias.reshape((1,-1,1,1))
        return output
    @staticmethod
    def backward(ctx, grad_output):
        N = grad_output.shape[0] * grad_output.shape[2] * grad_output.shape[3]
        inter_output, inter_mean, inter_var, weight, inter_info = ctx.saved_tensors
        grad_weight = torch.sum(torch.mul(grad_output, inter_output), (0,2,3))
        grad_bias = torch.sum(grad_output, (0,2,3))
        # 由于计算的复杂性，在这里分步计算
        if inter_info.item():
            l_xi = torch.mul(grad_output, weight.reshape((1,-1,1,1)))
            l_sigma = torch.sum(l_xi * inter_mean * inter_var.pow(-3) * (-0.5), (0,2,3))
            l_u = torch.sum(l_xi * inter_var.pow(-1) * (-1), (0,2,3))
            grad_input = l_xi * inter_var.pow(-1) + l_sigma.reshape((1,-1,1,1)) * inter_mean * (2/N) + (l_u * (1/N)).reshape((1,-1,1,1))
        else:
            grad_input = grad_output * weight.reshape((1,-1,1,1)) * inter_var.pow(-1)
        return grad_input, None, None, grad_weight, grad_bias, None, None, None
MyBatchNorm = MyBatchNormFunction.apply

class MyBatchNormModule(nn.Module):
    def __init__(self, num_features, eps = 1e-5, momentum = 0.1):
        super(MyBatchNormModule, self).__init__()
        self.weight = nn.Parameter(torch.zeros(num_features))
        self.bias = nn.Parameter(torch.zeros(num_features))
        self.register_buffer('running_mean', torch.zeros(num_features))
        self.register_buffer('running_var', torch.ones(num_features))
        self.eps = eps
        self.momentum = momentum
        self.reset_parameters()
    def reset_parameters(self):
        nn.init.uniform_(self.weight)
        nn.init.zeros_(self.bias)
    def forward(self, input):
        return MyBatchNorm(input, self.running_mean, self.running_var, self.weight, self.bias, self.training, self.momentum, self.eps)

# l = MyBatchNormModule(2, eps = 1)
# l = nn.BatchNorm2d(2, eps = 1)
# l.train()
# l.eval()
# l.weight.data.fill_(1)
# l.bias.data.fill_(0)
# x = torch.arange(16.).reshape((2,2,2,2)).requires_grad_()
# s = l(x).pow(2)
# s.sum().backward()
# print(s)
# print(x.grad)
# print(l.weight.grad)
# print(l.bias.grad)
# print(l.running_mean)
# print(l.running_var)

class MyDropOutFunction(Function):
    @staticmethod
    def forward(ctx, input, p, training_state):
        p = p if training_state else 0
        inter_mask = torch.bernoulli(torch.ones_like(input) * (1 - p))
        inter_info = torch.tensor([p, training_state], device = input.device)
        ctx.save_for_backward(inter_mask, inter_info)
        return input * inter_mask * (1 / (1 - p))
    @staticmethod
    def backward(ctx, grad_output):
        inter_mask, inter_info = ctx.saved_tensors
        p = inter_info[0]
        grad_input = grad_output * inter_mask * (1 / (1 - p))
        return grad_input, None, None
MyDropOut = MyDropOutFunction.apply

class MyDropOutModule(nn.Module):
    def __init__(self, p = 0.5):
        super(MyDropOutModule, self).__init__()
        self.p = p
    def forward(self, input):
        return MyDropOut(input, self.p, self.training)

# l = MyDropOutModule()
# l = nn.Dropout()
# l.train()
# l.eval()
# x = torch.arange(16.).reshape((2,2,2,2)).requires_grad_()
# s = l(x)
# s.sum().backward()
# print(s)
# print(x.grad)
