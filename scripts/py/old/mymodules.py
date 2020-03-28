from torch.nn.parameter import Parameter
from torch.nn import init
from torch.autograd import Variable
from torch.autograd import Function
import torch.nn as nn
import torch
import ipdb

class MyBN_f(Function):

    @staticmethod
    def forward(ctx, inputs, running_mean, running_var, weight, bias, is_training ,momemtum, eps, returned_mean, returned_var):

        if is_training:
            # print("Is_Training")
            N = inputs.shape[0]*inputs.shape[1]*inputs.shape[2]
            mean = torch.mean(inputs, dim = [0,2,3])
            var = torch.var(inputs, unbiased = False, dim = [0,2,3])
            var = torch.mean((inputs - mean.reshape([1,-1,1,1]))**2, dim=[0,2,3])
            # N = inputs.shape[0]*inputs.shape[2]*inputs.shape[3]
            returned_mean.copy_(mean) # !!! Will Not Work On Multi-GPU!
            returned_var.copy_(var)

            # Replace The Running Mean/Var Inplace So We Don't Need To Return Them
            running_mean.copy_((1-momemtum)*running_mean + momemtum*mean)
            running_var.copy_((1-momemtum)*running_var + momemtum*var*(N/(N-1)))


        else:
            mean = running_mean
            var = running_var
        
        squared_var = torch.sqrt(var + eps).reshape([1,-1,1,1]) # Prepare For the Backward
        # The Main Body
        mean = mean.reshape([1,-1,1,1])
        var = var.reshape([1,-1,1,1])
        
        whitened_inputs = (inputs - mean) / \
            squared_var
        # returned_whiten_inputs.copy_(whitened_inputs) # Save The Whiten Input

        outputs = whitened_inputs*weight.reshape([1,-1,1,1]) + bias.reshape([1,-1,1,1])
        
        ctx.save_for_backward(whitened_inputs, mean, squared_var, weight, \
                              torch.tensor(is_training,device = inputs.device))

        return outputs

    @staticmethod
    def backward(ctx, grad_y):
        # print(grad_y.shape)
        N = grad_y.shape[0]*grad_y.shape[2]*grad_y.shape[3]
        whitened_inputs, mean, squared_var, weight, is_training = ctx.saved_tensors


        # The Main Body
        g_w = torch.sum(grad_y*whitened_inputs, dim = [0,2,3])
        g_b = torch.sum(grad_y, (0,2,3))
        if is_training.item():
            g_whitened_x = torch.mul(grad_y, weight.reshape([1,-1,1,1]))
            g_var = torch.sum(g_whitened_x *(whitened_inputs) * (squared_var.pow(-2)) * (-0.5), dim = [0,2,3]) # The grad for Var not Squared Var
            g_mean = torch.sum(g_whitened_x * squared_var.pow(-1) * (-1),dim = [0,2,3])


            # ipdb.set_trace()
            grad_x = (1/(N*squared_var))*(N*g_whitened_x - torch.sum(g_whitened_x, dim = [0,2,3]).reshape([1,-1,1,1]) \
                                            - whitened_inputs* torch.sum(g_whitened_x*whitened_inputs, dim = [0,2,3]).reshape([1,-1,1,1]) )
            # grad_x = g_whitened_x*squared_var.pow(-1) + ((1/N)*g_mean).reshape([1,-1,1,1])
                # + (whitened_inputs)*(squared_var)*(2/N)*(g_var.reshape([1,-1,1,1]))
        else:
            grad_x = grad_y * weight.reshape([1,-1,1,1]) * squared_var.pow(-1)

        return grad_x, None, None, g_w, g_b, None, None, None, g_mean, g_var


class MyBN(nn.Module):
    def __init__(self, num_features, eps = 1e-5, momemtum = 0.1, affine = True, track_running_stats = True):
        super(MyBN, self).__init__()
        self.num_features = num_features
        self.eps = eps
        self.momemtum = momemtum
        self.track_running_stats = track_running_stats
        self.affine = affine
        # self.mean = Variable(torch.Tensor(num_features))
        # self.var = Variable(torch.Tensor(num_features))
        # self.mean = torch.Tensor(num_features).requires_grad_() # Dirty, needs chanding
        # self.var = torch.Tensor(num_features).requires_grad_()

        self.register_buffer('mean', torch.Tensor(num_features).requires_grad_())
        self.register_buffer('var', torch.Tensor(num_features).requires_grad_())

        if self.affine:
            self.weight = Parameter(torch.Tensor(num_features))
            self.bias = Parameter(torch.Tensor(num_features))

        if self.track_running_stats:
            self.register_buffer('running_mean', torch.zeros(num_features))
            self.register_buffer('running_var', torch.zeros(num_features))
            self.register_buffer('num_batches_tracked', torch.zeros(num_features, dtype = torch.long))
        
        self.reset_parameters()

    def reset_running_stats(self):
        if self.track_running_stats:
            self.running_mean.zero_()
            self.running_var.fill_(1)
            self.num_batches_tracked.zero_()

    def reset_parameters(self):
        self.reset_running_stats()
        if self.affine:
            init.ones_(self.weight)
            init.zeros_(self.bias)

    def _check_input_dim(self, input):
        if (input.dim() != 4):
            raise ValueError("The Input Should Be in [Num_Batch, Channel, W, H]")

    def forward(self, inputs):
        self._check_input_dim(inputs)

        # whitened_inputs = torch.Tensor(inputs.size()) # Dirty, Need Proper Changin
        # whitened_inputs.requires_grad_().to(inputs.device)
        # self.whitened_inputs.to(inputs.device)
        # self.register_buffer('whitened_inputs', whitened_inputs)
        # self.whitened_inputs = self.whitened_inputs.to(inputs.device)

        return MyBN_f.apply(inputs, self.running_mean, self.running_var, self.weight, self.bias, \
                            self.training , self.momemtum, self.eps, self.mean, self.var)



import torch.nn.functional as F
class RandomMaskedConv(nn.Module):

    def __init__(self, in_c, out_c, kernel_size, beta = 0.5, *args):
    # Beta Being The Hyper-Param in Softmax Function (Maybe Simualted Annealing?)
        super(RandomMaskedConv, self).__init__()
        
        # Hyper-Parmas Here
        self.beta = beta
        self.in_c = in_c
        self.out_c = out_c
        self.sparsity_ratio = nn.Parameter(torch.tensor([0.5])) # How Many Channel To Keep
        self.type = 1 # Type 0 means FilterPrunnig / Type 1 means ChannelPrunning
        # The Traditional Conv
        self.conv = nn.Conv2d(in_c, out_c, kernel_size, stride = 1, padding = 0, 
                              dilation=1, groups=1, bias = True)
        


    def forward(self, x):

        if (self.type == 0):

            # Apply The Traditional Loss
            # Calc The Softmax Of Filters According to L1 Norm
            # The Weight Shape [out_c, in_c, kernel_size, kernel_size]
            filters_l1 = self.conv.weight.abs().sum(dim = [1,2,3]) # Getting The L1 Sum of each Filter
            filters_softmax = (filters_l1*self.beta).exp() / (filters_l1*self.beta).exp().sum()
            # Mask In The Out_C, Generating Masks
            probs = torch.clamp(self.sparsity_ratio*self.out_c*filters_softmax,0,1)
            masks = torch.bernoulli(probs)

            y = self.conv(x)
            y = F.relu(y)
            y = y*masks.reshape([1,-1,1,1])

        if (self.type == 1):
            channels_l1 = self.conv.weight.abs().sum(dim = [0,2,3]) 
            channels_softmax = (channels_l1*self.beta).exp() / (channels_l1*self.beta).exp().sum()
            probs = torch.clamp(self.sparsity_ratio*self.in_c*channels_softmax,0,1)
            masks = torch.bernoulli(probs)
            print(masks)
            self.conv.weight = nn.Parameter(self.conv.weight*masks.reshape([1,-1,1,1]))
            y = F.relu(self.conv(x))

        return y
        


