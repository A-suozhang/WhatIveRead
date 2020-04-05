######################################################################
# (c) Copyright EFC of NICS, Tsinghua University. All rights reserved.
#  Author: Kai Zhong
#  Email : zhongk19@mails.tsinghua.edu.cn
#  
#  Create Date : 2019.04.09
#  File Name   : Qmodules.py
#  Description : quantized neural network modules 
#  Dependencies: 
#  Ported from https://github.com/
#  and         https://github.com/
######################################################################

import pdb
import math
import numpy as np
import torch
import torch.nn as nn 
import torch.nn.functional as F
from torch.autograd.function import InplaceFunction, Function

__all__ = ['QConv2d', 'QLinear', 'QBatchNorm2d']


class Quantize_W(Function):
    ''' Function which quantizes w directly during both forward and backward
        Call it by Quantize_W.apply(weight, ...)
    '''

    @staticmethod
    def forward(ctx, input, num_bits, gnum_bits, max_value, min_value, signed=True,
                stochastic=False, dequantize=True, inplace=False, magnitude=None):
        ''' Forward function
                input           : unquantized weight       
                num_bits        : bw of quantized weight 
                gnum_bits       : bw of gradient of w, don't quantize g of w if ==0
                max/min_value   : max/min of unquantized weight
                signed          : signed quantize or not    
                stochastic      : add noise when quantize or not 
                dequantize      : dequantize 
                inplace         : inplace 
                magnitude       : quantize with ceil magnitude if ==ceil or GEMMLOWP if ==None
                lr_scale        : learning rate scale p
        '''
        ctx.num_bits   = num_bits
        ctx.gnum_bits  = gnum_bits
        ctx.signed     = signed
        ctx.stochastic = stochastic
        ctx.dequantize = dequantize
        ctx.inplace    = False
        ctx.magnitude  = magnitude
        ctx.p          = 0              # temp of ctx.p

        if ctx.inplace:
            ctx.mark_dirty(input)
            output = input
        else:
            output = input.clone()

        qmin = -(2.**(num_bits - 1)) if signed else 0.
        qmax = qmin + 2.**num_bits - 1.
        scale = 1
        zero_point = 0

        if magnitude is None:
            # TODO: re-add true zero computation
            # using GEMMLOWP quantize, shift and scale real input to qmin~qmax
            range_value = max_value - min_value
            zero_point = min_value
            scale = range_value / (qmax - qmin)
            #print("zkdebug w quantize gemmlowp, scale", scale)
        elif magnitude == 'ceil':
            # using magnitude quantize, just magnitude scale real input within qmin~qmax
            range_value = max(abs(max_value), abs(min_value))
            p = torch.ceil(torch.log2(range_value))-num_bits+1 if signed else \
                torch.ceil(torch.log2(range_value))-num_bits
            ctx.p = p
            scale = 2.**p 
            zero_point = qmin * scale           # just for matching the code following
            #print("zkdebug w quantize magnitude, range_value:", range_value)
            #print("zkdebug w quantize magnitude, num_bits:", num_bits)
            #print("zkdebug w quantize magnitude, p:", p)
            #print("zkdebug w quantize magnitude, scale:", scale)
        else:
            print(magnitude)
            exit("ERROR w quantize magnitude error")
                 
        with torch.no_grad():
            output.add_(qmin * scale - zero_point).div_(scale)
            #print("zkdebug w min max:", output.min(), output.max())
            #print("zkdebug w qmin qmax:", qmin, qmax)
            if stochastic:
                noise = output.new(output.shape).uniform_(-0.5, 0.5)
                output.add_(noise)
            output.clamp_(qmin, qmax).round_() # quantize
            if dequantize:
                output.mul_(scale).add_(zero_point - qmin * scale) # dequantize

        return output


    @staticmethod
    def backward(ctx, grad_output):

        ''' Backword function
                ctx         : ctx record info from forward
                grad_output : unquantized gradient of weight
        '''

        if ctx.gnum_bits == 0:      # don't do quantization
            grad_input = grad_output
            
        else:                       # do quantization
            grad_input = grad_output.clone()
            qmin = -(2.**(ctx.gnum_bits - 1)) if ctx.signed else 0.
            qmax = qmin + 2.**ctx.gnum_bits - 1.
            scale = 1
            zero_point = 0

            if ctx.magnitude is None:
                # TODO: re-add true zero computation
                # using GEMMLOWP quantize so calculate min max first
                max_value = grad_input.max()
                min_value = grad_input.min()
                range_value = max_value - min_value
                zero_point = min_value
                scale = range_value / (qmax - qmin)
                #print("zkdebug gw quantize gemmlowp, scale:", scale)
            elif ctx.magnitude == 'ceil':
                # using magnitude quantize and get p from forward
                p = ctx.p - (ctx.gnum_bits - ctx.num_bits)
                scale = 2.**p
                zero_point = qmin * scale
                #print("zkdebug gw quantize magnitude, ctx.p:", ctx.p)
                #print("zkdebug gw quantize magnitude, gnum_bits:", ctx.gnum_bits)
                #print("zkdebug gw quantize magnitude, num_bits:", ctx.num_bits)
                #print("zkdebug gw quantize magnitude, p:", p)
            else:
                print(ctx.magnitude)
                exit("gw quantize magnitude error")

            with torch.no_grad():
                (grad_input.add_(qmin * scale - zero_point)).div_(scale)
                #print("zkdebug gw min max:", grad_input.min(), grad_input.max())
                #print("zkdebug gw qmin qmax:", qmin, qmax)
                if ctx.stochastic:
                    noise = grad_input.new(grad_input.shape).uniform_(-0.5, 0.5)
                    grad_input.add_(noise)
                grad_input.clamp_(qmin, qmax).round_() # quantize
                if ctx.dequantize:
                    grad_input.mul_(scale).add_(zero_point - qmin * scale) # dequantize

        return grad_input, None, None, None, None, None, None, None, None, None



class Quantize_A(Function):

    ''' This is the function which quantizes a directly during forward
        Call it by Quantize_A.apply(input, ...)
    '''

    @staticmethod
    def forward(ctx, input, num_bits, max_value, min_value, signed=True, 
                stochastic=False, dequantize=True, inplace=False, magnitude=None):

        ''' Function arguements:
                input       : unquantized input         
                num_bits    : bw of quantized input
                max_value   : max of input              
                min_value   : min of input
                signed      : signed quantize or not    
                stochastic  : add noise when quantize or not 
                dequantize  : dequantize                
                inplace     : inplace 
                magnitude   : quantize with ceil magnitude if ==ceil or GEMMLOWP if ==None
        '''

        ctx.inplace = inplace
        if ctx.inplace:
            ctx.mark_dirty(input)
            output = input
        else:
            output = input.clone()

        qmin  = -(2.**(num_bits - 1)) if signed else 0.
        qmax  = qmin + 2.**num_bits - 1.
        scale = 1
        zero_point = 0

        if magnitude is None:
            # TODO: re-add true zero computation
            # using GEMMLOWP quantize, shift and scale real input to qmin~qmax
            range_value = max_value - min_value
            zero_point = min_value
            scale = range_value / (qmax - qmin)
            #print("zkdebug a quantize gemmlowp, scale:", scale)
        elif magnitude == 'ceil':
            # using magnitude quantize, just magnitude scale real input within qmin~qmax
            range_value = max(abs(max_value), abs(min_value))
            p = torch.ceil(torch.log2(range_value))-num_bits+1 if signed else \
                torch.ceil(torch.log2(range_value))-num_bits
            if np.isnan(p.cpu()):
                exit('ERROR p of a data nan')
            scale = 2.**p 
            zero_point = qmin * scale
            ##print("zkdebug a 0 quantize magnitude, range_value:", range_value)
            #print("zkdebug a quantize magnitude, num_bits:", num_bits)
            #print("zkdebug a quantize magnitude, p:", p)
            #print("zkdebug a quantize magnitude, scale:", scale)
        else:
            print(magnitude)
            exit("ERROR a quantize magnitude error")
                    
        with torch.no_grad():
            (output.add_(qmin * scale - zero_point)).div_(scale)
            #print("zkdebug a zeropoint:", zero_point)
            #print("zkdebug a 1 min max:", output.min(), output.max())
            #print("zkdebug a qmin qmax:", qmin, qmax)
            if np.isnan(output.min().cpu()):
                exit('ERROR activation data nan')
            if stochastic:
                noise = output.new(output.shape).uniform_(-0.5, 0.5)
                output.add_(noise)
            # quantize
            output.clamp_(qmin, qmax).round_()
            if dequantize:
                output.mul_(scale).add_(zero_point - qmin * scale) # dequantize

        return output

    @staticmethod
    def backward(ctx, grad_output):
        # straight-through estimator

        grad_input = grad_output
        ##torch.set_printoptions(precision=8)
        ##print("- 0 zkdebug grad of input:",grad_input.max(), grad_input.min(), grad_input.shape)

        return grad_input, None, None, None, None, None, None, None, None



class Quantize_G(Function):

    ''' This is the function which quantizes g directly during backward
        Call it by Quantize_G.apply(input, ...)
    '''

    @staticmethod
    def forward(ctx, input, num_bits, signed=True, stochastic=True, 
                dequantize=True, inplace=False, magnitude=None):

        ''' Function arguements:
                input       : activation input                     
                num_bits    : bw of quantized gradient 
                signed      : signed quantize or not    
                stochastic  : add noise when quantize or not 
                dequantize  : dequantize                
                inplace     : inplace 
                magnitude   : quantize with ceel magnitude if ==ceil or GEMMLOWP if ==None
        '''

        # record info and return input when forward
        ctx.num_bits   = num_bits
        ctx.signed     = signed
        ctx.stochastic = stochastic
        ctx.dequantize = dequantize
        ctx.inplace    = False
        ctx.magnitude  = magnitude
        #print("zkdebug g 22 quantize")
        output = input.clone()

        return output


    @staticmethod
    def backward(ctx, grad_output):
        #print("zkdebug g backward")

        grad_input = grad_output.clone()

        num_bits = ctx.num_bits
        qmin = -(2.**(num_bits - 1)) if ctx.signed else 0.
        qmax = qmin + 2.**num_bits - 1.
        scale = 1
        zero_point = 0

        # TODO: decide how to do dynamic quantize based on up to date distribution
        # and add num_chunks into cfg and change it to suitable form
        num_chunks = 1
        if len(grad_output.shape) == 4:
            B,C,H,W = grad_output.shape
            g = grad_output.transpose(0, 1).contiguous()  # C x B x H x W
            g = g.view(C, num_chunks, (B*H*W) // num_chunks)
            max_value = g.max(-1)[0].max()  # calculate max of maxs of C*num_chunks chunks
            min_value = g.min(-1)[0].min()  # calculate min of mins of C*num_chunks chunks
        else:
            B,C = grad_output.shape
            max_value = grad_output.max() 
            min_value = grad_output.min()

        if ctx.magnitude is None:
            # TODO: re-add true zero computation
            # using GEMMLOWP quantize, shift and scale real input to qmin~qmax
            range_value = max_value - min_value
            zero_point = min_value
            scale = range_value / (qmax - qmin)
            #print("zkdebug g quantize gemmlowp, scale:", scale)
        elif ctx.magnitude == 'ceil':
            # using magnitude quantize, just magnitude scale real input within qmin~qmax
            range_value = max(abs(max_value), abs(min_value))
            p = torch.ceil(torch.log2(range_value))-num_bits+1 if ctx.signed else \
                torch.ceil(torch.log2(range_value))-num_bits
            scale = 2.**p 
            zero_point = qmin * scale
            ##print("1 zkdebug grad of output, range_value:", range_value, max_value, min_value)
            #print("zkdebug g quantize magnitude, num_bits:", num_bits)
            #print("2 zkdebug grad of output, p:", p)
            ##print("2 zkdebug grad of output, shape:", grad_output.shape)
            #print("4 zkdebug g quantize magnitude, scale:", scale)
            if range_value <= 0:
                #pdb.set_trace()
                exit("ERROR g range_value <= 0")
        else:
            print(ctx.magnitude)
            exit("g quantize magnitude error")

        with torch.no_grad():
            grad_input.add_(qmin * scale - zero_point).div_(scale)
            #print("zkdebug g min max:", grad_input.min(), grad_input.max())
            #print("zkdebug g qmin qmax:", qmin, qmax)
            if ctx.stochastic:
                noise = grad_input.new(grad_input.shape).uniform_(-0.5, 0.5)
                grad_input.add_(noise)
            # quantize
            grad_input.clamp_(qmin, qmax).round_()
            if ctx.dequantize:
                grad_input.mul_(scale).add_(zero_point - qmin * scale) # dequantize

            #print("4 zkdebug grad of output, quantized:", grad_input.max(), grad_input.min())

        return grad_input, None, None, None, None, None, None



def gQuantize(g, bw=8, signed=True, stochastic=True, dequantize=True, inplace=False, magnitude=None):
    ''' This is the function which does g quantize by calling Quantize_G
    '''
    return Quantize_G.apply(g, bw, signed, stochastic, dequantize, inplace, magnitude)



class wQuantizer(nn.Module):
    ''' This is the class which calculates distribute and does w quantize
    '''
    def __init__(self, bw=8, gbw=0, wbasis='this', wvbasis='last1', momentum=0.125, signed=True, 
                    stochastic=True, dequantize=True, inplace=False, magnitude=None):
        super(wQuantizer, self).__init__()
        self.quantize   = False
        self.bw         = bw
        self.gbw        = gbw             # bit width of gradient of weight, 0 for no quantize
        self.wbasis     = wbasis          # quantize measure basis during training
        self.wvbasis    = wvbasis         # quantize measure basis during validation
        self.momentum   = momentum        # momentum to calculate running basis
        self.signed     = signed          # bool of signed quantize
        self.stochastic = stochastic      # bool of stochastic rounding during quantize
        self.dequantize = dequantize      # bool of dequantize during the software simulation
        self.inplace    = inplace         # bool of inplace druing quantize
        self.magnitude  = magnitude       # none for GEMMLOWP quantize and ceil for magnitude quantize
        self.lr_scale_p = 0               # gw bit cut from learning rate scale
                
        # buffer of quantize basis
        self.register_buffer('run_max', torch.zeros(1))
        self.register_buffer('run_min', torch.zeros(1))
        self.register_buffer('last2_max', torch.zeros(1))
        self.register_buffer('last2_min', torch.zeros(1))
        self.register_buffer('last1_max', torch.zeros(1))
        self.register_buffer('last1_min', torch.zeros(1))
        self.register_buffer('first_max', torch.zeros(1))
        self.register_buffer('first_min', torch.zeros(1))
        self.register_buffer('first'    , torch.ones(1))

    def forward(self, w):

        if self.training:
            with torch.no_grad():
                # calculate min max
                max_value = w.max()                  # calculate max directely
                min_value = w.min()                  # calculate min directely

                if self.quantize:
                    # this first register should not be update when float
                    self.first_max.add_(self.last1_max.mul(self.first))
                    self.first_min.add_(self.last1_min.mul(self.first))
                    self.first.mul_(0)      # it should not be update after quantize either
                    
                    # get used param for quantize
                    if   self.wbasis == 'this':
                        used_max = max_value
                        used_min = min_value
                    elif self.wbasis == 'stable':
                        used_max = self.first_max
                        used_min = self.first_min
                    elif self.wbasis == 'run':
                        used_max = self.run_max
                        used_min = self.run_min
                    elif self.wbasis == 'last2_max':
                        used_max = torch.max(self.last2_max, self.last1_max)
                        used_min = torch.min(self.last2_min, self.last1_min)
                    else:
                        exit("w quantize used real")
                    if used_max.requires_grad:
                        print("zkdebug w train used max min:", used_max, used_min)
                        print("zkdebug w train max min:", max_value, min_value)
                        exit("ERROR used_max require grad")

                self.run_max.mul_(self.momentum).add_(max_value.mul(1-self.momentum))
                self.run_min.mul_(self.momentum).add_(min_value.mul(1-self.momentum))
                self.last2_max.copy_(self.last1_max)
                self.last2_min.copy_(self.last1_min)
                self.last1_max.copy_(max_value)
                self.last1_min.copy_(min_value)
                #print("zkdebug w run max min:", self.run_max, self.run_min)
                if self.run_max.requires_grad:
                    exit("ERROR run_max require grad")

        # if not training, do quantize if quantize while do nothing if float
        elif self.quantize:
            with torch.no_grad():
                if   self.wvbasis == 'stable':
                    used_max = self.first_max
                    used_min = self.first_min
                elif self.wvbasis == 'last1':
                    used_max = self.last1_max
                    used_min = self.last1_min
                elif self.wvbasis == 'run':
                    used_max = self.run_max
                    used_min = self.run_min
                elif self.wvbasis == 'last2_max':
                    used_max = torch.max(self.last2_max, self.last1_max)
                    used_min = torch.min(self.last2_min, self.last1_min)
                else:
                    exit("w quantize test used real")

        else:
            pass

        if self.quantize:
            return Quantize_W.apply(w, self.bw, self.gbw+self.lr_scale_p, used_max, used_min, 
                                    self.signed, self.stochastic, self.dequantize, 
                                    self.inplace, self.magnitude)
        else:
            return w



class aQuantizer(nn.Module):
    ''' This is the class which calculates distribute and does a quantize
    '''
    def __init__(self, bw=8, tbasis='this', vbasis='run', momentum=0.125, signed=False, stochastic=False, 
                    dequantize=True, inplace=False, magnitude=None, place='conv'):
        super(aQuantizer, self).__init__()
        self.quantize   = False           # calculate with float at the begining
        self.place      = place
        self.bw         = bw
        self.tbasis     = tbasis          # quantize measure basis during training
        self.vbasis     = vbasis          # quantize measure basis during validation
        self.momentum   = momentum        # momentum to calculate running basis
        self.signed     = signed          # bool of signed quantize
        self.stochastic = stochastic      # bool of stochastic rounding during quantize
        self.dequantize = dequantize      # bool of dequantize during the software simulation
        self.inplace    = inplace         # bool of inplace druing quantize
        self.magnitude  = magnitude       # None for GEMMLOWP quantize and ceil for magnitude
        self.num_chunks = 1
        # TODO: decide how to do dynamic quantize based on up to date distribution
        # and add num_chunks into cfg and change it to suitable form

        # buffer of quantize basis
        self.register_buffer('run_max'  , torch.zeros(1))
        self.register_buffer('run_min'  , torch.zeros(1))
        self.register_buffer('mrun_max' , torch.zeros(1))
        self.register_buffer('mrun_min' , torch.zeros(1))
        self.register_buffer('last2_max', torch.zeros(1))
        self.register_buffer('last2_min', torch.zeros(1))
        self.register_buffer('last1_max', torch.zeros(1))
        self.register_buffer('last1_min', torch.zeros(1))
        #self.register_buffer('real_max' , torch.zeros(1))
        #self.register_buffer('real_min' , torch.zeros(1))
        self.register_buffer('first_max', torch.zeros(1))
        self.register_buffer('first_min', torch.zeros(1))
        #self.register_buffer('number'   , torch.zeros(1))
        self.register_buffer('first'    , torch.ones(1))
        #print("zkdebug aQuantizer training",self.training)
        #print("zkdebug aQuantizer basis",self.basis)

    def forward(self, x):
        
        # when training, only update registers if float while update registers and set used value if quant
        if self.training:
            with torch.no_grad():
                # calculate current data max and min value
                if len(x.shape) == 2:
                    B,C = x.shape
                    # TODO: support quantize of activation of Linear layers by chunks
                    max_value = x.max()  
                    min_value = x.min() 
                else:
                    B,C,H,W = x.shape
                    y = x.transpose(0, 1).contiguous()  # C x B x H x W
                    y = y.view(C, self.num_chunks, (B*H*W) // self.num_chunks)
                    max_value = y.max(-1)[0].max()  # calculate max of maxs of C*num_chunks chunks
                    min_value = y.min(-1)[0].min()  # calculate min of mins of C*num_chunks chunks

                #print("zkdebug max min",max_value,min_value)
                if self.quantize:
                    # this first register should not be update when float
                    self.first_max.add_((torch.max(self.last2_max,self.last1_max).sub_(
                                         torch.min(self.last2_max,self.last1_max)).add(self.last1_max)
                                        ).mul(self.first).mul(1.1))
                    self.first_min.add_((torch.min(self.last2_min,self.last1_min).sub_(
                                         torch.max(self.last2_min,self.last1_min)).add(self.last1_min)
                                        ).mul(self.first).mul(1.1))
                    self.first.mul_(0)
                    if   self.tbasis == 'this':
                        used_max = max_value
                        used_min = min_value
                    elif self.tbasis == 'stable':
                        used_max = self.first_max
                        used_min = self.first_min
                    elif self.tbasis == 'run':
                        used_max = self.run_max.clone().detach()
                        used_min = self.run_min.clone().detach()
                    elif self.tbasis == 'mrun':
                        used_max = self.mrun_max.clone().detach()
                        used_min = self.mrun_min.clone().detach()
                    elif self.tbasis == 'last2_mean':
                        used_max = self.last2_max.mul(0.25).add(self.last1_max.mul(0.75))
                        used_min = self.last2_min.mul(0.25).add(self.last1_min.mul(0.75))
                    elif self.tbasis == 'last2_line':
                        #print("zkdebug last2 last1", self.place, self.last2_max,self.last1_max)
                        used_max = self.last1_max.sub(self.last2_max).add(self.last1_max)
                        #print("zkdebug last2 line",used_max)
                        used_min = self.last1_min.sub(self.last2_min).add(self.last1_min)
                    elif self.tbasis == 'last2_max':
                        #print("zkdebug last2 last1",self.last2_max,self.last1_max)
                        used_max = torch.max(self.last2_max, self.last1_max)
                        #print("zkdebug last2 max",used_max)
                        used_min = torch.min(self.last2_min, self.last1_min)
                    elif self.tbasis == 'last2_pre':
                        #print("zkdebug last2 last1",self.last2_max,self.last1_max)
                        used_max = torch.max(self.last2_max,self.last1_max).sub_(
                                   torch.min(self.last2_max,self.last1_max)).add_(self.last1_max)
                        #print("zkdebug last2 pre",used_max)
                        used_min = torch.min(self.last2_min,self.last1_min).sub_(
                                   torch.max(self.last2_min,self.last1_min)).add_(self.last1_min)
                    else:
                        exit("ERROR train using real")
                        used_max = self.real_max
                        used_min = self.real_min

                # the following registers should be calculated whether quantize or not
                self.run_max.mul_(self.momentum).add_(max_value.mul(1-self.momentum))
                self.run_min.mul_(self.momentum).add_(min_value.mul(1-self.momentum))
                #print("zkdebug mrun max_value",self.mrun_max, max_value)
                self.mrun_max.copy_(torch.max(self.mrun_max,max_value).mul_(1-self.momentum).add_(
                                    torch.min(self.mrun_max,max_value).mul_(self.momentum))
                                   )
                #print("zkdebug new mrun",self.mrun_max)
                self.mrun_min.copy_(torch.min(self.mrun_min,min_value).mul_(1-self.momentum).add_(
                                    torch.max(self.mrun_min,min_value).mul_(self.momentum))
                                   )
                self.last2_max.copy_(self.last1_max)
                self.last2_min.copy_(self.last1_min)
                self.last1_max.copy_(max_value)
                self.last1_min.copy_(min_value)
                #self.real_max.mul_(self.number).div_(self.number+1).add_(max_value.div(self.number+1)) 
                #self.real_min.mul_(self.number).div_(self.number+1).add_(max_value.div(self.number+1)) 
                #self.number.add_(1)
                #print("zkdebug number",self.number)

        # if not training, do quantize if quantize while do nothing if float
        elif self.quantize:
            with torch.no_grad():
                if   self.vbasis == 'stable':
                    used_max = self.first_max
                    used_min = self.first_min
                elif self.vbasis == 'run':
                    used_max = self.run_max
                    used_min = self.run_min
                elif self.vbasis == 'mrun':
                    used_max = self.mrun_max
                    used_min = self.mrun_min
                elif self.vbasis == 'last2_mean':
                    used_max = self.last2_max.mul(0.25).add(self.last1_max.mul(0.75))
                    used_min = self.last2_min.mul(0.25).add(self.last1_min.mul(0.75))
                elif self.vbasis == 'last2_line':
                    used_max = self.last1_max.add(self.last1_max.sub(self.last2_max))
                    used_min = self.last1_min.add(self.last1_min.sub(self.last2_min))
                elif self.vbasis == 'last2_max':
                    used_max = torch.max(self.last2_max, self.last1_max)
                    used_min = torch.min(self.last2_min, self.last1_min)
                elif self.vbasis == 'last2_pre':
                    used_max = torch.max(self.last2_max,self.last1_max).sub_(
                               torch.min(self.last2_max,self.last1_max)).add_(self.last1_max)
                    used_min = torch.min(self.last2_min,self.last1_min).sub_(
                               torch.max(self.last2_min,self.last1_min)).add_(self.last1_min)
                else:
                    exit("ERROR test using real")
        else:
            pass

        if self.quantize:
            #print("zkdebug quantize ---- Q A once")
            return Quantize_A.apply(x, self.bw, used_max, used_min, self.signed,
                        self.stochastic, self.dequantize, self.inplace, self.magnitude)
        else:
            return x



class QConv2d(nn.Conv2d):
    """quantized conv2d
    """

    def __init__(self, in_planes, out_planes, kernel_size, stride=1, padding=0,
                 dilation=1, groups=1, bias=True, q_cfg=None):
        super(QConv2d, self).__init__(in_planes, out_planes, kernel_size, stride, padding,
                 dilation, groups, bias)

        self.quantize    = False
        self.bw_a = q_cfg["bw"][0]                  # acivation bit width
        self.bw_w = q_cfg["bw"][1]                  # weight bit width
        self.bw_g = q_cfg["bw"][2]                  # gradient bit width
        self.bw_u = q_cfg["bw"][3]                  # update bit width
        self.tbasis      = q_cfg["basis"][0]        # quantize measure basis of training
        self.vbasis      = q_cfg["basis"][1]        # quantize measure basis of valid
        self.wbasis      = q_cfg["basis"][4]        # quantize measure basis of weight 
        self.wvbasis     = q_cfg["basis"][5]        # quantize measure basis of weight valid
        self.momentum    = q_cfg["momentum"]        # momentum to calculate running basis
        self.signed      = q_cfg["signed"]          # bool of signed quantize
        self.stochastic  = q_cfg["stochastic"]      # bool of stochastic rounding during quantize
        self.dequantize  = q_cfg["dequantize"]      # bool of dequantize during the software simulation
        self.inplace     = q_cfg["inplace"]         # bool of inplace druing quantize
        self.bifurcation = q_cfg["bifurcation"]     # bool of bifurcation fo gradient
        self.magnitude   = q_cfg["magnitude"]       # None for GEMMLOWP quantize and ceil for magnitude

        # w and aQuantizer are modules which will record the distribution during training
        # while gQuantize is only function containing InplaceFunction class so they are
        # quantized according to current distribute without history info
        self.aquantize_m = aQuantizer(bw=self.bw_a, tbasis=self.tbasis, vbasis=self.vbasis, 
                                      momentum=self.momentum, signed=self.signed, stochastic=self.stochastic,
                                      dequantize=self.dequantize, inplace=self.inplace, 
                                      magnitude=self.magnitude)
        self.wquantize_m = wQuantizer(bw=self.bw_w, gbw=self.bw_u, wbasis=self.wbasis, wvbasis=self.wvbasis,
                                      signed=self.signed, stochastic=self.stochastic, dequantize=self.dequantize,
                                      inplace=self.inplace, magnitude=self.magnitude)
        self.gquantize_f = gQuantize
        #print("zkdebug self training",self.training)

    def forward(self, input):
        
        Qinput  = self.aquantize_m(input)
        Qweight = self.wquantize_m(self.weight) 
        #Qweight = self.weight

        if self.bias is not None:
            bquantize_m = wQuantizer(bw=self.bw_w+self.bw_a, gbw=self.bw_u, wbasis=self.wbasis, 
                                     wvbasis=self.wvbasis, signed=self.signed, stochastic=self.stochastic, 
                                     dequantize=self.dequantize, inplace=self.inplace, 
                                     magnitude=self.magnitude)
            Qbias = bquantize_m(self.bias)
        else:
            Qbias = None

        if self.quantize:
            if not self.bifurcation:
                output_ = F.conv2d(Qinput, Qweight, Qbias, 
                                  self.stride, self.padding, self.dilation, self.groups)
                if self.bw_g is not None:
                    output = self.gquantize_f(output_, bw=self.bw_g, signed=self.signed, stochastic=True,
                              dequantize=self.dequantize, inplace=self.inplace, magnitude=self.magnitude)
                    #pdb.set_trace()
            else:
                out1 = F.conv2d(Qinput.detach(), Qweight, Qbias, self.stride,
                                self.padding, self.dilation, self.groups)
                out2 = F.conv2d(Qinput, Qweight.detach(), Qbias.detach() if Qbias is not None else None,
                                self.stride, self.padding, self.dilation, self.groups)
                out2 = self.gquantize_f(out2, bw=self.bw_g, signed=self.signed, stochastic=True,
                                dequantize=self.dequantize, inplace=self.inplace, magnitude=self.magnitude)
                output = out1 + out2 - out1.detach()
        else:
            output = F.conv2d(Qinput, Qweight, Qbias, 
                              self.stride, self.padding, self.dilation, self.groups)

        return output



class QLinear(nn.Linear):
    """quantized linear
    """

    def __init__(self, in_planes, out_planes, bias=True, q_cfg=None):
        super(QLinear, self).__init__(in_planes, out_planes, bias)

        self.quantize    = False
        self.bw_a = q_cfg["bw"][0]                  # acivation bit width
        self.bw_w = q_cfg["bw"][1]                  # weight bit width
        self.bw_g = q_cfg["bw"][2]                  # gradient bit width
        self.bw_u = q_cfg["bw"][3]                  # update bit width
        self.tbasis      = q_cfg["basis"][0]        # quantize measure basis of training
        self.vbasis      = q_cfg["basis"][1]        # quantize measure basis of valid
        self.wbasis      = q_cfg["basis"][4]        # quantize measure basis of weight
        self.wvbasis     = q_cfg["basis"][5]        # quantize measure basis of weight valid
        self.momentum    = q_cfg["momentum"]        # momentum to calculate running basis
        self.signed      = q_cfg["signed"]          # bool of signed quantize
        self.stochastic  = q_cfg["stochastic"]      # bool of stochastic rounding during quantize
        self.dequantize  = q_cfg["dequantize"]      # bool of dequantize during the software simulation
        self.inplace     = q_cfg["inplace"]         # bool of inplace druing quantize
        self.bifurcation = q_cfg["bifurcation"]     # bool of bifurcation fo gradient
        self.magnitude   = q_cfg["magnitude"]       # None for GEMMLOWP quantize and ceil for magnitude

        # aQuantizer is a module which will record the distribute of activation during training
        # while wQuantizer & gQuantizer are only functions containing InplaceFunction class so they are
        # be quantized according to current distribute without history info
        self.aquantize_m = aQuantizer(bw=self.bw_a, tbasis=self.tbasis, vbasis=self.vbasis, 
                                    momentum=self.momentum, signed=self.signed, stochastic=self.stochastic,
                                    dequantize=self.dequantize,inplace=self.inplace,
                                    magnitude=self.magnitude)
        self.wquantize_m = wQuantizer(bw=self.bw_w, gbw=self.bw_u, wbasis=self.wbasis, wvbasis=self.wvbasis,
                                    signed=self.signed, stochastic=self.stochastic, dequantize=self.dequantize,
                                    inplace=self.inplace, magnitude=self.magnitude)
        self.gquantize_f = gQuantize

    def forward(self, input):

        Qinput  = self.aquantize_m(input)
        Qweight = self.wquantize_m(self.weight) 
        #Qweight = self.weight

        if self.bias is not None:
            bquantize_m = wQuantizer(bw=self.bw_w+self.bw_a, gbw=self.bw_u, wbasis=self.wbasis, 
                                     wvbasis=self.wvbasis, signed=self.signed, stochastic=self.stochastic, 
                                     dequantize=self.dequantize, inplace=self.inplace, 
                                     magnitude=self.magnitude)
            Qbias = bquantize_m(self.bias)
        else:
            Qbias = None

        if self.quantize:
            if not self.bifurcation:
                output = F.linear(Qinput, Qweight, Qbias)
                if self.bw_g is not None:
                    output = self.gquantize_f(output, bw=self.bw_g, signed=self.signed, stochastic=True,
                             dequantize=self.dequantize, inplace=self.inplace, magnitude=self.magnitude)
                    #pdb.set_trace()
            else:
                out1 = F.linear(Qinput.detach(), Qweight, Qbias)
                out2 = F.linear(Qinput, Qweight.detach(), Qbias.detach() if Qbias is not None else None)
                out2 = self.gquantize_f(out2, bw=self.bw_g, signed=self.signed, stochastic=True, 
                       dequantize=self.dequantize, inplace=self.inplace, magnitude=self.magnitude)
                output = out1 + out2 - out1.detach()
        else:
            output = F.linear(Qinput, Qweight, Qbias)

        return output



class QBatchNorm2d(nn.Module):
    """quantized batchnorm"""

    def __init__(self, num_planes, dim=1, affine=True, eps=1e-5, num_chunks=1, q_cfg=None):
        super(QBatchNorm2d, self).__init__()

        self.quantize = False
        self.quantize_first = False
        self.dim = dim
        if affine:
            self.bias = nn.Parameter(torch.Tensor(num_planes))
            self.weight = nn.Parameter(torch.Tensor(num_planes))
        self.eps = eps
        self.num_chunks = num_chunks
        self.bw_a = q_cfg["bw"][0]                  # acivation bit width
        self.bw_w = q_cfg["bw"][1]                  # weight bit width
        self.bw_g = q_cfg["bw"][2]                  # gradient bit width
        self.bw_u = q_cfg["bw"][3]                  # update bit width
        self.tbasis      = q_cfg["basis"][0]        # quantize measure basis of training
        self.vbasis      = q_cfg["basis"][1]        # quantize measure basis of valid
        self.rtbasis     = q_cfg["basis"][2]        # nomalize measure basis of BN of training
        self.rvbasis     = q_cfg["basis"][3]        # nomalize measure basis of BN of valid
        self.wbasis      = q_cfg["basis"][4]        # quantize measure basis of weight
        self.wvbasis     = q_cfg["basis"][5]        # quantize measure basis of weight valid
        self.momentum    = q_cfg["momentum"]        # momentum to calculate running basis
        self.signed      = q_cfg["signed"]          # bool of signed quantize
        self.stochastic  = q_cfg["stochastic"]      # bool of stochastic rounding during quantize
        self.dequantize  = q_cfg["dequantize"]      # bool of dequantize during the software simulation
        self.inplace     = q_cfg["inplace"]         # bool of inplace druing quantize
        self.bifurcation = q_cfg["bifurcation"]     # bool of bifurcation fo gradient
        self.magnitude   = q_cfg["magnitude"]       # 
        self.reset_params()

        # w and aQuantizer are modules which will record the distribute of activation during training
        # while gQuantize is only function containing InplaceFunction class so they can only
        # be quantized according to current distribute without history info
        self.aquantize_m = aQuantizer(bw=self.bw_a, tbasis=self.tbasis, vbasis=self.vbasis, 
                                      momentum=self.momentum, signed=self.signed, stochastic=self.stochastic,
                                      dequantize=self.dequantize,inplace=self.inplace, place=' --BN--')
        self.wquantize_m = wQuantizer(bw=self.bw_w, gbw=self.bw_u, wbasis=self.wbasis, wvbasis=self.wvbasis,
                                      signed=self.signed, stochastic=self.stochastic, dequantize=self.dequantize,
                                      inplace=self.inplace, magnitude=self.magnitude)
        self.gquantize_f = gQuantize

        # buffer of quantize basis
        self.register_buffer('run_mean', torch.zeros(num_planes))
        self.register_buffer('run_scale', torch.zeros(num_planes))
        self.register_buffer('first_mean', torch.zeros(num_planes))
        self.register_buffer('first_scale', torch.zeros(num_planes))
        #self.register_buffer('last3_mean', torch.zeros(num_planes))
        #self.register_buffer('last3_scale', torch.zeros(num_planes))
        #self.register_buffer('last2_mean', torch.zeros(num_planes))
        #self.register_buffer('last2_scale', torch.zeros(num_planes))
        #self.register_buffer('last1_mean', torch.zeros(num_planes))
        #self.register_buffer('last1_scale', torch.zeros(num_planes))
        #self.register_buffer('real_mean', torch.zeros(num_planes))
        #self.register_buffer('real_scale', torch.zeros(num_planes))
        self.register_buffer('number', torch.zeros(1))
        self.register_buffer('first', torch.ones(1))

    def reset_params(self):
        if self.weight is not None:
            self.weight.data.uniform_()
        if self.bias is not None:
            self.bias.data.zero_()

    def forward(self, Qinput):

        if self.quantize_first:
            print("zkdebug a quantize before batchnorm")
            Qinput = self.aquantize_m(Qinput)

        if Qinput.dim() == 2:
            Qinput = Qinput.unsqueeze(-1,).unsqueeze(-1)

        if self.training:
            B,C,H,W = Qinput.shape
            y = Qinput.transpose(0, 1).contiguous()  # C x B x H x W
            y = y.view(C, self.num_chunks, (B * H * W) // self.num_chunks)
            mean_max = y.max(-1)[0].mean(-1)    # calculate max shape C by mean of maxs of chunks
            mean_min = y.min(-1)[0].mean(-1)    # calculate min shape C by mean of mins of chunks
            max_max  = y.max(-1)[0].max(-1)[0]  # calculate max shape C by mean of maxs of chunks
            min_min  = y.min(-1)[0].min(-1)[0]  # calculate min shape C by mean of mins of chunks
            mean = y.view(C, -1).mean(-1)       # C
            std  = y.view(C, -1).var(-1)**(0.5)
            scale_fix = (0.5*0.35) * (1+(math.pi*math.log(4))**0.5) / ((2*math.log(y.size(-1)))**0.5)
            scale  = ((mean_max - mean_min) * scale_fix)
            uscale = (max_max  - min_min)  * scale_fix

            with torch.no_grad():
                if self.quantize:
                    # this first register should not be update when float
                    self.first_mean.add_(self.run_mean.mul(self.first))
                    self.first_scale.add_(self.run_scale.mul(self.first))
                    self.first.mul_(0)
                if   self.rtbasis == 'this' or self.quantize == False:
                    used_mean  = mean
                    used_scale = scale
                elif self.rtbasis == 'run':
                    used_mean  = self.run_mean
                    used_scale = self.run_scale
                elif self.rtbasis == 'stable':
                    used_mean  = self.first_mean
                    used_scale = self.first_scale
                    '''
                elif self.rtbasis == 'last2_mean':
                    used_mean  = self.last2_mean.mul(0.5).add(self.last1_mean.mul(0.5))
                    used_scale = self.last2_scale.mul(0.5).add(self.last1_scale.mul(0.5))
                elif self.rtbasis == 'last2_pre':
                    used_mean  = self.last2_mean.mul(0.25).add(self.last1_mean.mul(0.75))
                    used_scale = self.last2_scale.mul(0.25).add(self.last1_scale.mul(0.75))
                elif self.rtbasis == 'last3_mean':
                    used_mean  = (self.last3_mean.add(self.last2_mean).add(self.last1_mean)).div(3)
                    used_scale = (self.last3_scale.add(self.last2_scale).add(self.last1_scale)).div(3)
                    '''
                else:
                    exit("ERROR rtrain using real")
                    used_mean  = self.real_mean
                    used_scale = self.real_scale

            #print("zkdebug mean.requires_grad:",used_mean.requires_grad) 
            #print("zkdebug scale.requires_grad:",used_scale.requires_grad) 
            assert(mean.requires_grad == True)
            assert(scale.requires_grad == True)
            if not self.rtbasis == 'this' and self.quantize == True:
                assert(used_mean.requires_grad == False)
                assert(used_scale.requires_grad == False)
            # we need the following format to remain the right backward flow of BN
            out = (Qinput-(mean.view(1,-1,1,1)-mean.view(1,-1,1,1).detach()+used_mean.view(1,-1,1,1).detach())) \
                 /(used_scale.view(1,-1,1,1) + self.eps)

            with torch.no_grad():
                self.run_mean.mul_(1-self.momentum).add_(mean.mul(self.momentum))
                self.run_scale.mul_(1-self.momentum).add_(scale.mul(self.momentum))
                #self.last3_mean.copy_(self.last2_mean)
                #self.last3_scale.copy_(self.last2_scale)
                #self.last2_mean.copy_(self.last1_mean)
                #self.last2_scale.copy_(self.last1_scale)
                #self.last1_mean.copy_(mean)
                #self.last1_scale.copy_(scale)
                #self.number = self.number.add_(1)

        else: # test
            if   self.rvbasis == 'run' or self.quantize == False:
                used_mean  = self.run_mean
                used_scale = self.run_scale
            elif self.rtbasis == 'stable':
                used_mean  = self.first_mean
                used_scale = self.first_scale
                '''
            elif self.rvbasis == 'last2_mean':
                used_mean  = (self.last2_mean.add(self.last1_mean)).div_(2)
                used_scale = (self.last2_scale.add(self.last1_scale)).div_(2)
            elif self.rvbasis == 'last2_pre':
                used_mean  = self.last2_mean.mul(0.25).add(self.last1_mean.mul(0.75))
                used_scale = self.last2_scale.mul(0.25).add(self.last1_scale.mul(0.75))
            elif self.rtbasis == 'last3_mean':
                used_mean  = (self.last3_mean.add(self.last2_mean).add(self.last1_mean)).div(3)
                used_scale = (self.last3_scale.add(self.last2_scale).add(self.last1_scale)).div(3)
                '''
            else:
                exit("ERROR rtest using real")
                used_mean  = self.real_mean
                used_scale = self.real_scale

            out = (Qinput - used_mean.view(1,-1,1,1)) \
                 /(used_scale.view(1, -1, 1, 1) + self.eps)

        if self.weight is not None:
            Qweight = self.weight
            #if self.quantize and self.rtbasis=='stable':
            #    Qweight = Qweight.detach()
            #Qweight = self.quantize_w(self.weight, bw=self.bw_w)
            out = out * Qweight.view(1, -1, 1, 1)
        if self.bias is not None:
            Qbias = self.bias
            #if self.quantize and self.rtbasis=='stable':
            #    Qbias = Qbias.detach()
            #Qbias = self.quantize_w(self.bias, bw=self.bw_w+self.bw_a)
            out = out + Qbias.view(1, -1, 1, 1)

        #if not self.quantize_first:
        #    out = self.quantize_a(out)

        #if self.bw_g is not None and self.quantize:
        #    out = self.gquantize_f(out, bw=self.bw_g, dequantize=self.dequantize, 
        #                           signed=self.signed, stochastic=True, 
        #                           inplace=self.inplace, magnitude=None)

        if out.size(3) == 1 and out.size(2) == 1:
            out = out.squeeze(-1).squeeze(-1)

        return out

