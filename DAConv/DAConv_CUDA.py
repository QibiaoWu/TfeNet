import math
import torch
import numpy as np
import torch.nn as nn
import torch.nn.init as init
from torch.autograd import Function
from torch.autograd.function import once_differentiable
from torch.nn.modules.utils import _pair,_triple
import DACONV_CUDA
import os

class DAConvFunction(Function):
    @staticmethod
    def forward(ctx, input, offset, weight, bias=None, stride=1, padding=0, dilation=1,
                groups=1, deformable_groups=1 , in_step=1, axis=0,scale=0.1):
        ctx.stride = _triple(stride)
        ctx.padding = _triple(padding)
        ctx.dilation = _triple(dilation)
        ctx.groups = groups
        ctx.deformable_groups = deformable_groups
        ctx.in_step = in_step
        ctx.with_bias = bias is not None
        ctx.axis=axis
        ctx.scale=scale
        if not ctx.with_bias:
            bias = input.new_empty(0)  # fake tensor
        if not input.is_cuda:
            raise NotImplementedError
        if weight.requires_grad or offset.requires_grad or input.requires_grad:
            ctx.save_for_backward(input, offset, weight, bias)
        output = input.new_empty(DAConvFunction._infer_shape(ctx, input, weight))
    
        DACONV_CUDA.DAConv_forward_cuda(
            input, weight, bias, offset, output,
            weight.shape[2],weight.shape[3],weight.shape[4],
            ctx.stride[0], ctx.stride[1],ctx.stride[2],
            ctx.padding[0], ctx.padding[1],ctx.padding[2],
            ctx.dilation[0],ctx.dilation[1],ctx.dilation[2],
            ctx.groups, ctx.deformable_groups,ctx.in_step, ctx.with_bias,axis)
        '''
        int DAConv_forward_cuda(
                at::Tensor input, at::Tensor weight,at::Tensor bias,
                at::Tensor offset, at::Tensor output,
                const int kernel_h,const int kernel_w,const int kernel_l,
                const int stride_h,const int stride_w,const int stride_l,
                const int pad_h,const int pad_w,const int pad_l,
                const int dilation_h,const int dilation_w,const int dilation_l,
                const int group,const int deformable_group, const int in_step,const bool with_bias);
        '''
        return output

    @staticmethod
    # @once_differentiable
    def backward(ctx, grad_output):
        grad_output=grad_output.contiguous()
        if not grad_output.is_cuda:
            raise NotImplementedError
        input, offset, weight, bias = ctx.saved_tensors
        grad_input = torch.zeros_like(input)
        grad_offset = torch.zeros_like(offset)
        grad_weight = torch.zeros_like(weight)
        grad_bias = torch.zeros_like(bias)
        DACONV_CUDA.DAConv_backward_cuda(
            input, weight, bias, offset,
            grad_input, grad_weight,grad_bias,grad_offset, grad_output,
            weight.shape[2], weight.shape[3],weight.shape[4],
            ctx.stride[0], ctx.stride[1], ctx.stride[2],
            ctx.padding[0], ctx.padding[1], ctx.padding[2],
            ctx.dilation[0], ctx.dilation[1],ctx.dilation[2],
            ctx.groups, ctx.deformable_groups,ctx.in_step,ctx.with_bias,ctx.axis)

        '''
        int DAConv_backward_cuda(
            at::Tensor input, at::Tensor weight, at::Tensor bias,at::Tensor offset,
            at::Tensor grad_input, at::Tensor grad_weight, at::Tensor grad_bias,
            at::Tensor grad_offset, at::Tensor grad_output,
            const int kernel_h,const int kernel_w,const int kernel_l,
            const int stride_h,const int stride_w,const int stride_l,
            const int pad_h,const int pad_w,const int pad_l,
            const int dilation_h,const int dilation_w,const int dilation_l,
            const int group, int deformable_group,const int in_step,const bool with_bias) ;
        '''
        if not ctx.with_bias:
            grad_bias = None
        else:
            grad_bias = torch.tensor(ctx.scale)*grad_bias
        grad_input = torch.tensor(ctx.scale)*grad_input
        grad_offset = torch.tensor(ctx.scale)*grad_offset
        grad_weight = torch.tensor(ctx.scale)*grad_weight
        return (grad_input, grad_offset, grad_weight, grad_bias, None, None, None, None, None,None,None,None)


    @staticmethod
    def _infer_shape(ctx, input, weight):
        n = input.size(0)
        channels_out = weight.size(0)
        height, width,length = input.shape[2:5]
        kernel_h, kernel_w ,kernel_l= weight.shape[2:5]
        height_out = (height + 2 * ctx.padding[0] - (ctx.dilation[0] * (kernel_h - 1) + 1)) // ctx.stride[0] + 1
        width_out = (width + 2 * ctx.padding[1] - (ctx.dilation[1] * (kernel_w - 1) + 1)) // ctx.stride[1] + 1
        length_out = (length + 2 * ctx.padding[2] - (ctx.dilation[2] * (kernel_l - 1) + 1)) // ctx.stride[2] + 1
        return n, channels_out, height_out, width_out, length_out

DAConv_Function = DAConvFunction.apply

class DAConv(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=9, stride=1, padding=4, dilation=1,
                 groups=1, deformable_groups=1, bias=False,in_step=1, axis=0,scale=0.1):
        super(DAConv, self).__init__()
        assert in_channels % groups == 0, \
            'in_channels {} cannot be divisible by groups {}'.format(
                in_channels, groups)
        assert out_channels % groups == 0, \
            'out_channels {} cannot be divisible by groups {}'.format(
                out_channels, groups)

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.axis = axis
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.dilation = _triple(dilation)
        self.groups = groups
        self.deformable_groups = deformable_groups
        self.in_step=in_step
        self.scale = scale
        
        if self.axis == 0:
            self.weight = nn.Parameter(
                torch.Tensor(out_channels, in_channels // self.groups, *(self.kernel_size,1,1)))
        elif self.axis == 1:
            self.weight = nn.Parameter(
                torch.Tensor(out_channels, in_channels // self.groups, *(1,self.kernel_size,1)))
        else:
            self.weight = nn.Parameter(
                torch.Tensor(out_channels, in_channels // self.groups, *(1,1,self.kernel_size)))

        self.with_bias=bias
        if self.with_bias:
            self.bias = nn.Parameter(torch.Tensor(out_channels))
        else:
            self.bias=None

        self.reset_parameters()

    def reset_parameters(self):
        n = self.in_channels
        for k in (1,1,self.kernel_size):
            n *= k
        stdv = 1. / math.sqrt(n)
        self.weight.data.uniform_(-stdv, stdv)
        if self.with_bias:
            self.bias.data.fill_(0)

    def forward(self, x, offset):
        if self.axis == 0:
            return DAConv_Function(x, offset, self.weight, self.bias, (self.stride,1,1), (self.padding,0,0), self.dilation,
                            self.groups, self.deformable_groups,self.in_step, self.axis,self.scale)
        elif self.axis == 1:
            return DAConv_Function(x, offset, self.weight, self.bias, (1,self.stride,1), (0,self.padding,0), self.dilation,
                            self.groups, self.deformable_groups,self.in_step, self.axis,self.scale)
        else:
            return DAConv_Function(x, offset, self.weight, self.bias, (1,1,self.stride), (0,0,self.padding), self.dilation,
                        self.groups, self.deformable_groups,self.in_step, self.axis,self.scale)     

class DAConvPack(DAConv):
    def __init__(self, *args, **kwargs):
        super(DAConvPack, self).__init__(*args, **kwargs)
        self.conv_channel = 4
        self.conv_angel = nn.Conv3d(
            self.in_channels,
            self.conv_channel,
            kernel_size=(3,3,3), stride=(1,1,1), 
            padding=(1,1,1),bias=True)
    
        self.In = nn.InstanceNorm3d(self.conv_channel)
        self.init_offset()
   
    def init_offset(self):
        self.conv_angel.weight.data.zero_()
        self.conv_angel.bias.data.zero_()


    def forward(self, x):
        offset = self.conv_angel(x)
        offset = self.In(offset)
        offset = (np.pi/4)*torch.tanh(offset)
       
        if self.axis == 0:
            snake_conv = DAConv_Function(x, offset, self.weight, self.bias, (self.stride,1,1), (self.padding,0,0), self.dilation,
                            self.groups, self.deformable_groups,self.in_step, self.axis,self.scale)
        elif self.axis == 1:
            snake_conv = DAConv_Function(x, offset, self.weight, self.bias, (1,self.stride,1), (0,self.padding,0), self.dilation,
                            self.groups, self.deformable_groups,self.in_step, self.axis,self.scale)
        else:
            snake_conv = DAConv_Function(x, offset, self.weight, self.bias, (1,1,self.stride), (0,0,self.padding), self.dilation,
                        self.groups, self.deformable_groups,self.in_step, self.axis,self.scale)
       
        return snake_conv


