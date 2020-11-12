import numpy as np
import torch
import copy

from torch import nn

class myConv2d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding):
        super(myConv2d, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride 
        self.padding = padding
        # kernel weight
        self.weight = torch.randn(out_channels, in_channels, *kernel_size) 
        # bias per out_channel
        self.bias = torch.randn(out_channels)

        self.weight = nn.Parameter(self.weight)
        self.bias = nn.Parameter(self.bias)

    def forward(self, x):
        temp = []
        x = self.zero_padding(x, self.padding)
        for kernel, bias in zip(self.weight, self.bias):
            temp.append(self.kernel_op(x, kernel, bias, self.kernel_size, self.stride))
        return torch.cat(temp, dim=1)

    def zero_padding(self, x, padding):
        padding = torch.nn.ConstantPad2d((padding, padding, padding, padding), value=0)
        return padding(x)

    def kernel_op(self, x, kernel, bias, kernel_size, stride):
        kRows, kCols = kernel_size
        outRows = int((x.shape[2] - kRows)/stride[0] + 1)
        outCols = int((x.shape[3] - kCols)/stride[1] + 1)
        output = torch.zeros((x.shape[0], 1, outRows, outCols))
        for xi in range(x.shape[0]):
            for colOut, colIn in enumerate(range(0, x.shape[3]-kCols+1, stride[1])):
                for rowOut, rowIn in enumerate(range(0, x.shape[2]-kRows+1, stride[0])):
                    output[xi, 0, rowOut, colOut] = torch.dot(x[xi, : , rowIn:rowIn+kRows, colIn:colIn+kCols].flatten(), kernel.flatten()) + bias

        return output


##############################################################
#
#  If your implementation is correct, your function should pass all the following tests
#  Please do not modify this cell. It is for evaluation only. 
#  If you want use it to debug your code, please make a copy of it and modify the copied version
#
##############################################################

# TODO: check all possible cases that one may makes mistakes

def conv_assert(kernel_size, stride, padding):
    in_channels = 3
    out_channels = 3
    x = torch.randn(2,in_channels,10,10)
    # TODO: function for calculate output size
    y = torch.randn(2,out_channels, 
                    (10+padding*2-kernel_size[0])//stride[0] + 1,
                    (10+padding*2-kernel_size[1])//stride[1] + 1, requires_grad=True)
    y2 = copy.deepcopy(y)
    a = nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size, stride=stride,padding=padding)
    b = myConv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size, stride=stride,padding=padding)
    
    b.weight = copy.deepcopy(a.weight)
    b.bias   = copy.deepcopy(a.bias)
    
    assert torch.equal(b.weight, a.weight)
    assert torch.equal(b.bias, a.bias)
    
    py = a(x)
    #print(py[:,0])
    my = b(x)
    assert torch.allclose(my, py, atol=1e-06), "diff: \n{}".format(my-py)
    dpy = torch.sum(py - y)
    dmy = torch.sum(my - y2)
    dpy.backward()
    dmy.backward()

    print(a.weight.grad.shape) 
    print(b.weight.grad.shape) 
    #print(a.weight)
    #print(b.weight)
    assert np.allclose(b.weight.grad.numpy(), a.weight.grad.numpy())
    assert torch.allclose(b.weight.grad, a.weight.grad)
    assert torch.allclose(b.bias.grad, a.bias.grad)

# common case
torch.random.manual_seed(10086)
print("common case")
kernel_size = [2,2]
stride = [1,1]
padding= 0
conv_assert(kernel_size, stride, padding)

# padding
print("padding")
kernel_size = [3,3]
stride = [1,1]
padding= 1
conv_assert(kernel_size, stride, padding)

# large padding
print("large padding")
kernel_size = [5,5]
stride = [1,1]
padding= 0
conv_assert(kernel_size, stride, padding)

# large stride 
print("large stride")
kernel_size = [3,3]
stride = [3,3]
padding= 1
conv_assert(kernel_size, stride, padding)

# uneven stride 
print("uneven stride")
kernel_size = [3,3]
stride = [2,3]
padding= 1
conv_assert(kernel_size, stride, padding)