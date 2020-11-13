import numpy as np
import torch
import copy
from torch import nn

class myMaxPool2d(nn.Module):
    def __init__(self, kernel_size, stride):
        super(myMaxPool2d, self).__init__()
        self.kernel_size = kernel_size
        self.stride = stride

    def forward(self, x):

        kRows, kCols = self.kernel_size
        outRows = int((x.shape[2] - kRows)/self.stride[0] + 1)
        outCols = int((x.shape[3] - kCols)/self.stride[1] + 1)

        x_unf = torch.nn.functional.unfold(x, kernel_size=self.kernel_size, stride=self.stride)
        x_view = x_unf.view(x_unf.shape[0], x_unf.shape[1]//(kRows*kCols), (kRows*kCols), -1)
        x_max = x_view.argmax(dim=2, keepdims=True)
        max_vals = torch.gather(x_view, 2, x_max)
        output = max_vals.view(max_vals.shape[0], max_vals.shape[1], outRows, outCols)

        return output

    ##############################################################
#
#  If your implementation is correct, your function should pass all the following tests
#  Please do not modify this cell. It is for evaluation only. 
#  If you want use it to debug your code, please make a copy of it and modify the copied version
#
##############################################################

def maxpool_evaluate(kernel_size, stride, input, label):
    a = myMaxPool2d(kernel_size=kernel_size, stride=stride)
    b = nn.MaxPool2d(kernel_size=kernel_size, stride=stride)
    
    input_a = copy.deepcopy(input)
    label_a = copy.deepcopy(label)
    input_b = copy.deepcopy(input)
    label_b = copy.deepcopy(label)
    
    ref_b = b(input_b)
    ref_a = a(input_a)
    # evaluation forward pass
    assert torch.equal(ref_a, ref_b), "model outputs does not match.\na:\n{}\nb:\n{}".format(ref_a, ref_b)
    
    # evaluation backward pass
    torch.sum(ref_a - label_a).backward()
    torch.sum(ref_b - label_b).backward()
    assert torch.equal(input_a.grad, input_b.grad), "gradients does not match.\na:\n{}\nb:\n{}".format(input_a, input_b)

if __name__ == '__main__':
    torch.manual_seed(42)
    print("Evaluate non-overlapped case")
    kernel_size=[2,2]
    stride=[2,2]
    #input = torch.randn(128,16,28,28, requires_grad=True)
    #label = torch.randn(128,16,14,14)
    input = torch.randn(3,2,3,3, requires_grad=True)
    label = torch.randn(3,2,2,2)
    maxpool_evaluate(kernel_size, stride, input, label)

    print("Evaluate overlapped case")
    kernel_size=[2,2]
    stride=[1,1]
    input = torch.randn(3,2,3,3, requires_grad=True)
    label = torch.randn(3,2,1,1)
    maxpool_evaluate(kernel_size, stride, input, label)