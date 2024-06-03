import sys

import torch
from torchvision.models.inception import InceptionA

sys.path.append("../../../tools")
from utils import print_cpp_vector

torch.manual_seed(0)

@torch.no_grad()
def init_weights(m):
    if isinstance(m, torch.nn.Conv2d):
        torch.nn.init.uniform_(m.weight)
    elif isinstance(m, torch.nn.BatchNorm2d):
        torch.nn.init.uniform_(m.weight)
        torch.nn.init.uniform_(m.bias)

with torch.no_grad():
    inception_a = InceptionA(3, 6)
    inception_a.apply(init_weights)

    # branch1x1
    print_cpp_vector(torch.flatten(inception_a.branch1x1.conv.weight), "branch1x1_conv_weights")
    print_cpp_vector(torch.flatten(inception_a.branch1x1.bn.weight), "branch1x1_bn_weights")
    print_cpp_vector(torch.flatten(inception_a.branch1x1.bn.bias), "branch1x1_bn_bias")

    # branch5x5
    print_cpp_vector(torch.flatten(inception_a.branch5x5_1.conv.weight), "branch5x5_1_conv_weights")
    print_cpp_vector(torch.flatten(inception_a.branch5x5_1.bn.weight), "branch5x5_1_bn_weights")
    print_cpp_vector(torch.flatten(inception_a.branch5x5_1.bn.bias), "branch5x5_1_bn_bias")

    print_cpp_vector(torch.flatten(inception_a.branch5x5_2.conv.weight), "branch5x5_2_conv_weights")
    print_cpp_vector(torch.flatten(inception_a.branch5x5_2.bn.weight), "branch5x5_2_bn_weights")
    print_cpp_vector(torch.flatten(inception_a.branch5x5_2.bn.bias), "branch5x5_2_bn_bias")

    # branch3x3dbl
    print_cpp_vector(torch.flatten(inception_a.branch3x3dbl_1.conv.weight), "branch3x3dbl_1_conv_weights")
    print_cpp_vector(torch.flatten(inception_a.branch3x3dbl_1.bn.weight), "branch3x3dbl_1_bn_weights")
    print_cpp_vector(torch.flatten(inception_a.branch3x3dbl_1.bn.bias), "branch3x3dbl_1_bn_bias")

    print_cpp_vector(torch.flatten(inception_a.branch3x3dbl_2.conv.weight), "branch3x3dbl_2_conv_weights")
    print_cpp_vector(torch.flatten(inception_a.branch3x3dbl_2.bn.weight), "branch3x3dbl_2_bn_weights")
    print_cpp_vector(torch.flatten(inception_a.branch3x3dbl_2.bn.bias), "branch3x3dbl_2_bn_bias")

    print_cpp_vector(torch.flatten(inception_a.branch3x3dbl_3.conv.weight), "branch3x3dbl_3_conv_weights")
    print_cpp_vector(torch.flatten(inception_a.branch3x3dbl_3.bn.weight), "branch3x3dbl_3_bn_weights")
    print_cpp_vector(torch.flatten(inception_a.branch3x3dbl_3.bn.bias), "branch3x3dbl_3_bn_bias")

    # branchPool
    print_cpp_vector(torch.flatten(inception_a.branch_pool.conv.weight), "branchPool_2_conv_weights")
    print_cpp_vector(torch.flatten(inception_a.branch_pool.bn.weight), "branchPool_2_bn_weights")
    print_cpp_vector(torch.flatten(inception_a.branch_pool.bn.bias), "branchPool_2_bn_bias")

    input_shape = (1, 3, 8, 8)
    input = torch.randn(input_shape)
    print_cpp_vector(torch.flatten(input), "input")

    output = inception_a(input)
    output = torch.flatten(output)
    print_cpp_vector(output)

