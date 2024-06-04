import sys

import torch
from torchvision.models.inception import InceptionB

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
    inception_b = InceptionB(3)
    inception_b.apply(init_weights)

    # branch3x3
    print_cpp_vector(torch.flatten(inception_b.branch3x3.conv.weight), "branch3x3_conv_weights")
    print_cpp_vector(torch.flatten(inception_b.branch3x3.bn.weight), "branch3x3_bn_weights")
    print_cpp_vector(torch.flatten(inception_b.branch3x3.bn.bias), "branch3x3_bn_bias")

    # branch3x3dbl
    print_cpp_vector(torch.flatten(inception_b.branch3x3dbl_1.conv.weight), "branch3x3dbl_1_conv_weights")
    print_cpp_vector(torch.flatten(inception_b.branch3x3dbl_1.bn.weight), "branch3x3dbl_1_bn_weights")
    print_cpp_vector(torch.flatten(inception_b.branch3x3dbl_1.bn.bias), "branch3x3dbl_1_bn_bias")

    print_cpp_vector(torch.flatten(inception_b.branch3x3dbl_2.conv.weight), "branch3x3dbl_2_conv_weights")
    print_cpp_vector(torch.flatten(inception_b.branch3x3dbl_2.bn.weight), "branch3x3dbl_2_bn_weights")
    print_cpp_vector(torch.flatten(inception_b.branch3x3dbl_2.bn.bias), "branch3x3dbl_2_bn_bias")

    print_cpp_vector(torch.flatten(inception_b.branch3x3dbl_3.conv.weight), "branch3x3dbl_3_conv_weights")
    print_cpp_vector(torch.flatten(inception_b.branch3x3dbl_3.bn.weight), "branch3x3dbl_3_bn_weights")
    print_cpp_vector(torch.flatten(inception_b.branch3x3dbl_3.bn.bias), "branch3x3dbl_3_bn_bias")

    input_shape = (1, 3, 8, 8)
    input = torch.randn(input_shape)
    print_cpp_vector(torch.flatten(input), "input")

    output = inception_b(input)
    output = torch.flatten(output)
    print_cpp_vector(output, "expected")