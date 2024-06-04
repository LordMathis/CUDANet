import sys

import torch
from torchvision.models.inception import InceptionD

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
    inception_c = InceptionD(3)
    inception_c.apply(init_weights)

    # branch3x3
    print_cpp_vector(torch.flatten(inception_c.branch3x3_1.conv.weight), "branch3x3_1_conv_weights")
    print_cpp_vector(torch.flatten(inception_c.branch3x3_1.bn.weight), "branch3x3_1_bn_weights")
    print_cpp_vector(torch.flatten(inception_c.branch3x3_1.bn.bias), "branch3x3_1_bn_bias")

    print_cpp_vector(torch.flatten(inception_c.branch3x3_2.conv.weight), "branch3x3_2_conv_weights")
    print_cpp_vector(torch.flatten(inception_c.branch3x3_2.bn.weight), "branch3x3_2_bn_weights")
    print_cpp_vector(torch.flatten(inception_c.branch3x3_2.bn.bias), "branch3x3_2_bn_bias")

    # branch7x7x3
    print_cpp_vector(torch.flatten(inception_c.branch7x7x3_1.conv.weight), "branch7x7x3_1_conv_weights")
    print_cpp_vector(torch.flatten(inception_c.branch7x7x3_1.bn.weight), "branch7x7x3_1_bn_weights")
    print_cpp_vector(torch.flatten(inception_c.branch7x7x3_1.bn.bias), "branch7x7x3_1_bn_bias")

    print_cpp_vector(torch.flatten(inception_c.branch7x7x3_2.conv.weight), "branch7x7x3_2_conv_weights")
    print_cpp_vector(torch.flatten(inception_c.branch7x7x3_2.bn.weight), "branch7x7x3_2_bn_weights")
    print_cpp_vector(torch.flatten(inception_c.branch7x7x3_2.bn.bias), "branch7x7x3_2_bn_bias")

    print_cpp_vector(torch.flatten(inception_c.branch7x7x3_3.conv.weight), "branch7x7x3_3_conv_weights")
    print_cpp_vector(torch.flatten(inception_c.branch7x7x3_3.bn.weight), "branch7x7x3_3_bn_weights")
    print_cpp_vector(torch.flatten(inception_c.branch7x7x3_3.bn.bias), "branch7x7x3_3_bn_bias")

    print_cpp_vector(torch.flatten(inception_c.branch7x7x3_4.conv.weight), "branch7x7x3_4_conv_weights")
    print_cpp_vector(torch.flatten(inception_c.branch7x7x3_4.bn.weight), "branch7x7x3_4_bn_weights")
    print_cpp_vector(torch.flatten(inception_c.branch7x7x3_4.bn.bias), "branch7x7x3_4_bn_bias")

    input_shape = (1, 3, 8, 8)
    input = torch.randn(input_shape)
    print_cpp_vector(torch.flatten(input), "input")

    output = inception_c(input)
    output = torch.flatten(output)
    print_cpp_vector(output, "expected")