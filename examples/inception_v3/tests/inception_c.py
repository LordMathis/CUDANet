import sys

import torch
from torchvision.models.inception import InceptionC

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
    inception_c = InceptionC(3, 64)
    inception_c.apply(init_weights)

    # branch1x1
    print_cpp_vector(torch.flatten(inception_c.branch1x1.conv.weight), "branch1x1_conv_weights")
    print_cpp_vector(torch.flatten(inception_c.branch1x1.bn.weight), "branch1x1_bn_weights")
    print_cpp_vector(torch.flatten(inception_c.branch1x1.bn.bias), "branch1x1_bn_bias")

    # branch7x7
    print_cpp_vector(torch.flatten(inception_c.branch7x7_1.conv.weight), "branch7x7_1_conv_weights")
    print_cpp_vector(torch.flatten(inception_c.branch7x7_1.bn.weight), "branch7x7_1_bn_weights")
    print_cpp_vector(torch.flatten(inception_c.branch7x7_1.bn.bias), "branch7x7_1_bn_bias")

    print_cpp_vector(torch.flatten(inception_c.branch7x7_2.conv.weight), "branch7x7_2_conv_weights")
    print_cpp_vector(torch.flatten(inception_c.branch7x7_2.bn.weight), "branch7x7_2_bn_weights")
    print_cpp_vector(torch.flatten(inception_c.branch7x7_2.bn.bias), "branch7x7_2_bn_bias")

    print_cpp_vector(torch.flatten(inception_c.branch7x7_3.conv.weight), "branch7x7_3_conv_weights")
    print_cpp_vector(torch.flatten(inception_c.branch7x7_3.bn.weight), "branch7x7_3_bn_weights")
    print_cpp_vector(torch.flatten(inception_c.branch7x7_3.bn.bias), "branch7x7_3_bn_bias")

    # branch7x7dbl
    print_cpp_vector(torch.flatten(inception_c.branch7x7dbl_1.conv.weight), "branch7x7dbl_1_conv_weights")
    print_cpp_vector(torch.flatten(inception_c.branch7x7dbl_1.bn.weight), "branch7x7dbl_1_bn_weights")
    print_cpp_vector(torch.flatten(inception_c.branch7x7dbl_1.bn.bias), "branch7x7dbl_1_bn_bias")

    print_cpp_vector(torch.flatten(inception_c.branch7x7dbl_2.conv.weight), "branch7x7dbl_2_conv_weights")
    print_cpp_vector(torch.flatten(inception_c.branch7x7dbl_2.bn.weight), "branch7x7dbl_2_bn_weights")
    print_cpp_vector(torch.flatten(inception_c.branch7x7dbl_2.bn.bias), "branch7x7dbl_2_bn_bias")

    print_cpp_vector(torch.flatten(inception_c.branch7x7dbl_3.conv.weight), "branch7x7dbl_3_conv_weights")
    print_cpp_vector(torch.flatten(inception_c.branch7x7dbl_3.bn.weight), "branch7x7dbl_3_bn_weights")
    print_cpp_vector(torch.flatten(inception_c.branch7x7dbl_3.bn.bias), "branch7x7dbl_3_bn_bias")

    print_cpp_vector(torch.flatten(inception_c.branch7x7dbl_4.conv.weight), "branch7x7dbl_4_conv_weights")
    print_cpp_vector(torch.flatten(inception_c.branch7x7dbl_4.bn.weight), "branch7x7dbl_4_bn_weights")
    print_cpp_vector(torch.flatten(inception_c.branch7x7dbl_4.bn.bias), "branch7x7dbl_4_bn_bias")

    print_cpp_vector(torch.flatten(inception_c.branch7x7dbl_5.conv.weight), "branch7x7dbl_5_conv_weights")
    print_cpp_vector(torch.flatten(inception_c.branch7x7dbl_5.bn.weight), "branch7x7dbl_5_bn_weights")
    print_cpp_vector(torch.flatten(inception_c.branch7x7dbl_5.bn.bias), "branch7x7dbl_5_bn_bias")

    # branch_pool
    print_cpp_vector(torch.flatten(inception_c.branch_pool.conv.weight), "branchPool_2_conv_weights")
    print_cpp_vector(torch.flatten(inception_c.branch_pool.bn.weight), "branchPool_2_bn_weights")
    print_cpp_vector(torch.flatten(inception_c.branch_pool.bn.bias), "branchPool_2_bn_bias")

    input_shape = (1, 3, 8, 8)
    input = torch.randn(input_shape)
    print_cpp_vector(torch.flatten(input), "input")

    output = inception_c(input)
    output = torch.flatten(output)
    print_cpp_vector(output, "expected")