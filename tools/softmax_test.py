import torch
from utils import print_cpp_vector


def gen_softmax1_test_result():
    input = torch.tensor([
        0.573, 0.619, 0.732, 0.055, 0.243
    ])

    output = torch.nn.Softmax(dim=0)(input)
    print_cpp_vector(output)


def gen_softmax2_test_result():
    input = torch.tensor([
        22.496, 36.9006, 30.9904, 28.4213, 26.4541, 31.7887
    ])

    output = torch.nn.Softmax(dim=0)(input)
    print_cpp_vector(output)


def gen_softmax_exp():
    input = torch.tensor([
        22.496, 36.9006, 30.9904, 28.4213, 26.4541, 31.7887
    ])

    output = torch.exp(input)
    print_cpp_vector(output)


if __name__ == "__main__":
    print("Generating test results...")
    print("Softmax 1 test:")
    gen_softmax1_test_result()
    print("Softmax 2 test:")
    gen_softmax2_test_result()
    print("Softmax exp test:")
    gen_softmax_exp()