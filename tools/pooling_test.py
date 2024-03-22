import torch
from utils import print_cpp_vector


def _get_pool_input():
    return torch.tensor([
        0.573, 0.619, 0.732, 0.055,
        0.243, 0.316, 0.573, 0.619,
        0.712, 0.055, 0.243, 0.316,
        0.573, 0.619, 0.742, 0.055,
        0.473, 0.919, 0.107, 0.073,
        0.073, 0.362, 0.973, 0.059,
        0.473, 0.455, 0.283, 0.416,
        0.532, 0.819, 0.732, 0.850
    ]).reshape(1, 2, 4, 4)

def gen_max_pool_test_result():
    input = _get_pool_input()

    output = torch.nn.MaxPool2d(kernel_size=2, stride=2)(input)
    output = torch.flatten(output)

    print_cpp_vector(output)


def gen_avg_pool_test_result():

    input = _get_pool_input()

    output = torch.nn.AvgPool2d(kernel_size=2, stride=2)(input)
    output = torch.flatten(output)

    print_cpp_vector(output)


if __name__ == "__main__":
    print("Generating test results...")
    print("Max pool test:")
    gen_max_pool_test_result()
    print("Avg pool test:")
    gen_avg_pool_test_result()
