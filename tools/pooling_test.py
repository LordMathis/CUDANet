import torch
from utils import print_cpp_vector


def _get_pool_input():
    # fmt: off
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
    # fmt: on


def _get_pool_input_non_square():
    # fmt: off
    return torch.Tensor([
        0.573, 0.619, 0.732, 0.055, 0.123, 0.234,
        0.243, 0.316, 0.573, 0.619, 0.456, 0.789,
        0.712, 0.055, 0.243, 0.316, 0.654, 0.987,
        0.573, 0.619, 0.742, 0.055, 0.321, 0.654,
        0.473, 0.919, 0.107, 0.073, 0.321, 0.654,
        0.073, 0.362, 0.973, 0.059, 0.654, 0.987,
        0.473, 0.455, 0.283, 0.416, 0.789, 0.123,
        0.532, 0.819, 0.732, 0.850, 0.987, 0.321
    ]).reshape(1, 2, 4, 6)
    # fmt: on


def gen_max_pool_test_result():
    input = _get_pool_input()

    output = torch.nn.MaxPool2d(kernel_size=2, stride=2)(input)
    output = torch.flatten(output)

    print_cpp_vector(output)


def gen_max_pool_non_square_input_test_result():
    input = _get_pool_input_non_square()

    output = torch.nn.MaxPool2d(kernel_size=2, stride=2)(input)
    output = torch.flatten(output)

    print_cpp_vector(output)


def gen_max_non_square_pool_test_result():
    input = _get_pool_input()

    output = torch.nn.MaxPool2d(kernel_size=(2, 3), stride=2)(input)
    output = torch.flatten(output)

    print_cpp_vector(output)


def gen_max_pool_non_square_stride_test_result():
    input = _get_pool_input()

    output = torch.nn.MaxPool2d(kernel_size=2, stride=(1, 2))(input)
    output = torch.flatten(output)

    print_cpp_vector(output)


def gen_max_pool_non_square_padding_test_result():
    input = _get_pool_input()

    output = torch.nn.MaxPool2d(kernel_size=2, stride=2, padding=(0, 1))(input)
    output = torch.flatten(output)

    print_cpp_vector(output)


def gen_avg_pool_test_result():

    input = _get_pool_input()

    output = torch.nn.AvgPool2d(kernel_size=2, stride=2)(input)
    output = torch.flatten(output)

    print_cpp_vector(output)


def gen_avg_pool_non_square_input_test_result():

    input = _get_pool_input_non_square()

    output = torch.nn.AvgPool2d(kernel_size=2, stride=2)(input)
    output = torch.flatten(output)

    print_cpp_vector(output)


def gen_avg_non_square_pool_test_result():

    input = _get_pool_input()

    output = torch.nn.AvgPool2d(kernel_size=(2, 3), stride=2)(input)
    output = torch.flatten(output)

    print_cpp_vector(output)


def gen_avg_pool_non_square_stride_test_result():

    input = _get_pool_input()

    output = torch.nn.AvgPool2d(kernel_size=2, stride=(1, 2))(input)
    output = torch.flatten(output)

    print_cpp_vector(output)


def gen_avg_pool_non_square_padding_test_result():

    input = _get_pool_input()

    output = torch.nn.AvgPool2d(kernel_size=2, stride=2, padding=(1, 0))(input)
    output = torch.flatten(output)

    print_cpp_vector(output)


if __name__ == "__main__":
    print("Generating test results...")
    print("Max pool test:")
    gen_max_pool_test_result()
    print("Max pool non square input test:")
    gen_max_pool_non_square_input_test_result()
    print("Max non square pool test:")
    gen_max_non_square_pool_test_result()
    print("Max pool non square stride test:")
    gen_max_pool_non_square_stride_test_result()
    print("Max pool non square padding test:")
    gen_max_pool_non_square_padding_test_result()

    print("--------------")

    print("Avg pool test:")
    gen_avg_pool_test_result()
    print("Avg pool non square input test:")
    gen_avg_pool_non_square_input_test_result()
    print("Avg non square pool test:")
    gen_avg_non_square_pool_test_result()
    print("Avg pool non square stride test:")
    gen_avg_pool_non_square_stride_test_result()
    print("Avg pool non square padding test:")
    gen_avg_pool_non_square_padding_test_result()
