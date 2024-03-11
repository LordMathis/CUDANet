import torch
import numpy as np

def conv2d(in_channels, out_channels, kernel_size, stride, padding, inputs, weights):

    conv2d = torch.nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size, stride=stride, padding=padding, bias=False)
    conv2d.weight = torch.nn.Parameter(weights)

    output = conv2d(inputs)

    # Print the output as cpp vector
    output = torch.flatten(output)
    return output

def print_cpp_vector(vector):
    print("std::vector<float> expected = {", end="")
    for i in range(len(vector)):
        if i != 0:
            print(", ", end="")
        print(str(round(vector[i].item(), 5)) + "f", end="")
    print("};")


def gen_padded_test_result():

    in_channels = 3
    out_channels = 2
    kernel_size = 3
    stride = 1
    padding = 1

    # Define input and kernel data as tensors
    inputs = torch.tensor([
        0.823, 0.217, 0.435, 0.981, 0.742,
        0.109, 0.518, 0.374, 0.681, 0.147,
        0.956, 0.729, 0.654, 0.087, 0.392,
        0.784, 0.921, 0.543, 0.231, 0.816,
        0.472, 0.614, 0.102, 0.987, 0.398,
        0.051, 0.756, 0.841, 0.293, 0.128,
        0.417, 0.632, 0.095, 0.184, 0.529,
        0.871, 0.958, 0.213, 0.347, 0.725,
        0.461, 0.012, 0.278, 0.195, 0.649,
        0.853, 0.707, 0.988, 0.988, 0.322,
        0.345, 0.123, 0.789, 0.123, 0.456,
        0.456, 0.789, 0.123, 0.345, 0.123,
        0.789, 0.123, 0.345, 0.123, 0.456,
        0.123, 0.345, 0.123, 0.789, 0.123,
        0.345, 0.123, 0.789, 0.123, 0.456
    ], dtype=torch.float).reshape(1, 3, 5, 5)

    weights = torch.tensor([
        0.128, 0.754, 0.987,
        0.321, 0.412, 0.635,
        0.298, 0.017, 0.845,
        0.514, 0.729, 0.952,
        0.684, 0.378, 0.159,
        0.823, 0.547, 0.216,
        0.983, 0.231, 0.456,
        0.178, 0.654, 0.821,
        0.345, 0.987, 0.123,
        0.789, 0.543, 0.210,
        0.012, 0.371, 0.638,
        0.456, 0.198, 0.907,
        0.101, 0.432, 0.759,
        0.234, 0.567, 0.890,
        0.543, 0.876, 0.219,
        0.345, 0.678, 0.011,
        0.678, 0.011, 0.345,
        0.011, 0.345, 0.678
    ], dtype=torch.float).reshape(2, 3, 3, 3)

    output = conv2d(in_channels, out_channels, kernel_size, stride, padding, inputs, weights)
    print_cpp_vector(output)

def gen_strided_test_result():

    in_channels = 2
    out_channels = 2
    kernel_size = 3
    stride = 2
    padding = 3
    
    input = torch.tensor([
        0.946, 0.879, 0.382, 0.542, 0.453,
        0.128, 0.860, 0.778, 0.049, 0.974,
        0.400, 0.874, 0.161, 0.271, 0.580,
        0.373, 0.078, 0.366, 0.396, 0.181,
        0.246, 0.112, 0.179, 0.979, 0.026,
        0.598, 0.458, 0.776, 0.213, 0.199,
        0.853, 0.170, 0.609, 0.269, 0.777,
        0.776, 0.694, 0.430, 0.238, 0.968,
        0.473, 0.303, 0.084, 0.785, 0.444,
        0.464, 0.413, 0.779, 0.298, 0.783
    ], dtype=torch.float).reshape(1, 2, 5, 5)
    weights = torch.tensor([
        0.744, 0.745, 0.641,
        0.164, 0.157, 0.127,
        0.732, 0.761, 0.601,
        0.475, 0.335, 0.499,
        0.833, 0.793, 0.176,
        0.822, 0.163, 0.175,
        0.918, 0.340, 0.497,
        0.233, 0.218, 0.847,
        0.931, 0.926, 0.199,
        0.510, 0.432, 0.567,
        0.236, 0.397, 0.739,
        0.939, 0.891, 0.006
    ], dtype=torch.float).reshape(2, 2, 3, 3)

    output = conv2d(in_channels, out_channels, kernel_size, stride, padding, input, weights)
    print_cpp_vector(output)


if __name__ == "__main__":
    print("Generating test results...")
    print("Padded convolution test:")
    gen_padded_test_result()
    print("Strided convolution test:")
    gen_strided_test_result()