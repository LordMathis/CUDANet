import torch

from utils import print_cpp_vector


def gen_batch_norm_test_result(input):

    batch_norm = torch.nn.BatchNorm2d(2, track_running_stats=True)

    weights = torch.Tensor([0.63508, 0.64903])
    biases = torch.Tensor([0.25079, 0.66841])

    batch_norm.weight = torch.nn.Parameter(weights)
    batch_norm.bias = torch.nn.Parameter(biases)

    batch_norm.running_mean = torch.Tensor([0.5, 0.5])
    batch_norm.running_var = torch.Tensor([1.0, 1.0])

    batch_norm.eval()

    output = batch_norm(input)

    print_cpp_vector(output.flatten())


if __name__ == "__main__":

    print("Generating test results...")
    print("Batch norm test:")

    # fmt: off
    input = torch.Tensor([
        # Channel 0
        0.38899, 0.80478, 0.48836, 0.97381,
        0.57508, 0.60835, 0.65467, 0.00168,
        0.65869, 0.74235, 0.17928, 0.70349,
        0.15524, 0.38664, 0.23411, 0.7137,
        # Channel 1
        0.32473, 0.15698, 0.314, 0.60888,
        0.80268, 0.99766, 0.93694, 0.89237,
        0.13449, 0.27367, 0.53036, 0.18962,
        0.57672, 0.48364, 0.10863, 0.0571
    ]).reshape(1, 2, 4, 4)
    # fmt: on

    gen_batch_norm_test_result(input)

    print("Batch norm test non square input:")

    # fmt: off
    input = torch.Tensor([
        0.38899, 0.80478, 0.48836, 0.97381, 0.21567, 0.92312,
        0.57508, 0.60835, 0.65467, 0.00168, 0.31567, 0.71345,
        0.65869, 0.74235, 0.17928, 0.70349, 0.12856, 0.95645,
        0.15524, 0.38664, 0.23411, 0.7137, 0.26789, 0.83412,
        0.32473, 0.15698, 0.314, 0.60888, 0.23145, 0.78945,
        0.80268, 0.99766, 0.93694, 0.89237, 0.61234, 0.92314,
        0.13449, 0.27367, 0.53036, 0.18962, 0.45623, 0.14523,
        0.57672, 0.48364, 0.10863, 0.0571, 0.78934, 0.67545
    ]).reshape(1, 2, 4, 6)
    # fmt: on

    gen_batch_norm_test_result(input)
