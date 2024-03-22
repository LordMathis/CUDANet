import torch
from utils import print_cpp_vector


def gen_softmax_test_result():
    input = torch.tensor([
        0.573, 0.619, 0.732, 0.055, 0.243
    ])

    output = torch.nn.Softmax(dim=0)(input)
    print_cpp_vector(output)


if __name__ == "__main__":
    print("Generating test results...")
    print("Softmax test:")
    gen_softmax_test_result()