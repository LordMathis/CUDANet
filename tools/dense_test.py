import torch
from utils import print_cpp_vector

def gen_dense_softmax_test():

    input = torch.tensor([
        0.1, 0.2, 0.3, 0.4, 0.5
    ])

    weights = torch.tensor([
        0.5, 0.1, 0.1, 0.4, 0.2,
        0.4, 0.3, 0.9, 0.0, 0.8,
        0.8, 0.4, 0.6, 0.2, 0.0,
        0.1, 0.7, 0.3, 1.0, 0.1
    ]).reshape(4, 5)

    biases = torch.tensor([
        0.1, 0.2, 0.3, 0.4
    ])

    dense = torch.nn.Linear(5, 4)
    dense.weight = torch.nn.Parameter(weights)
    dense.bias = torch.nn.Parameter(biases)

    output = dense(input)
    print_cpp_vector(output)

    # Manual softmax
    softmax_exp = torch.exp(output)
    print(softmax_exp)

    softmax_sum = torch.sum(softmax_exp, dim=0)
    print(softmax_sum)

    souftmax_out = softmax_exp / softmax_sum
    print(souftmax_out)


    softmax = torch.nn.Softmax(dim=0)(output)
    print_cpp_vector(softmax)


if __name__ == "__main__":
    gen_dense_softmax_test()