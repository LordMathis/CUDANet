import numpy as np
import utils
from sys import argv


def gen_random_vector(size):
    return np.random.rand(size)


if __name__ == "__main__":

    if len(argv) < 2:
        print("Usage: python gen_random_vector.py <size>")
        exit(1)

    vector = gen_random_vector(int(argv[1]))
    utils.print_cpp_vector(vector)
