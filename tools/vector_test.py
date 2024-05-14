import torch

def gen_vector_mean_test_result():
    input = torch.tensor([0.44371, 0.20253, 0.73232, 0.40378, 0.93348, 0.72756, 0.63388, 0.5251, 0.23973, 0.52233])
    output = torch.mean(input)

    print(output)

if __name__ == "__main__":
    gen_vector_mean_test_result()