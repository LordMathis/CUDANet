import sys

import torch
from torchvision.models.inception import (
    BasicConv2d,
    InceptionA,
    InceptionB,
    InceptionC,
    InceptionD,
    InceptionE
)

sys.path.append("../../../tools")
from utils import print_cpp_vector, export_model_weights

torch.manual_seed(0)

output_size = 50

class InceptionBlockModel(torch.nn.Module):
    def __init__(self, inception_block: torch.nn.Module, linear_in: int, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.inception_block = inception_block
        self.fc = torch.nn.Linear(linear_in, output_size)

    def forward(self, x):
        x = self.inception_block(x)
        x = torch.flatten(x)
        x = self.fc(x)
        # x = torch.nn.functional.tanh(x)
        return x


@torch.no_grad()
def init_weights(m: torch.nn.Module):
    if isinstance(m, torch.nn.Conv2d):
        torch.nn.init.uniform_(m.weight, -1, 1)
    elif isinstance(m, torch.nn.BatchNorm2d) or isinstance(m, torch.nn.Linear):
        torch.nn.init.uniform_(m.weight, -1)
        torch.nn.init.uniform_(m.bias, 1)

    if isinstance(m, torch.nn.BatchNorm2d):
        # Initialize running_mean and running_var
        m.running_mean.uniform_(-1, 1)
        m.running_var.uniform_(0, 1)  # Variance should be positive


@torch.no_grad()
def generate_module_test_data(m: torch.nn.Module, name: str):

    print(name)
    
    input_shape = (1, 3, 4, 4)
    input = torch.randn(input_shape)
    print_cpp_vector(torch.flatten(input), "input")

    m.eval()
    inception_out = m(input)
    linear_in = torch.flatten(inception_out).size(0)

    inception_block = InceptionBlockModel(m, linear_in)
    inception_block.apply(init_weights)

    export_model_weights(inception_block, f"resources/{name}.bin")

    inception_block.eval()
    output = inception_block(input)
    print_cpp_vector(torch.flatten(output), "expected")

    print()


if __name__ == "__main__":
    # m = BasicConv2d(3, 32, kernel_size=3, stride=2, padding=0)
    # generate_module_test_data(m, "basic_conv2d")

    # m = InceptionA(3, 6)
    # generate_module_test_data(m, "inception_a")

    # m = InceptionB(3)
    # generate_module_test_data(m, "inception_b")

    # m = InceptionC(3, 64)
    # generate_module_test_data(m, "inception_c")

    # m = InceptionD(3)
    # generate_module_test_data(m, "inception_d")

    m = InceptionE(3)
    generate_module_test_data(m, "inception_e")



