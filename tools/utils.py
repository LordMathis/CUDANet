import torch
import struct

import numpy as np


def print_cpp_vector(vector, name="expected"):
    print("std::vector<float> " + name + " = {", end="")
    for i in range(len(vector)):
        if i != 0:
            print(", ", end="")
        print(str(round(vector[i].item(), 5)) + "f", end="")
    print("};")


def export_model_weights(model: torch.nn.Module, filename):
    with open(filename, 'wb') as f:

        header = ""
        offset = 0
        tensor_data = b""

        for name, param in model.named_parameters():
            if 'weight' not in name and 'bias' not in name:
                continue

            tensor_bytes = param.type(torch.float32).detach().numpy().tobytes()
            tensor_size = param.numel()

            header += f"{name},{tensor_size},{offset}\n"            
            offset += len(tensor_bytes)

            tensor_data += tensor_bytes

        f.seek(0)
        f.write(struct.pack('q', len(header)))           
        f.write(header.encode('utf-8'))
        f.write(tensor_data)

def print_model_parameters(model: torch.nn.Module):
    for name, param in model.named_parameters():
        print(name, param.numel())
