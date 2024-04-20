import torch
import struct


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


        for name, param in model.named_parameters():
            if 'weight' not in name and 'bias' not in name:
                continue

            tensor_values = param.flatten().tolist()
            tensor_bytes = struct.pack('f' * len(tensor_values), *tensor_values)

            tensor_size = param.numel()

            header += f"{name},{tensor_size},{offset}\n"
            
            offset += len(tensor_bytes)

            f.write(tensor_bytes)

        f.seek(0)
        f.write(struct.pack('q', len(header)))            
        f.write(header.encode('utf-8'))

def print_model_parameters(model: torch.nn.Module):
    for name, param in model.named_parameters():
        print(name, param.numel())