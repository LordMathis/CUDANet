import torch
import struct

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