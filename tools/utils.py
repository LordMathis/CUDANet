import torch
import struct

from PIL import Image

from torchvision import transforms


def print_cpp_vector(vector, name="expected"):
    print("std::vector<float> " + name + " = {", end="")
    for i in range(len(vector)):
        if i != 0:
            print(", ", end="")
        print(str(round(vector[i].item(), 5)) + "f", end="")
    print("};")


def export_model_weights(model: torch.nn.Module, filename):
    with open(filename, "wb") as f:

        version = 1
        header = ""
        offset = 0
        tensor_data = b""

        for name, param in model.named_parameters():
            if "weight" not in name and "bias" not in name:
                continue

            tensor_bytes = param.type(torch.float32).detach().numpy().tobytes()
            tensor_size = param.numel()

            header += f"{name},{tensor_size},{offset}\n"
            offset += len(tensor_bytes)

            tensor_data += tensor_bytes

        # print(model.named_buffers)

        # Add buffers (for running_mean and running_var)
        for name, buf in model.named_buffers():
            if "running_mean" not in name and "running_var" not in name:
                continue
            
            tensor_bytes = buf.type(torch.float32).detach().numpy().tobytes()
            tensor_size = buf.numel()
            header += f"{name},{tensor_size},{offset}\n"
            offset += len(tensor_bytes)
            tensor_data += tensor_bytes

        f.seek(0)
        f.write(struct.pack("H", version))
        f.write(struct.pack("Q", len(header)))
        f.write(header.encode("utf-8"))
        f.write(tensor_data)


def print_model_parameters(model: torch.nn.Module):
    for name, param in model.named_parameters():
        print(name, param.numel())


def predict(model, image_path, resize=299, crop=299, preprocess=None):
    input_image = Image.open(image_path)

    if preprocess is None:
        preprocess = transforms.Compose(
            [
                transforms.Resize(resize),
                transforms.CenterCrop(crop),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
                ),
            ]
        )

    input_tensor = preprocess(input_image)
    input_batch = input_tensor.unsqueeze(
        0
    )  # create a mini-batch as expected by the model

    # move the input and model to GPU for speed if available
    if torch.cuda.is_available():
        input_batch = input_batch.to("cuda")
        model.to("cuda")

    with torch.no_grad():
        output = model(input_batch)
        return torch.argmax(output)
