import torch
import numpy as np

# Define input and kernel data as tensors
input_data = torch.tensor([
    0.823, 0.217, 0.435, 0.981, 0.742,
    0.109, 0.518, 0.374, 0.681, 0.147,
    0.956, 0.729, 0.654, 0.087, 0.392,
    0.784, 0.921, 0.543, 0.231, 0.816,
    0.472, 0.614, 0.102, 0.987, 0.398,
    0.051, 0.756, 0.841, 0.293, 0.128,
    0.417, 0.632, 0.095, 0.184, 0.529,
    0.871, 0.958, 0.213, 0.347, 0.725,
    0.461, 0.012, 0.278, 0.195, 0.649,
    0.853, 0.707, 0.988, 0.988, 0.322,
    0.345, 0.123, 0.789, 0.123, 0.456,
    0.456, 0.789, 0.123, 0.345, 0.123,
    0.789, 0.123, 0.345, 0.123, 0.456,
    0.123, 0.345, 0.123, 0.789, 0.123,
    0.345, 0.123, 0.789, 0.123, 0.456
], dtype=torch.float)

kernel_data = torch.tensor([
    0.128, 0.754, 0.987,
    0.321, 0.412, 0.635,
    0.298, 0.017, 0.845,
    0.514, 0.729, 0.952,
    0.684, 0.378, 0.159,
    0.823, 0.547, 0.216,
    0.983, 0.231, 0.456,
    0.178, 0.654, 0.821,
    0.345, 0.987, 0.123,
    0.789, 0.543, 0.210,
    0.012, 0.371, 0.638,
    0.456, 0.198, 0.907,
    0.101, 0.432, 0.759,
    0.234, 0.567, 0.890,
    0.543, 0.876, 0.219,
    0.345, 0.678, 0.011,
    0.678, 0.011, 0.345,
    0.011, 0.345, 0.678
], dtype=torch.float)

# Reshape input data to a 4D tensor (batch_size, channels, height, width)
input_data = input_data.reshape(1, 3, 5, 5)

# Define the convolution layer
conv2d = torch.nn.Conv2d(in_channels=3, out_channels=2, kernel_size=3, padding=1, bias=False)

# Set the weights of the convolution layer
conv2d.weight = torch.nn.Parameter(kernel_data.reshape(2, 3, 3, 3))

# Perform the convolution
output = conv2d(input_data)

# Print the output as cpp vector
output = torch.flatten(output)
print("std::vector<float> expected = {", end="")
for i in range(len(output)):
    if i != 0:
        print(", ", end="")
    print(str(round(output[i].item(), 5)) + "f", end="")
print("};")
