import sys

import torch
import torchvision.transforms as transforms
from PIL import Image
import numpy as np

sys.path.append("../../../tools")
from utils import print_cpp_vector

torch.manual_seed(0)

def generate_random_image(size=(24, 24)):
    # Generate a random RGB image
    random_image = np.random.randint(0, 256, size=(*size, 3), dtype=np.uint8)
    return Image.fromarray(random_image)

def preprocess_image(image, resize=16, crop=16):
    preprocess = transforms.Compose([
        transforms.Resize(resize),
        transforms.CenterCrop(crop),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    return preprocess(image)

def normalize_tensor(tensor):
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    return normalize(tensor)


def gen_preprocess_test_result():
    # Generate a random image
    random_image = generate_random_image()
    random_image.save("resources/test_image.jpg")

    # Preprocess the image
    preprocessed = preprocess_image(random_image)

    # Print the preprocessed data
    print("Preprocessed image data:")
    print_cpp_vector(preprocessed.flatten(), "output")


def gen_normalize_test_result():
    input_tensor = torch.rand(3, 8, 8)

    print("Input tensor: ")
    print_cpp_vector(input_tensor.flatten(), "input")

    normalized = normalize_tensor(input_tensor)
    print_cpp_vector(normalized.flatten(), "expected_output")


if __name__ == "__main__":
    # print("Preprocess Test\n")
    # gen_preprocess_test_result()

    print("\nNormalize Test\n")
    gen_normalize_test_result()