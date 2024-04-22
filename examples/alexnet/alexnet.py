import sys

import torch
import torchvision
from PIL import Image
from torchvision import transforms

sys.path.append('../../tools')  # Ugly hack 
from utils import export_model_weights, print_model_parameters


def predict(model, image_path):
    input_image = Image.open(image_path)
    preprocess = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(227),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    input_tensor = preprocess(input_image)
    input_batch = input_tensor.unsqueeze(0) # create a mini-batch as expected by the model

    # move the input and model to GPU for speed if available
    if torch.cuda.is_available():
        input_batch = input_batch.to('cuda')
        model.to('cuda')

    with torch.no_grad():
        output = model(input_batch)
        print(torch.argmax(output))


if __name__ == "__main__":
    alexnet = torchvision.models.alexnet(weights=torchvision.models.AlexNet_Weights.DEFAULT)
    print_model_parameters(alexnet)  # print layer names and number of parameters
    export_model_weights(alexnet, 'alexnet_weights.bin')
    # predict('cat.jpg')

