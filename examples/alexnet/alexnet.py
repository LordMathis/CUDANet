import torchvision
import sys

sys.path.append('../../tools')  # Ugly hack 
from utils import export_model_weights, print_model_parameters

if __name__ == "__main__":
    alexnet = torchvision.models.alexnet(pretrained=True)
    print_model_parameters(alexnet)  # print layer names and number of parameters
    export_model_weights(alexnet, 'alexnet_weights.bin')
    print()
    print(alexnet)

