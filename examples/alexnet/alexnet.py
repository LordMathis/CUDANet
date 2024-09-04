import sys

import torchvision

sys.path.append('../../tools')  # Ugly hack 
from utils import export_model_weights, print_model_parameters, predict

if __name__ == "__main__":

    weights = torchvision.models.AlexNet_Weights.DEFAULT
    alexnet = torchvision.models.alexnet(weights=weights)

    # print_model_parameters(alexnet)  # print layer names and number of parameters
    export_model_weights(alexnet, 'alexnet_weights.bin')
    
    # class_labels = weights.meta["categories"]
    # prediction = predict(alexnet, "margot.jpg")
    # print(prediction, class_labels[prediction])

