import torchvision
import sys

sys.path.append("../../tools")  # Ugly hack
from utils import export_model_weights, print_model_parameters, predict

import torch

if __name__ == "__main__":

    weights = torchvision.models.Inception_V3_Weights.DEFAULT
    inception = torchvision.models.inception_v3(
        weights=weights,
        transform_input=False
    )

    inception.transform_input = False
    inception.eval()

    export_model_weights(inception, "inception_v3_weights.bin")

    # class_labels = weights.meta["categories"]
    # prediction = predict(inception, "bird.jpg")
    # print(prediction, class_labels[prediction])
