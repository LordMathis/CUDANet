import torchvision
import sys

sys.path.append('../../tools')  # Ugly hack 
from utils import export_model_weights, print_model_parameters, predict


if __name__ == "__main__":
    inception = torchvision.models.inception_v3(weights=torchvision.models.Inception_V3_Weights.DEFAULT)
    inception.eval()

    # print_model_parameters(inception)  # print layer names and number of parameters

    # export_model_weights(inception, 'inception_v3_weights.bin')

    print(predict(inception, "./margot.jpg"))
