import torch
import torchvision
import sys

from torchsummary import summary

inception = torchvision.models.inception_v3(weights=torchvision.models.Inception_V3_Weights.DEFAULT)
inception.eval()

sys.path.append('../../tools')  # Ugly hack 
from utils import export_model_weights, print_model_parameters

print_model_parameters(inception)  # print layer names and number of parameters

inception.cuda()

summary(inception, (3, 299, 299))