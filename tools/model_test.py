import torch
import utils

class TestModel(torch.nn.Module):

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)

        self.conv1 = torch.nn.Conv2d(
            in_channels=2,
            out_channels=2,
            kernel_size=3,
            stride=1,
            padding=0,
            bias=False
        )
        
        self.maxpool1 = torch.nn.MaxPool2d(
            kernel_size=2,
            stride=2
        )
        self.activation = torch.nn.ReLU()

        self.linear = torch.nn.Linear(
            in_features=8,
            out_features=3,
            bias=False
        )
        self.softmax = torch.nn.Softmax(dim=0)

    def set_weights(self):

        conv2d_weights = torch.tensor([
            0.18313, 0.53363, 0.39527, 0.27575, 0.3433,  0.41746,
            0.16831, 0.61693, 0.54599, 0.99692, 0.77127, 0.25146,
            0.4206,  0.16291, 0.93484, 0.79765, 0.74982, 0.78336,
            0.6386,  0.87744, 0.33587, 0.9691,  0.68437, 0.65098,
            0.48153, 0.97546, 0.8026,  0.36689, 0.98152, 0.37351,
            0.68407, 0.2684,  0.2855,  0.76195, 0.67828, 0.603
            
        ]).reshape(2, 2, 3, 3)
        self.conv1.weight = torch.nn.Parameter(conv2d_weights)

        linear_weights = torch.tensor([
            0.36032, 0.33115, 0.02948,
            0.09802, 0.45072, 0.56266,
            0.43514, 0.80946, 0.43439,
            0.90916, 0.08605, 0.07473,
            0.94788, 0.66168, 0.34927,
            0.09464, 0.61963, 0.73775,
            0.51559, 0.81916, 0.64915,
            0.03934, 0.87608, 0.68364,
        ]).reshape(3, 8)
        self.linear.weight = torch.nn.Parameter(linear_weights)

    def forward(self, x):
        x = self.conv1(x)
        x = self.maxpool1(x)
        x = self.activation(x)

        x = torch.flatten(x)
        x = self.linear(x)
        x = self.softmax(x)

        return x
    
if __name__ == "__main__":
    model = TestModel()
    model.set_weights()

    input = torch.tensor([
        0.12762, 0.99056, 0.77565, 0.29058, 0.29787, 0.58415, 0.20484,
        0.05415, 0.60593, 0.3162,  0.08198, 0.92749, 0.72392, 0.91786,
        0.65266, 0.80908, 0.53389, 0.36069, 0.18614, 0.52381, 0.08525,
        0.43054, 0.3355,  0.96587, 0.98194, 0.71336, 0.78392, 0.50648,
        0.40355, 0.31863, 0.54686, 0.1836,  0.77171, 0.01262, 0.41108,
        0.53467, 0.3553,  0.42808, 0.45798, 0.29958, 0.3923,  0.98277,
        0.02033, 0.99868, 0.90584, 0.57554, 0.15957, 0.91273, 0.38901,
        0.27097, 0.64788, 0.84272, 0.42984, 0.07466, 0.53658, 0.83388,
        0.28232, 0.48046, 0.85626, 0.04721, 0.36139, 0.6123,  0.56991,
        0.84854, 0.61415, 0.2466,  0.20017, 0.78952, 0.93797, 0.27884,
        0.30514, 0.23521
    ]).reshape(2, 6, 6)

    # input = torch.rand(2, 6, 6)

    out = model(input)
    utils.print_cpp_vector(out)
