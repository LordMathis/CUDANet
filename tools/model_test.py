import torch

from utils import export_model_weights, print_cpp_vector


class TestModel(torch.nn.Module):

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)

        self.conv1 = torch.nn.Conv2d(
            in_channels=2,
            out_channels=2,
            kernel_size=3,
            stride=1,
            padding=0,
            bias=False,
        )

        self.maxpool1 = torch.nn.MaxPool2d(kernel_size=2, stride=2)
        self.activation = torch.nn.ReLU()

        self.linear = torch.nn.Linear(in_features=8, out_features=3, bias=False)
        self.softmax = torch.nn.Softmax(dim=0)

    def set_weights(self):

        # fmt: off
        conv2d_weights = torch.tensor([
            0.18313, 0.53363, 0.39527, 0.27575, 0.3433,  0.41746,
            0.16831, 0.61693, 0.54599, 0.99692, 0.77127, 0.25146,
            0.4206,  0.16291, 0.93484, 0.79765, 0.74982, 0.78336,
            0.6386,  0.87744, 0.33587, 0.9691,  0.68437, 0.65098,
            0.48153, 0.97546, 0.8026,  0.36689, 0.98152, 0.37351,
            0.68407, 0.2684,  0.2855,  0.76195, 0.67828, 0.603
            
        ]).reshape(2, 2, 3, 3)
        # fmt: on
        self.conv1.weight = torch.nn.Parameter(conv2d_weights)

        # fmt: off
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
        # fmt: on
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

    # fmt: off
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
    # fmt: on

    print("Single test output:")
    out = model(input)
    print_cpp_vector(out)

    print("Multiple predict test output 1:")
    # fmt: off
    input = torch.tensor([
        0.81247, 0.03579, 0.26577, 0.80374, 0.64584, 0.19658, 0.04817,
        0.50769, 0.33502, 0.01739, 0.32263, 0.69625, 0.07433, 0.98283,
        0.21217, 0.48437, 0.58012, 0.86991, 0.81879, 0.63589, 0.30264,
        0.90318, 0.12978, 0.35972, 0.95847, 0.58633, 0.55025, 0.68302,
        0.61422, 0.79524, 0.7205,  0.72481, 0.51553, 0.83032, 0.23561,
        0.80631, 0.23548, 0.84634, 0.05839, 0.76526, 0.39654, 0.95635,
        0.75422, 0.75341, 0.82431, 0.79371, 0.72413, 0.88557, 0.33594,
        0.56363, 0.12415, 0.05635, 0.15952, 0.27887, 0.05417, 0.58474,
        0.75129, 0.1788,  0.88958, 0.49793, 0.85386, 0.86262, 0.05568,
        0.16811, 0.72188, 0.08683, 0.66985, 0.62707, 0.4035,  0.51822,
        0.46545, 0.88722
    ]).reshape(2, 6, 6)
    # fmt: on
    out = model(input)
    print_cpp_vector(out)

    print("Multiple predict test output 2:")
    # fmt: off
    input = torch.tensor([
        0.83573, 0.19191, 0.16004, 0.27137, 0.64768, 0.38417, 0.02167,
        0.28834, 0.21401, 0.16624, 0.12037, 0.12706, 0.3588,  0.10685,
        0.49224, 0.71267, 0.17677, 0.29276, 0.92467, 0.76689, 0.8209,
        0.82226, 0.11581, 0.6698,  0.01109, 0.47085, 0.44912, 0.45936,
        0.83645, 0.83272, 0.81693, 0.97726, 0.60649, 0.9,     0.37,
        0.20517, 0.81921, 0.83573, 0.00271, 0.30453, 0.78925, 0.8453,
        0.80416, 0.09041, 0.0802,  0.98408, 0.19746, 0.25598, 0.09437,
        0.27681, 0.92053, 0.35385, 0.17389, 0.14293, 0.60151, 0.12338,
        0.81858, 0.56294, 0.97378, 0.93272, 0.36075, 0.64944, 0.2433,
        0.66075, 0.64496, 0.1191,  0.66261, 0.63431, 0.7137,  0.14851,
        0.84456, 0.44482
    ]).reshape(2, 6, 6)
    # fmt: on
    out = model(input)
    print_cpp_vector(out)

    export_model_weights(model, "test/resources/model.bin")
