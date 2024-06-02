#include <gtest/gtest.h>

#include <inception_v3.hpp>

class BasicConv2dTest : public ::testing::Test {
  protected:
    BasicConv2d *basic_conv2d;

    shape2d     inputShape;
    int         inputChannels;
    int         outputChannels;
    shape2d     kernelSize;
    shape2d     stride;
    shape2d     padding;
    std::string prefix = "test";

    float *d_input;
    float *d_output;

    std::vector<float> input;
    std::vector<float> expected;

    std::vector<float> convWeights;
    std::vector<float> convBiases;

    std::vector<float> bnWeights;
    std::vector<float> bnBiases;

    virtual void SetUp() override {
        basic_conv2d = nullptr;
    }

    virtual void TearDown() override {
        // Clean up
        delete basic_conv2d;
    }

    void runTest() {
        cudaError_t cudaStatus;

        basic_conv2d = new BasicConv2d(
            inputShape, inputChannels, outputChannels, kernelSize, stride,
            padding, prefix
        );

        std::pair<std::string, CUDANet::Layers::SequentialLayer *> layerPair =
            basic_conv2d->getLayers()[0];

        ASSERT_EQ(layerPair.first, prefix + ".conv");

        CUDANet::Layers::Conv2d *conv =
            dynamic_cast<CUDANet::Layers::Conv2d *>(layerPair.second);
        conv->setWeights(convWeights.data());
        conv->setBiases(convBiases.data());

        ASSERT_EQ(conv->getWeights().size(), convWeights.size());
        ASSERT_EQ(conv->getBiases().size(), convBiases.size());

        cudaStatus = cudaGetLastError();
        EXPECT_EQ(cudaStatus, cudaSuccess);

        layerPair = basic_conv2d->getLayers()[1];
        ASSERT_EQ(layerPair.first, prefix + ".bn");

        CUDANet::Layers::BatchNorm2d *bn =
            dynamic_cast<CUDANet::Layers::BatchNorm2d *>(layerPair.second);
        bn->setWeights(bnWeights.data());
        bn->setBiases(bnBiases.data());

        ASSERT_EQ(bn->getWeights().size(), bnWeights.size());
        ASSERT_EQ(bn->getBiases().size(), bnBiases.size());

        cudaStatus = cudaGetLastError();
        EXPECT_EQ(cudaStatus, cudaSuccess);

        cudaStatus =
            cudaMalloc((void **)&d_input, sizeof(float) * input.size());
        EXPECT_EQ(cudaStatus, cudaSuccess);

        cudaStatus = cudaMemcpy(
            d_input, input.data(), sizeof(float) * input.size(),
            cudaMemcpyHostToDevice
        );
        EXPECT_EQ(cudaStatus, cudaSuccess);

        d_output = basic_conv2d->forward(d_input);

        cudaStatus = cudaGetLastError();
        EXPECT_EQ(cudaStatus, cudaSuccess);

        int                outputSize = basic_conv2d->getOutputSize();
        std::vector<float> output(outputSize);
        cudaStatus = cudaMemcpy(
            output.data(), d_output, sizeof(float) * output.size(),
            cudaMemcpyDeviceToHost
        );
        EXPECT_EQ(cudaStatus, cudaSuccess);

        for (int i = 0; i < output.size(); ++i) {
            EXPECT_NEAR(expected[i], output[i], 1e-5f);
        }
    }
};

TEST_F(BasicConv2dTest, BasicConv2dTest1) {
    inputShape     = {8, 8};
    inputChannels  = 3;
    outputChannels = 6;
    kernelSize     = {3, 3};
    stride         = {1, 1};
    padding        = {1, 1};

    // 3x3x3x6
    convWeights = {
        0.18365f, 0.08568f, 0.08126f, 0.68022f, 0.41391f, 0.71204f, 0.66917f,
        0.63586f, 0.28914f, 0.43624f, 0.03018f, 0.47986f, 0.71336f, 0.82706f,
        0.587f,   0.58516f, 0.29813f, 0.19312f, 0.42975f, 0.62522f, 0.34256f,
        0.28057f, 0.37367f, 0.54325f, 0.63421f, 0.46445f, 0.56908f, 0.95247f,
        0.73934f, 0.51263f, 0.14464f, 0.0956f,  0.68846f, 0.14675f, 0.75427f,
        0.50547f, 0.37078f, 0.03316f, 0.42855f, 0.94293f, 0.73855f, 0.86475f,
        0.20687f, 0.37793f, 0.77947f, 0.24402f, 0.07547f, 0.22212f, 0.57188f,
        0.5098f,  0.71999f, 0.63828f, 0.53237f, 0.42874f, 0.43621f, 0.87348f,
        0.0073f,  0.07752f, 0.45232f, 0.78307f, 0.74813f, 0.73456f, 0.0378f,
        0.78518f, 0.6989f,  0.50484f, 0.74265f, 0.39178f, 0.91015f, 0.11684f,
        0.11499f, 0.10394f, 0.30637f, 0.86116f, 0.63743f, 0.64142f, 0.97882f,
        0.30948f, 0.32144f, 0.76108f, 0.81794f, 0.50111f, 0.82209f, 0.49028f,
        0.79417f, 0.3257f,  0.32221f, 0.4007f,  0.86371f, 0.2271f,  0.9414f,
        0.66233f, 0.60802f, 0.65701f, 0.41021f, 0.1135f,  0.21892f, 0.93389f,
        0.65786f, 0.26068f, 0.59535f, 0.15048f, 0.48185f, 0.91072f, 0.18252f,
        0.64154f, 0.89179f, 0.54726f, 0.60756f, 0.31149f, 0.30717f, 0.79877f,
        0.71727f, 0.12418f, 0.48471f, 0.46097f, 0.66898f, 0.35467f, 0.38027f,
        0.16989f, 0.88578f, 0.84377f, 0.26529f, 0.26057f, 0.30256f, 0.84876f,
        0.8849f,  0.08982f, 0.88191f, 0.1944f,  0.42052f, 0.62898f, 0.692f,
        0.51155f, 0.99903f, 0.56947f, 0.73144f, 0.88091f, 0.28472f, 0.98895f,
        0.41364f, 0.1927f,  0.07227f, 0.421f,   0.85347f, 0.19329f, 0.07098f,
        0.19418f, 0.06585f, 0.49083f, 0.85071f, 0.96747f, 0.45057f, 0.54361f,
        0.49552f, 0.23454f, 0.97412f, 0.26663f, 0.09274f, 0.1662f,  0.04784f,
        0.76303f
    };
    convBiases.resize(outputChannels, 0.0f);

    bnWeights = {0.69298f, 0.27049f, 0.85854f, 0.52973f, 0.29644f, 0.68932f};
    bnBiases  = {0.74976f, 0.42745f, 0.22132f, 0.21262f, 0.03726f, 0.9719f};

    input = {
        0.75539f, 0.17641f, 0.8331f,  0.80627f, 0.51712f, 0.87756f, 0.97027f,
        0.21354f, 0.28498f, 0.05118f, 0.37124f, 0.40528f, 0.13661f, 0.08692f,
        0.73809f, 0.57278f, 0.73534f, 0.31338f, 0.15362f, 0.80245f, 0.49524f,
        0.81208f, 0.24074f, 0.42534f, 0.62236f, 0.75915f, 0.06382f, 0.66723f,
        0.13448f, 0.96896f, 0.87197f, 0.67366f, 0.67885f, 0.49345f, 0.08446f,
        0.94116f, 0.8659f,  0.22848f, 0.53262f, 0.51307f, 0.89661f, 0.72223f,
        0.90541f, 0.47353f, 0.85476f, 0.04177f, 0.04039f, 0.7917f,  0.56188f,
        0.53777f, 0.91714f, 0.84847f, 0.16995f, 0.59803f, 0.05454f, 0.00365f,
        0.01429f, 0.42586f, 0.31519f, 0.222f,   0.9149f,  0.51885f, 0.82969f,
        0.42778f, 0.82913f, 0.01303f, 0.92699f, 0.09225f, 0.00284f, 0.75769f,
        0.74072f, 0.59012f, 0.40777f, 0.0469f,  0.08751f, 0.23163f, 0.51327f,
        0.67095f, 0.31971f, 0.97841f, 0.82292f, 0.58917f, 0.31565f, 0.4728f,
        0.41885f, 0.36524f, 0.28194f, 0.70945f, 0.36008f, 0.23199f, 0.71093f,
        0.33364f, 0.34199f, 0.42114f, 0.40026f, 0.77819f, 0.79858f, 0.93793f,
        0.45238f, 0.97922f, 0.73814f, 0.11831f, 0.08414f, 0.56552f, 0.99841f,
        0.53862f, 0.71138f, 0.42274f, 0.48724f, 0.48201f, 0.5361f,  0.97138f,
        0.27607f, 0.33018f, 0.07456f, 0.77788f, 0.58824f, 0.77027f, 0.3938f,
        0.28081f, 0.14074f, 0.06907f, 0.75419f, 0.11888f, 0.35715f, 0.34481f,
        0.05669f, 0.21063f, 0.8664f,  0.00087f, 0.88281f, 0.55202f, 0.68655f,
        0.96262f, 0.53907f, 0.9227f,  0.74055f, 0.84487f, 0.22792f, 0.83233f,
        0.42938f, 0.39054f, 0.59604f, 0.4141f,  0.25982f, 0.9311f,  0.35475f,
        0.71432f, 0.29186f, 0.16604f, 0.90708f, 0.00171f, 0.11541f, 0.35719f,
        0.9221f,  0.18793f, 0.90198f, 0.29281f, 0.72144f, 0.54645f, 0.71165f,
        0.59584f, 0.24041f, 0.60954f, 0.64945f, 0.8122f,  0.34145f, 0.92178f,
        0.99894f, 0.25076f, 0.45067f, 0.71997f, 0.09573f, 0.57334f, 0.63273f,
        0.49469f, 0.72747f, 0.33449f, 0.13755f, 0.49458f, 0.50319f, 0.91328f,
        0.57269f, 0.21927f, 0.36831f, 0.88708f, 0.62277f, 0.08318f, 0.01425f,
        0.17998f, 0.34614f, 0.82303f
    };

    expected = {
        0.0f,     0.49814f, 0.22097f, 0.3619f,  0.46957f, 0.69706f, 1.06759f,
        0.25578f, 0.0f,     0.91978f, 0.53499f, 0.78382f, 1.13748f, 1.27999f,
        1.39561f, 0.59403f, 0.1681f,  1.1653f,  0.9397f,  0.99945f, 1.09875f,
        1.11738f, 1.48957f, 0.39551f, 0.17473f, 1.36075f, 1.38633f, 1.10036f,
        1.66809f, 1.24004f, 1.51673f, 0.35859f, 0.50363f, 1.90002f, 1.76062f,
        1.77264f, 1.653f,   0.98297f, 0.97645f, 0.36179f, 0.65388f, 1.82326f,
        1.62819f, 1.53234f, 1.52987f, 1.1909f,  1.19085f, 0.0f,     0.0f,
        1.00418f, 0.9884f,  1.06528f, 1.10918f, 0.95965f, 1.01066f, 0.0f,
        0.0f,     0.0f,     0.0f,     0.0f,     0.0f,     0.06699f, 0.0f,
        0.0f,     0.0f,     0.31227f, 0.1577f,  0.24142f, 0.29244f, 0.35219f,
        0.55728f, 0.09206f, 0.18279f, 0.52608f, 0.43298f, 0.57281f, 0.64957f,
        0.67697f, 0.79076f, 0.25769f, 0.17322f, 0.45144f, 0.50649f, 0.44384f,
        0.45046f, 0.52827f, 0.65169f, 0.26233f, 0.33391f, 0.54569f, 0.61824f,
        0.71162f, 0.72201f, 0.59606f, 0.69006f, 0.17808f, 0.53409f, 0.84795f,
        0.81671f, 0.72767f, 0.70439f, 0.49824f, 0.77586f, 0.28972f, 0.41066f,
        0.78739f, 0.74518f, 0.69849f, 0.72851f, 0.58154f, 0.59843f, 0.0988f,
        0.12992f, 0.69539f, 0.58411f, 0.53047f, 0.67763f, 0.45745f, 0.42961f,
        0.02356f, 0.0f,     0.1524f,  0.17941f, 0.20621f, 0.07853f, 0.0f,
        0.01425f, 0.0f,     0.0f,     0.0f,     0.0f,     0.0f,     0.0f,
        0.0f,     0.0f,     0.0f,     0.0f,     0.53197f, 0.23141f, 0.65858f,
        0.51061f, 1.18983f, 1.88715f, 0.0f,     0.0f,     0.48249f, 0.27706f,
        0.4758f,  0.37868f, 0.19115f, 1.3417f,  0.0f,     0.0f,     0.79729f,
        0.40467f, 0.75802f, 1.25205f, 1.05397f, 0.99662f, 0.0f,     0.05866f,
        1.25683f, 1.37623f, 1.3692f,  0.8155f,  0.79031f, 0.79231f, 0.0f,
        0.66813f, 1.55738f, 0.86795f, 1.74891f, 1.46206f, 0.44267f, 0.71223f,
        0.0f,     0.01532f, 0.9517f,  0.9068f,  0.04987f, 0.68475f, 0.60834f,
        0.5695f,  0.0f,     0.0f,     0.0f,     0.0f,     0.0f,     0.0f,
        0.0f,     0.0f,     0.0f,     0.0f,     0.0f,     0.0f,     0.0f,
        0.0f,     0.0f,     0.13772f, 0.0f,     0.0f,     0.54903f, 0.17714f,
        0.56106f, 0.37474f, 0.59682f, 0.80188f, 0.23357f, 0.0f,     0.3935f,
        0.10723f, 0.21271f, 0.2933f,  0.40208f, 0.98239f, 0.19075f, 0.06934f,
        0.69707f, 0.59654f, 0.72836f, 0.94042f, 0.29819f, 0.65969f, 0.15544f,
        0.21691f, 0.94429f, 0.74025f, 0.57482f, 0.85235f, 0.6364f,  0.64997f,
        0.43117f, 0.23959f, 0.86925f, 0.74496f, 1.18404f, 0.91728f, 0.66074f,
        0.14145f, 0.0f,     0.0f,     0.82383f, 0.54479f, 0.37769f, 0.37376f,
        0.18698f, 0.41482f, 0.0f,     0.0f,     0.0f,     0.0f,     0.0f,
        0.0f,     0.0f,     0.0f,     0.0f,     0.0f,     0.0f,     0.0f,
        0.0f,     0.0f,     0.0f,     0.19054f, 0.0f,     0.0f,     0.13366f,
        0.02072f, 0.17679f, 0.21344f, 0.22093f, 0.39159f, 0.0f,     0.0f,
        0.21636f, 0.1152f,  0.05384f, 0.17127f, 0.31197f, 0.26403f, 0.0f,
        0.0f,     0.2079f,  0.40094f, 0.25855f, 0.2949f,  0.21378f, 0.29504f,
        0.0f,     0.0f,     0.55198f, 0.28422f, 0.44235f, 0.39818f, 0.24589f,
        0.24885f, 0.0f,     0.0f,     0.39978f, 0.49578f, 0.31662f, 0.57204f,
        0.22104f, 0.09188f, 0.0f,     0.0f,     0.30446f, 0.11957f, 0.18297f,
        0.21063f, 0.11165f, 0.1131f,  0.0f,     0.0f,     0.0f,     0.0f,
        0.0f,     0.0f,     0.0f,     0.0f,     0.0f,     0.0f,     0.04903f,
        0.0f,     0.21626f, 0.35491f, 0.86898f, 0.9025f,  0.0f,     0.36255f,
        1.46154f, 1.38429f, 1.44938f, 1.41407f, 1.45809f, 1.77706f, 0.88361f,
        0.09394f, 0.92029f, 1.01541f, 1.09078f, 1.05394f, 1.25418f, 1.40895f,
        0.78881f, 0.62721f, 1.55362f, 1.70365f, 1.83765f, 1.7833f,  1.52613f,
        1.39727f, 0.44845f, 0.80839f, 1.73151f, 1.63702f, 1.60352f, 1.63081f,
        1.5767f,  1.99697f, 0.91883f, 0.62179f, 1.8053f,  1.63263f, 1.72401f,
        2.45383f, 1.25455f, 1.07616f, 0.38183f, 0.56256f, 1.8342f,  1.49708f,
        1.54651f, 0.90693f, 0.85377f, 0.9732f,  0.0f,     0.0f,     0.42826f,
        0.47554f, 0.23275f, 0.5115f,  0.14327f, 0.23193f, 0.0f
    };

    runTest();
};