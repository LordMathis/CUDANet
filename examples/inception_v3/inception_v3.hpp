#ifndef INCEPTION_V3_HPP
#define INCEPTION_V3_HPP

#include <cudanet.cuh>

class BasicConv2d : public CUDANet::Module {
  public:
    BasicConv2d(
        const shape2d      inputShape,
        const int          inputChannels,
        const int          outputChannels,
        const shape2d      kernelSize,
        const shape2d      stride,
        const shape2d      padding,
        const std::string &prefix
    );

    ~BasicConv2d();

    float *forward(const float *d_input);

    shape2d getOutputDims();

    int getOutputChannels();

    int outputChannels;

    CUDANet::Layers::Conv2d      *conv;
    CUDANet::Layers::BatchNorm2d *batchNorm;
};

class InceptionA : public CUDANet::Module {
  public:
    InceptionA(
        const shape2d      inputShape,
        const int          inputChannels,
        const int          poolFeatures,
        const std::string &prefix
    );
    ~InceptionA();
    float *forward(const float *d_input);
    shape2d getOutputDims();
    int getOutputChannels();

  private:
    shape2d inputShape;
    int     inputChannels;
    int     poolFeatures;

    BasicConv2d *branch1x1;

    BasicConv2d *branch5x5_1;
    BasicConv2d *branch5x5_2;

    BasicConv2d *branch3x3dbl_1;
    BasicConv2d *branch3x3dbl_2;
    BasicConv2d *branch3x3dbl_3;

    CUDANet::Layers::AvgPooling2d *branchPool_1;
    BasicConv2d                   *branchPool_2;

    CUDANet::Layers::Concat *concat_1;
    CUDANet::Layers::Concat *concat_2;
    CUDANet::Layers::Concat *concat_3;
};

class InceptionB : public CUDANet::Module {
  public:
    InceptionB(
        const shape2d      inputShape,
        const int          inputChannels,
        const std::string &prefix
    );
    ~InceptionB();
    float *forward(const float *d_input);
    shape2d getOutputDims();
    int getOutputChannels();

  private:
    shape2d inputShape;
    int     inputChannels;

    BasicConv2d *branch3x3;

    BasicConv2d *branch3x3dbl_1;
    BasicConv2d *branch3x3dbl_2;
    BasicConv2d *branch3x3dbl_3;

    CUDANet::Layers::MaxPooling2d *branchPool;

    CUDANet::Layers::Concat *concat_1;
    CUDANet::Layers::Concat *concat_2;
};

class InceptionC : public CUDANet::Module {
  public:
    InceptionC(
        const shape2d      inputShape,
        const int          inputChannels,
        const int          nChannels_7x7,
        const std::string &prefix
    );
    ~InceptionC();
    float *forward(const float *d_input);
    shape2d getOutputDims();
    int getOutputChannels();

  private:

    shape2d inputShape;
    int     inputChannels;

    BasicConv2d *branch1x1;

    BasicConv2d *branch7x7_1;
    BasicConv2d *branch7x7_2;
    BasicConv2d *branch7x7_3;

    BasicConv2d *branch7x7dbl_1;
    BasicConv2d *branch7x7dbl_2;
    BasicConv2d *branch7x7dbl_3;
    BasicConv2d *branch7x7dbl_4;
    BasicConv2d *branch7x7dbl_5;

    CUDANet::Layers::AvgPooling2d *branchPool_1;
    BasicConv2d                   *branchPool_2;

    CUDANet::Layers::Concat *concat_1;
    CUDANet::Layers::Concat *concat_2;
    CUDANet::Layers::Concat *concat_3;
};

class InceptionD : public CUDANet::Module {
  public:
    InceptionD(
        const shape2d      inputShape,
        const int          inputChannels,
        const std::string &prefix
    );
    ~InceptionD();
    float *forward(const float *d_input);
    shape2d getOutputDims();
    int getOutputChannels();

  private:
    shape2d inputShape;
    int     inputChannels;

    BasicConv2d *branch3x3_1;
    BasicConv2d *branch3x3_2;

    BasicConv2d *branch7x7x3_1;
    BasicConv2d *branch7x7x3_2;
    BasicConv2d *branch7x7x3_3;
    BasicConv2d *branch7x7x3_4;

    CUDANet::Layers::MaxPooling2d *branchPool;

    CUDANet::Layers::Concat *concat_1;
    CUDANet::Layers::Concat *concat_2;
};

class InceptionE : public CUDANet::Module {
  public:
    InceptionE(
        const shape2d      inputShape,
        const int          inputChannels,
        const std::string &prefix
    );
    ~InceptionE();
    float *forward(const float *d_input);
    shape2d getOutputDims();
    int getOutputChannels();

  private:
    shape2d inputShape;
    int     inputChannels;

    BasicConv2d *branch1x1;

    BasicConv2d             *branch3x3_1;
    BasicConv2d             *branch3x3_2a;
    BasicConv2d             *branch3x3_2b;
    CUDANet::Layers::Concat *branch_3x3_2_concat;

    BasicConv2d             *branch3x3dbl_1;
    BasicConv2d             *branch3x3dbl_2;
    BasicConv2d             *branch3x3dbl_3a;
    BasicConv2d             *branch3x3dbl_3b;
    CUDANet::Layers::Concat *branch_3x3dbl_3_concat;

    CUDANet::Layers::AvgPooling2d *branchPool_1;
    BasicConv2d                   *branchPool_2;

    CUDANet::Layers::Concat *concat_1;
    CUDANet::Layers::Concat *concat_2;
    CUDANet::Layers::Concat *concat_3;
};

class InceptionV3 : public CUDANet::Model {
  public:
    InceptionV3(
        const shape2d inputShape,
        const int     inputChannels,
        const int     outputSize
    );
    ~InceptionV3();
    float *predict(const float *input);

  private:
    BasicConv2d *conv2d_1a_3x3;
    BasicConv2d *conv2d_2a_3x3;
    BasicConv2d *conv2d_2b_3x3;

    CUDANet::Layers::MaxPooling2d *maxpool1;

    BasicConv2d *conv2d_3b_1x1;
    BasicConv2d *conv2d_4a_3x3;

    CUDANet::Layers::MaxPooling2d *maxpool2;

    InceptionA *Mixed_5b;
    InceptionA *Mixed_5c;
    InceptionA *Mixed_5d;

    InceptionB *Mixed_6a;

    InceptionC *Mixed_6b;
    InceptionC *Mixed_6c;
    InceptionC *Mixed_6d;
    InceptionC *Mixed_6e;

    InceptionD *Mixed_7a;

    InceptionE *Mixed_7b;
    InceptionE *Mixed_7c;

    CUDANet::Layers::AdaptiveAvgPooling2d *avgpool;

    CUDANet::Layers::Dense *fc;
};

#endif