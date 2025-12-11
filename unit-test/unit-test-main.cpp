// Unit test for kernels

#include "core/slang-basic.h"
#include "example-base/example-base.h"
#include "kernels.h"
#include "inference-context.h"
#include "torch-reader.h"
#include <random>

using Slang::ComPtr;

static const ExampleResources resourceBase("unit-test");

enum class UNetBlockKind
{
    Down,
    Up
};

class UNetBlock : public RefObject
{
protected:
    RefPtr<InferencingContext> inferencingCtx;
    RefPtr<Conv2DKernel> conv1, conv2;
    RefPtr<Conv2DKernel> downTransform;
    RefPtr<TransposedConv2DKernel> upTransform;
    RefPtr<LinearKernel> timeEmbedTransform;
    RefPtr<BroadcastAddKernel> broadcastAdd;
public:
    int inChannels;
    int outChannels;
    UNetBlock(RefPtr<InferencingContext> inferencingCtx, UNetBlockKind kind, int inChannels, int outChannels, int timeEmbedDim)
        : inferencingCtx(inferencingCtx), inChannels(inChannels), outChannels(outChannels)
    {
        if (kind == UNetBlockKind::Down)
        {
            conv1 = new Conv2DKernel(inferencingCtx, 16, 3, inChannels, outChannels, "conv1");
            downTransform = new Conv2DKernel(inferencingCtx, 16, 4, outChannels, outChannels, "transformDown");
        }
        else
        {
            conv1 = new Conv2DKernel(inferencingCtx, 16, 3, 2*inChannels, outChannels, "conv1");
            upTransform = new TransposedConv2DKernel(inferencingCtx, 16, 4, 2, outChannels, outChannels, "transformUp");
        }
        conv2 = new Conv2DKernel(inferencingCtx, 16, 3, outChannels, outChannels, "conv2");
        timeEmbedTransform = new LinearKernel(inferencingCtx, ActivationFunction::ReLU, 128, timeEmbedDim, outChannels);
        broadcastAdd = new BroadcastAddKernel(inferencingCtx);
    }

    SlangResult loadParams(TorchParamReader& reader)
    {
        SLANG_RETURN_ON_FAIL(timeEmbedTransform->loadParams(reader));
        SLANG_RETURN_ON_FAIL(conv1->loadParams(reader, true));
        if (downTransform)
            SLANG_RETURN_ON_FAIL(downTransform->loadParams(reader, false));
        if (upTransform)
            SLANG_RETURN_ON_FAIL(upTransform->loadParams(reader));
        SLANG_RETURN_ON_FAIL(conv2->loadParams(reader, true));
        return SLANG_OK;
    }

    ComPtr<rhi::IBuffer> forward(InferencingTask& task, rhi::IBuffer* inputImage, int inputWidth, int inputHeight, rhi::IBuffer* timeEmbedding)
    {
        auto transformedTimeEmbedding = timeEmbedTransform->queueExecute(task, timeEmbedding);
        auto conv1Result = conv1->queueExecute(task, inputImage, inputWidth, inputHeight, 1, 1);
        int shapeA[] = { inputHeight, inputWidth, outChannels };
        int shapeB[] = { 1, 1, outChannels };
        auto added = broadcastAdd->queueExecute(task, conv1Result, makeArrayView(shapeA), transformedTimeEmbedding, makeArrayView(shapeB));
        auto conv2Result = conv2->queueExecute(task, added, inputWidth, inputHeight, 1, 1);
        rhi::IBuffer* finalResult = conv2Result;
        if (downTransform)
            finalResult = downTransform->queueExecute(task, conv2Result, inputWidth, inputHeight, 2, 1);
        else
            finalResult = upTransform->queueExecute(task, conv2Result, inputWidth, inputHeight, 1);
        return ComPtr<rhi::IBuffer>(finalResult);
    }
};

class UNetModel : public RefObject
{
protected:
    RefPtr<InferencingContext> inferencingCtx;
    List<RefPtr<UNetBlock>> downBlocks;
    List<RefPtr<UNetBlock>> upBlocks;
    RefPtr<TimeEmbedingKernel> timeEmbedKernel;
    RefPtr<Conv2DKernel> initialConv;
    RefPtr<Conv2DKernel> finalConv;
    RefPtr<ConcatKernel> concat;
public:
    UNetModel(RefPtr<InferencingContext> inferencingCtx, int inputChannels, int outputChannels)
        : inferencingCtx(inferencingCtx)
    {
        static const int timeEmbedDim = 32;
        timeEmbedKernel = new TimeEmbedingKernel(inferencingCtx, timeEmbedDim);
        int channelSizes[] = { 64, 128, 256, 512, 1024 };
        for (Index i = 0; i < SLANG_COUNT_OF(channelSizes) - 1; i++)
        {
            downBlocks.add(new UNetBlock(
                inferencingCtx,
                UNetBlockKind::Down,
                channelSizes[i],
                channelSizes[i+1],
                timeEmbedDim));
            upBlocks.add(new UNetBlock(
                inferencingCtx,
                UNetBlockKind::Up,
                channelSizes[SLANG_COUNT_OF(channelSizes) - 1 - i],
                channelSizes[SLANG_COUNT_OF(channelSizes) - 2 - i],
                timeEmbedDim));
        }
        initialConv = new Conv2DKernel(inferencingCtx, 16, 3, inputChannels, channelSizes[0], "initialConv");
        finalConv = new Conv2DKernel(inferencingCtx, 16, 1, channelSizes[0], outputChannels, "predictedNoiseConv");
        concat = new ConcatKernel(inferencingCtx);
    }

    SlangResult loadParams(TorchParamReader& reader)
    {
        SLANG_RETURN_ON_FAIL(timeEmbedKernel->loadParams(reader));
        SLANG_RETURN_ON_FAIL(initialConv->loadParams(reader, false));
        for (auto& block : downBlocks)
        {
            SLANG_RETURN_ON_FAIL(block->loadParams(reader));
        }
        for (auto& block : upBlocks)
        {
            SLANG_RETURN_ON_FAIL(block->loadParams(reader));
        }
        SLANG_RETURN_ON_FAIL(finalConv->loadParams(reader, false));
        return SLANG_OK;
    }

    ComPtr<rhi::IBuffer> forward(InferencingTask& task, rhi::IBuffer* inputImage, int inputWidth, int inputHeight, int timeStep)
    {
        auto timeEmbedding = timeEmbedKernel->queueExecute(task, timeStep);
        auto x = initialConv->queueExecute(task, inputImage, inputWidth, inputHeight, 1, 1);
        List<ComPtr<rhi::IBuffer>> skipConnections;
        for (auto& block : downBlocks)
        {
            x = block->forward(task, x, inputWidth, inputHeight, timeEmbedding);
            skipConnections.add(x);
            inputWidth /= 2;
            inputHeight /= 2;
        }
        for (Index i = 0; i < upBlocks.getCount(); i++)
        {
            auto& block = upBlocks[i];
            // Concat skip connection
            auto skipConnection = skipConnections[skipConnections.getCount() - 1 - i];
            int shape[] = { inputHeight, inputWidth, block->inChannels };
            auto shapeView = makeArrayView(shape);
            x = concat->queueExecute(task, x, shapeView, skipConnection, shapeView, 3);
            // Up block
            x = block->forward(task, x, inputWidth, inputHeight, timeEmbedding);
            inputWidth *= 2;
            inputHeight *= 2;
        }
        x = finalConv->queueExecute(task, x, inputWidth, inputHeight, 1, 1);
        return ComPtr<rhi::IBuffer>(x);
    }
};

class DiffusionReverseStepKernel : public RefObject
{
protected:
    RefPtr<InferencingContext> inferencingCtx;
    ComPtr<rhi::IComputePipeline> pipeline;
public:
    DiffusionReverseStepKernel(RefPtr<InferencingContext> inferencingCtx)
        : inferencingCtx(inferencingCtx)
    {
        slang::TypeLayoutReflection* paramTypeLayout = nullptr;
        pipeline = inferencingCtx->createComputePipeline("diffusionReverseStep", {});
    }
    void forward(
        InferencingTask& task,
        float alpha,
        float alphaCumprod,
        float beta,
        rhi::IBuffer* currentImage,
        rhi::IBuffer* predictedNoise,
        uint32_t imageWidth,
        uint32_t imageHeight,
        uint32_t channelCount,
        uint32_t seed,
        uint32_t t)
    {
        struct DiffusionStepParams
        {
            // Raw Pointers (Buffer Device Address)
            rhi::DeviceAddress currentImage;   // x_t (Input)
            rhi::DeviceAddress predictedNoise; // epsilon_theta (Input)
            rhi::DeviceAddress outputImage;    // x_{t-1} (Output)

            // Scalar Coefficients
            float coeff1;
            float coeff2;
            float coeff3;

            uint32_t totalElements;
            uint32_t seed;             // Change this every frame/step on CPU!
        };
        DiffusionStepParams params;
        params.coeff1 = 1.0f / sqrtf(alpha);
        params.coeff2 = (alpha - 1.0f) / (sqrtf(1.0f - alphaCumprod) * sqrtf(alpha));
        if (t > 0)
            params.coeff3 = sqrtf(beta);
        else
            params.coeff3 = 0.0f;
        params.currentImage = currentImage->getDeviceAddress();
        params.predictedNoise = predictedNoise->getDeviceAddress();
        params.outputImage = currentImage->getDeviceAddress(); // In-place
        params.totalElements = imageWidth * imageHeight * channelCount;
        params.seed = seed;
        task.dispatchKernel(pipeline, (params.totalElements + 255) / 256, 1, 1, params);
    }
};

void writeImagePNG(
    const char* filename,
    int width,
    int height,
    int numChannels,
    const void* data);

struct UnitTestProgram : public TestBase
{
    ComPtr<rhi::IDevice> gDevice;

    RefPtr<InferencingContext> gInferencingCtx;

    SlangResult execute(int argc, char* argv[])
    {
        parseOption(argc, argv);
        rhi::DeviceDesc deviceDesc;
        deviceDesc.slang.targetProfile = "spirv_1_6";
        deviceDesc.deviceType = rhi::DeviceType::Vulkan;
        deviceDesc.enableAftermath = true;
        deviceDesc.enableValidation = true;
        rhi::getRHI()->enableDebugLayers();
        gDevice = rhi::getRHI()->createDevice(deviceDesc);
        if (!gDevice)
            return SLANG_FAIL;

        gInferencingCtx = new InferencingContext(gDevice);

        SLANG_RETURN_ON_FAIL(testSimpleConvolution());
        SLANG_RETURN_ON_FAIL(testSimpleTransposedConvolution());
        SLANG_RETURN_ON_FAIL(testUNetModel());

        printf("all tests passed!\n");
        return SLANG_OK;
    }

    SlangResult testCheck(bool condition, const char* testName, const char* message)
    {
        if (!condition)
        {
            printf("%s: check failed: %s\n", testName, message);
            return SLANG_FAIL;
        }
        return SLANG_OK;
    }

#define TEST_CHECK(testName, condition) SLANG_RETURN_ON_FAIL(testCheck((condition), (testName), #condition)) 

    SlangResult testSimpleConvolution()
    {
        float inputData[] = { 1,2,3,4,5,
                           6,7,8,9,10,
                           11,12,13,14,15,
                           16,17,18,19,20,
                           21,22,23,24,25};
        auto readInput = [&](int x, int y) { return inputData[y * 5 + x]; };
        auto inputBuffer = gInferencingCtx->createBuffer(inputData, 5 * 5 * sizeof(float));
        float convWeights[9] = { 0.1, 0.5, 0.2,
                                0.5, 1.0, 0.5,
                                0.2, 0.5, 0.4 };
        float convBiases[] = { 1000.0f };
        Conv2DKernel convKernel = Conv2DKernel(gInferencingCtx.Ptr(), 4, 3, 1, 1);
        auto task = gInferencingCtx->createTask();
        convKernel.loadParams(3, 1, convWeights, convBiases);
        auto outputBuffer = convKernel.queueExecute(task, inputBuffer, 5, 5, 1, 1);

        auto readWeight = [&](int x, int y) {return convWeights[y * 3 + x]; };
        
        renderDocBeginFrame();
        task.execute();
        float outputData[25];
        gDevice->getQueue(rhi::QueueType::Graphics)->waitOnHost();
        gDevice->readBuffer(outputBuffer, 0, sizeof(outputData), outputData);
        renderDocEndFrame();
        float v0 = outputData[0];
        float expectedV0 = readInput(0, 0) * readWeight(1, 1) + readInput(1, 0) * readWeight(2, 1) +
            readInput(0, 1) * readWeight(2, 1) + readInput(1, 1) * readWeight(2, 2) + convBiases[0];
        TEST_CHECK("simpleConvolution", fabs(v0 - expectedV0) < 1e-3f);
        return SLANG_OK;
    }

    SlangResult testSimpleTransposedConvolution()
    {
        float inputData[] = { 1,2,3,4,5,
                           6,7,8,9,10,
                           11,12,13,14,15,
                           16,17,18,19,20,
                           21,22,23,24,25};
        float convWeights[9] = { 0.1, 0.5, 0.2,
                                0.5, 1.0, 0.5,
                                0.2, 0.5, 0.4 };
        float convBiases[] = { 1000.0f };
        auto readInput = [&](int x, int y) { return inputData[y * 5 + x]; };
        auto inputBuffer = gInferencingCtx->createBuffer(inputData, 5 * 5 * sizeof(float));
        TransposedConv2DKernel transposedConvKernel = TransposedConv2DKernel(gInferencingCtx.Ptr(), 4, 3, 1, 1, 1);
        auto task = gInferencingCtx->createTask();
        transposedConvKernel.loadParams(3, 1, convWeights, convBiases);
        auto outputBuffer = transposedConvKernel.queueExecute(task, inputBuffer, 5, 5, 1);
        auto readWeight = [&](int x, int y) {return convWeights[y * 3 + x]; };
        renderDocBeginFrame();
        task.execute();
        float outputData[25];
        gDevice->readBuffer(outputBuffer, 0, sizeof(outputData), outputData);
        renderDocEndFrame();
        float v0 = outputData[0];
        float expectedV0 = readInput(1, 0) * readWeight(1, 1) + readInput(0, 1) * readWeight(1, 0) +
            readInput(1, 1) * readWeight(0, 0) + convBiases[0];
        TEST_CHECK("simpleTransposedConvolution", fabs(v0 - expectedV0) < 1e-3f);
        return SLANG_OK;
    }

    void initImage(List<float>& imageData, int width, int height, int channels = 1)
    {
        uint32_t seed = 1723;
        std::mt19937 gen(seed);

        std::normal_distribution<float> dist(0.0f, 1.0f);
        imageData.setCount(width * height * channels);

        // 3. Generate
        for (Index i = 0; i < imageData.getCount(); i++)
        {
            imageData[i] = dist(gen);
        }
    }

    SlangResult testUNetModel()
    {
        UNetModel model = UNetModel(gInferencingCtx, 1, 1);
        
        DiffusionReverseStepKernel diffusionKernel = DiffusionReverseStepKernel(gInferencingCtx);

        RefPtr<FileStream> fileStream = new FileStream();
        SLANG_RETURN_ON_FAIL(fileStream->init(resourceBase.resolveResource("model_weights.bin"), FileMode::Open));
        TorchParamReader reader(fileStream);
        SLANG_RETURN_ON_FAIL(model.loadParams(reader));
        auto task = gInferencingCtx->createTask();
        
        uint32_t imageSize = 32;
        int outputChannelCount = 1;
        int inputChannelCount = 1;

        List<float> inputImageData;
        initImage(inputImageData, imageSize, imageSize, inputChannelCount);

        struct NoiseParam
        {
            float alpha;
            float beta;
            float alphaCumprod;
        };
        int stepCount = 3;
        float betaStart = 1e-4;
        float betaEnd = 0.02f;
        List<NoiseParam> noiseSchedule;
        noiseSchedule.setCount(stepCount);
        for (int step = 0; step < stepCount; step++)
        {
            float t = (float)step / (float)(stepCount - 1);
            float betaT = betaStart + t * (betaEnd - betaStart);
            float alphaT = 1.0f - betaT;
            float alphaCumprodT = (step == 0) ? alphaT : noiseSchedule[step - 1].alphaCumprod * alphaT;
            noiseSchedule[step].alpha = alphaT;
            noiseSchedule[step].beta = betaT;
            noiseSchedule[step].alphaCumprod = alphaCumprodT;
        }
        auto inputImage = gInferencingCtx->createBuffer(
            inputImageData.getBuffer(),
            inputImageData.getCount() * sizeof(float),
            "inputImage");
        
        static const int largePrime = 15485863;
        for (int step = stepCount - 1; step >= 0; step--)
        {
            auto noiseParam = noiseSchedule[step];
            auto predictedNoise = model.forward(task, inputImage, imageSize, imageSize, step);

            diffusionKernel.forward(
                task,
                noiseParam.alpha,
                noiseParam.alphaCumprod,
                noiseParam.beta,
                inputImage,
                predictedNoise,
                imageSize,
                imageSize,
                1,
                step*largePrime,
                step);
        }
        renderDocBeginFrame();
        task.execute();
        renderDocEndFrame();

        // Read back final image
        List<float> outputImageData;
        outputImageData.setCount(imageSize* imageSize * outputChannelCount);
        gDevice->readBuffer(inputImage, 0, outputImageData.getCount() * sizeof(float), outputImageData.getBuffer());

        // Save to disk as png
        // Convert to 8-bit
        List<uint8_t> outputImageData8Bit;
        outputImageData8Bit.setCount(imageSize* imageSize* outputChannelCount);
        for (int i = 0; i < outputImageData.getCount(); i++)
        {
            float v = outputImageData[i];
            v = (Slang::Math::Clamp(v, -1.0f, 1.0f) + 1.0) * 0.5f;
            outputImageData8Bit[i] = static_cast<uint8_t>(v * 255.0f);
        }
        writeImagePNG("output.png", imageSize, imageSize, outputChannelCount, outputImageData8Bit.getBuffer());
        return SLANG_OK;
    }

    inline unsigned short floatToHalf(float val)
    {
        uint32_t x = 0;
        memcpy(&x, &val, sizeof(float));

        unsigned short bits = (x >> 16) & 0x8000;
        unsigned short m = (x >> 12) & 0x07ff;
        unsigned int e = (x >> 23) & 0xff;
        if (e < 103)
            return bits;
        if (e > 142)
        {
            bits |= 0x7c00u;
            bits |= e == 255 && (x & 0x007fffffu);
            return bits;
        }
        if (e < 113)
        {
            m |= 0x0800u;
            bits |= (m >> (114 - e)) + ((m >> (113 - e)) & 1);
            return bits;
        }
        bits |= ((e - 112) << 10) | (m >> 1);
        bits += m & 1;
        return bits;
    }
};

int main(int argc, char** argv)
{
    UnitTestProgram app;
    if (SLANG_FAILED(app.execute(argc, argv)))
    {
        return -1;
    }
    return 0;
}
