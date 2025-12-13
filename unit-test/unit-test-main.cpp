// Unit test for kernels

#include "core/slang-basic.h"
#include "example-base/example-base.h"
#include "kernels.h"
#include "inference-context.h"
#include "torch-reader.h"
#include <random>
#include <chrono>

using Slang::ComPtr;

static const ExampleResources resourceBase("unit-test");

enum class UNetBlockKind
{
    Down,
    Up
};

class UNetBlock : public RefObject
{
public:
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
            conv1 = new Conv2DKernel(inferencingCtx, 16, 3, 1, inChannels, outChannels, ActivationFunction::ReLU, "conv1");
            downTransform = new Conv2DKernel(inferencingCtx, 16, 4, 2, outChannels, outChannels, ActivationFunction::None, "transformDown");
        }
        else
        {
            conv1 = new Conv2DKernel(inferencingCtx, 16, 3, 1, 2*inChannels, outChannels, ActivationFunction::ReLU, "conv1");
            upTransform = new TransposedConv2DKernel(inferencingCtx, 16, 4, 2, outChannels, outChannels, ActivationFunction::None, "transformUp");
        }
        conv2 = new Conv2DKernel(inferencingCtx, 16, 3, 1, outChannels, outChannels, ActivationFunction::ReLU, "conv2");
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

    void writeResult(const char* name, rhi::IBuffer* buffer)
    {
        ComPtr<ISlangBlob> blob;
        inferencingCtx->getDevice()->readBuffer(buffer, 0, buffer->getDesc().size, blob.writeRef());
        File::writeAllBytes(String(name) + ".bin", blob->getBufferPointer(), blob->getBufferSize());
    }

    ComPtr<rhi::IBuffer> forward(InferencingTask& task, rhi::IBuffer* inputImage, int inputWidth, int inputHeight, rhi::IBuffer* timeEmbedding)
    {
        auto transformedTimeEmbedding = timeEmbedTransform->queueExecute(task, timeEmbedding);
        writeResult("time_embed_out", transformedTimeEmbedding);
        auto conv1Result = conv1->queueExecute(task, inputImage, inputWidth, inputHeight, 1);
        writeResult("conv1_fused_out", conv1Result);

        int shapeA[] = { inputHeight, inputWidth, outChannels };
        int shapeB[] = { 1, 1, outChannels };
        auto added = broadcastAdd->queueExecute(task, conv1Result, makeArrayView(shapeA), transformedTimeEmbedding, makeArrayView(shapeB));
        auto conv2Result = conv2->queueExecute(task, added, inputWidth, inputHeight, 1);
        writeResult("conv2_fused_out", conv2Result);

        rhi::IBuffer* finalResult = conv2Result;
        if (downTransform)
            finalResult = downTransform->queueExecute(task, conv2Result, inputWidth, inputHeight, 1);
        else
            finalResult = upTransform->queueExecute(task, conv2Result, inputWidth, inputHeight, 1);
        return ComPtr<rhi::IBuffer>(finalResult);
    }
};

class UNetModel : public RefObject
{
protected:
    RefPtr<InferencingContext> inferencingCtx;
    RefPtr<TimeEmbedingKernel> timeEmbedKernel;
    RefPtr<Conv2DKernel> initialConv;
    RefPtr<Conv2DKernel> finalConv;
    RefPtr<ConcatKernel> concat;
public:
    List<RefPtr<UNetBlock>> downBlocks;
    List<RefPtr<UNetBlock>> upBlocks;
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
        initialConv = new Conv2DKernel(inferencingCtx, 16, 3, 1, inputChannels, channelSizes[0], ActivationFunction::None, "initialConv");
        finalConv = new Conv2DKernel(inferencingCtx, 16, 1, 1, channelSizes[0], outputChannels, ActivationFunction::None, "predictedNoiseConv");
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
        auto x = initialConv->queueExecute(task, inputImage, inputWidth, inputHeight, 1);
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
            x = concat->queueExecute(task, x, shapeView, skipConnection, shapeView, 2);
            // Up block
            x = block->forward(task, x, inputWidth, inputHeight, timeEmbedding);
            inputWidth *= 2;
            inputHeight *= 2;
        }
        x = finalConv->queueExecute(task, x, inputWidth, inputHeight, 0);
        return ComPtr<rhi::IBuffer>(x);
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
        //rhi::getRHI()->enableDebugLayers();
        gDevice = rhi::getRHI()->createDevice(deviceDesc);
        if (!gDevice)
            return SLANG_FAIL;

        gInferencingCtx = new InferencingContext(gDevice);

        SLANG_RETURN_ON_FAIL(testSimpleConvolution());
        SLANG_RETURN_ON_FAIL(testSimpleTransposedConvolution());

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
                           21,22,23,24,25 };
        auto readInput = [&](int x, int y) { return inputData[y * 5 + x]; };
        auto inputBuffer = gInferencingCtx->createBuffer(inputData, 5 * 5 * sizeof(float));
        float convWeights[9] = { 0.1, 0.5, 0.2,
                                0.5, 1.0, 0.5,
                                0.2, 0.5, 0.4 };
        float convBiases[] = { 1000.0f };
        Conv2DKernel convKernel = Conv2DKernel(gInferencingCtx.Ptr(), 4, 3, 1, 1, 1);
        auto task = gInferencingCtx->createTask();
        convKernel.loadParams(3, 1, convWeights, convBiases);
        auto outputBuffer = convKernel.queueExecute(task, inputBuffer, 5, 5, 1);

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
                           21,22,23,24,25 };
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
