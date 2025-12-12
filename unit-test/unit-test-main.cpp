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
            upTransform = new TransposedConv2DKernel(inferencingCtx, 16, 4, 2, outChannels, outChannels, "transformUp");
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
        deviceDesc.enableAftermath = true;
        deviceDesc.enableValidation = true;
        rhi::getRHI()->enableDebugLayers();
        gDevice = rhi::getRHI()->createDevice(deviceDesc);
        if (!gDevice)
            return SLANG_FAIL;

        gInferencingCtx = new InferencingContext(gDevice);

#if 0
        SLANG_RETURN_ON_FAIL(testUp0());
        SLANG_RETURN_ON_FAIL(testBottleneckConcat());
        SLANG_RETURN_ON_FAIL(testDown0());
        SLANG_RETURN_ON_FAIL(testBroadcastAdd());
        SLANG_RETURN_ON_FAIL(testDown0Conv1());
        SLANG_RETURN_ON_FAIL(testGlobalTimeEmbed());
        SLANG_RETURN_ON_FAIL(testDown0Transform());
        SLANG_RETURN_ON_FAIL(testInitialConv());
        SLANG_RETURN_ON_FAIL(testSimpleConvolution());
        SLANG_RETURN_ON_FAIL(testSimpleTransposedConvolution());
        SLANG_RETURN_ON_FAIL(testStep495());
#endif
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

        DDIMStepKernel diffusionKernel = DDIMStepKernel(gInferencingCtx);

        RefPtr<FileStream> fileStream = new FileStream();
        SLANG_RETURN_ON_FAIL(fileStream->init(resourceBase.resolveResource("model_weights.bin"), FileMode::Open));
        TorchParamReader reader(fileStream);
        SLANG_RETURN_ON_FAIL(model.loadParams(reader));

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
        int trainingSteps = 500; // Must match Python NOISE_STEPS
        int inferenceSteps = 50;
        float betaStart = 1e-4;
        float betaEnd = 0.02f;
        List<NoiseParam> noiseSchedule;
        noiseSchedule.setCount(trainingSteps);
        float currentAlphaCumprod = 1.0f;
        for (int t = 0; t < trainingSteps; t++)
        {
            // Linear Schedule Calculation
            float ratio = (float)t / (float)(trainingSteps - 1);
            float betaT = betaStart + ratio * (betaEnd - betaStart);
            float alphaT = 1.0f - betaT;

            currentAlphaCumprod *= alphaT;

            // Store in table
            noiseSchedule[t].alpha = alphaT;
            noiseSchedule[t].beta = betaT;
            noiseSchedule[t].alphaCumprod = currentAlphaCumprod;
        }
        auto imageA = gInferencingCtx->createBuffer(
            inputImageData.getBuffer(),
            inputImageData.getCount() * sizeof(float),
            "imageA");
        auto imageB = gInferencingCtx->createBuffer(
            nullptr,
            inputImageData.getCount() * sizeof(float),
            "imageB");
        auto outputImage = imageA;
        static const int largePrime = 15485863;
        //renderDocBeginFrame();
        for (int step = inferenceSteps - 1; step >= 0; step--)
        {
            // Map 0..100 -> 0..500
            // e.g. Step 99 -> t=495
            int t = (step * trainingSteps) / inferenceSteps;

            // B. Previous Training Time (The target we are jumping TO)
            // e.g., Step 98/100 -> t_prev=490
            int step_prev = step - 1;
            int t_prev = (step_prev * trainingSteps) / inferenceSteps;

            // C. Get Alpha Cumprod values
            float alphaBar_t = noiseSchedule[t].alphaCumprod;

            // Special handling for the final step:
            // If we are going below t=0, the target is the pure image (AlphaBar = 1.0)
            float alphaBar_prev = (step_prev < 0) ? 1.0f : noiseSchedule[t_prev].alphaCumprod;
            auto task = gInferencingCtx->createTask();

            // Use 't' for the model, but 'step' for the loop logic
            auto predictedNoise = model.forward(task, imageA, imageSize, imageSize, t);

            auto noiseParam = noiseSchedule[t];
            diffusionKernel.forward(
                task,
                imageA,
                predictedNoise,
                imageB,
                alphaBar_t,
                alphaBar_prev,
                imageSize * imageSize * outputChannelCount);
            task.execute();
            outputImage = imageB;
            Swap(imageA, imageB);

        }
        //renderDocEndFrame();

        // Read back final image
        List<float> outputImageData;
        outputImageData.setCount(imageSize * imageSize * outputChannelCount);
        gDevice->readBuffer(outputImage, 0, outputImageData.getCount() * sizeof(float), outputImageData.getBuffer());

        // Save to disk as png
        // Convert to 8-bit
        List<uint8_t> outputImageData8Bit;
        outputImageData8Bit.setCount(imageSize * imageSize * outputChannelCount);
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

    List<float> loadRawFloats(String path)
    {
        path = resourceBase.resolveResource(path.getBuffer());
        List<uint8_t> bytes;
        if (SLANG_FAILED(File::readAllBytes(path, bytes)))
            return {};

        List<float> result;
        result.setCount(bytes.getCount() / 4);
        memcpy(result.getBuffer(), bytes.getBuffer(), bytes.getCount());
        return result;
    }

    bool checkOutput(rhi::IBuffer* outputBuffer, const List<float>& expectedOutput)
    {
        List<float> outputData;
        outputData.setCount(expectedOutput.getCount());
        if (outputBuffer->getDesc().size < outputData.getCount() * sizeof(float))
            return false;
        gDevice->readBuffer(outputBuffer, 0, outputData.getCount() * sizeof(float), outputData.getBuffer());
        for (Index i = 0; i < outputData.getCount(); i++)
        {
            if (isnan(outputData[i]))
                return false;
            float diff = fabs(outputData[i] - expectedOutput[i]);
            if (diff < 1e-2f)
                continue;
            float abs = fabs(outputData[i]);
            if (abs > 1e-3f && diff / abs > 1e-3f)
                return false;
        }
        return true;
    }

    SlangResult testGlobalTimeEmbed()
    {
        auto expectedOutput = loadRawFloats("debug_dump/global_time_embed_output.bin");
        if (expectedOutput.getCount() == 0)
            return SLANG_FAIL;
        
        TimeEmbedingKernel glboalTimeEmbedKernel = TimeEmbedingKernel(gInferencingCtx, 32);
        TorchParamReader reader = TorchParamReader(resourceBase.resolveResource("debug_dump/global_time_linear1.bin"));
        SLANG_RETURN_ON_FAIL(glboalTimeEmbedKernel.loadParams(reader));
        
        auto task = gInferencingCtx->createTask();
        auto output = glboalTimeEmbedKernel.queueExecute(task, 495);
        task.execute();

        TEST_CHECK("testGlobalTimeEmbed", checkOutput(output, expectedOutput));
        return SLANG_OK;
    }

    SlangResult testInitialConv()
    {
        auto expectedOutput = loadRawFloats("debug_dump/conv0_output.bin");
        if (expectedOutput.getCount() == 0)
            return SLANG_FAIL;
        Conv2DKernel initialConvKernel = Conv2DKernel(gInferencingCtx, 16, 3, 1, 1, 64, ActivationFunction::None, "initialConv");
        TorchParamReader reader = TorchParamReader(resourceBase.resolveResource("debug_dump/conv0.bin"));
        SLANG_RETURN_ON_FAIL(initialConvKernel.loadParams(reader, false));
        List<float> inputImageData = loadRawFloats("debug_dump/initial_x_input.bin");
        auto inputImage = gInferencingCtx->createBuffer(
            inputImageData.getBuffer(),
            inputImageData.getCount() * sizeof(float),
            "inputImage");
        auto task = gInferencingCtx->createTask();
        auto output = initialConvKernel.queueExecute(task, inputImage, 32, 32, 1);
        task.execute();
        TEST_CHECK("testInitialConv", checkOutput(output, expectedOutput));
        return SLANG_OK;
    }

    SlangResult testDown0Transform()
    {
        // 1. Load Expected Output
        auto expectedOutput = loadRawFloats("debug_dump/down0_transform_output.bin");
        if (expectedOutput.getCount() == 0)
            return SLANG_FAIL;

        // 2. Initialize Kernel
        // Config: Kernel=4, InChannels=128, OutChannels=128
        // Note: The previous layers in Block 0 (conv1/conv2) expanded the channels from 64 to 128.
        Conv2DKernel transformKernel = Conv2DKernel(gInferencingCtx, 16, 4, 2, 128, 128, ActivationFunction::None, "down0Transform");

        // 3. Load Weights (No BatchNorm fusion for transform layers)
        TorchParamReader reader = TorchParamReader(resourceBase.resolveResource("debug_dump/down0_transform.bin"));
        SLANG_RETURN_ON_FAIL(transformKernel.loadParams(reader, false));

        // 4. Load Input
        // Input size should be 32x32x128
        List<float> inputImageData = loadRawFloats("debug_dump/down0_transform_input.bin");
        auto inputImage = gInferencingCtx->createBuffer(
            inputImageData.getBuffer(),
            inputImageData.getCount() * sizeof(float),
            "down0TransformInput");

        // 5. Execute
        // Input: 32x32. Stride: 2. Padding: 1.
        auto task = gInferencingCtx->createTask();
        auto output = transformKernel.queueExecute(task, inputImage, 32, 32, 1);

        task.execute();

        // 6. Verify
        TEST_CHECK("testDown0Transform", checkOutput(output, expectedOutput));
        return SLANG_OK;
    }

    SlangResult testDown0Conv1()
    {
        // 1. Load Expected Output
        auto expectedOutput = loadRawFloats("debug_dump/down0_conv1_fused_output.bin");
        if (expectedOutput.getCount() == 0)
            return SLANG_FAIL;
        // 2. Initialize Kernel
        // Config: Kernel=3, InChannels=64, OutChannels=128
        Conv2DKernel conv1Kernel = Conv2DKernel(gInferencingCtx, 16, 3, 1, 64, 128, ActivationFunction::ReLU, "conv1");
        // 3. Load Weights (With BatchNorm fusion)
        Conv2DLayerParams convParams;
        {
            TorchParamReader reader = TorchParamReader(resourceBase.resolveResource("debug_dump/down0_conv1.bin"));
            reader.readConv2DLayer(64, 128, 3, convParams);
        }
        BatchNorm2DLayerParams bnParams;
        {
            TorchParamReader reader = TorchParamReader(resourceBase.resolveResource("debug_dump/down0_bn1.bin"));
            reader.readBatchNorm2DLayer(128, bnParams);
        }
        convParams.fuseBatchNorm(bnParams);
        SLANG_RETURN_ON_FAIL(conv1Kernel.loadParams(
            3, 128, convParams.weights.getBuffer(), convParams.biases.getBuffer()));
        // 4. Load Input
        // Input size should be 32x32x64
        List<float> inputImageData = loadRawFloats("debug_dump/down0_conv1_input.bin");
        auto inputImage = gInferencingCtx->createBuffer(
            inputImageData.getBuffer(),
            inputImageData.getCount() * sizeof(float),
            "down0Conv1Input");
        // 5. Execute
        auto task = gInferencingCtx->createTask();
        auto output = conv1Kernel.queueExecute(task, inputImage, 32, 32, 1);
        task.execute();
        // 6. Verify
        TEST_CHECK("testDown0Conv1", checkOutput(output, expectedOutput));
        return SLANG_OK;
    }

    SlangResult testBroadcastAdd()
    {
        // 1. Load Expected Output
        auto expectedOutput = loadRawFloats("debug_dump/down0_conv2_input.bin");
        if (expectedOutput.getCount() == 0)
            return SLANG_FAIL;
        // 2. Initialize Kernel
        BroadcastAddKernel broadcastAddKernel = BroadcastAddKernel(gInferencingCtx);
        // 3. Load Inputs
        List<float> inputAData = loadRawFloats("debug_dump/down0_conv1_fused_output.bin");
        auto inputABuffer = gInferencingCtx->createBuffer(
            inputAData.getBuffer(),
            inputAData.getCount() * sizeof(float),
            "broadcastAddInputA");
        List<float> inputBData = loadRawFloats("debug_dump/down0_time_proj_output.bin");
        auto inputBBuffer = gInferencingCtx->createBuffer(
            inputBData.getBuffer(),
            inputBData.getCount() * sizeof(float),
            "broadcastAddInputB");
        // 4. Execute
        auto task = gInferencingCtx->createTask();
        int shapeA[] = { 32, 32, 128 };
        int shapeB[] = { 1, 1, 128 };
        auto output = broadcastAddKernel.queueExecute(
            task,
            inputABuffer,
            makeArrayView(shapeA),
            inputBBuffer,
            makeArrayView(shapeB));
        task.execute();
        // 5. Verify
        TEST_CHECK("testBroadcastAdd", checkOutput(output, expectedOutput));
        return SLANG_OK;
    }

    SlangResult testDown0()
    {
        UNetModel model = UNetModel(gInferencingCtx, 1, 1);
        TorchParamReader reader = TorchParamReader(resourceBase.resolveResource("model_weights.bin"));
        SLANG_RETURN_ON_FAIL(model.loadParams(reader));
        auto expectedOutput = loadRawFloats("debug_dump/down0_transform_output.bin");
        if (expectedOutput.getCount() == 0)
            return SLANG_FAIL;

        // Verify loaded weights.
        {
            TorchParamReader reader1 = TorchParamReader(resourceBase.resolveResource("debug_dump/down0_time_proj.bin"));
            LinearLayerParams linearParams;
            reader1.readLinearLayer(32, 128, linearParams);
            TEST_CHECK("testDown0_timeProjWeights",
                checkOutput(model.downBlocks[0]->timeEmbedTransform->weightsBuffer, linearParams.weights));

        }

        List<float> imageInputData = loadRawFloats("debug_dump/down0_conv1_input.bin");
        auto inputImage = gInferencingCtx->createBuffer(
            imageInputData.getBuffer(),
            imageInputData.getCount() * sizeof(float),
            "inputImage");
        List<float> timeEmbedInputData = loadRawFloats("debug_dump/down0_time_proj_input.bin");
        auto timeEmbedInput = gInferencingCtx->createBuffer(
            timeEmbedInputData.getBuffer(),
            timeEmbedInputData.getCount() * sizeof(float),
            "timeEmbedInput");
        auto task = gInferencingCtx->createTask();
        auto result = model.downBlocks[0]->forward(task, inputImage, 32, 32, timeEmbedInput);
        task.execute();
        TEST_CHECK("testDown0", checkOutput(result, expectedOutput));
        return SLANG_OK;
    }

    SlangResult testBottleneckConcat()
    {
        // 1. Load the input (Output of the last encoder block)
        // Note: You might need to add a hook in Python to dump 'down3_output' or 
        // just use the 'up0_concat_output' and slice it in half if you want to be clever.
        // Better: Add hook for 'model.downs[-1]' output in Python -> 'down3_output.bin'
        auto down3Out = loadRawFloats("debug_dump/down3_transform_output.bin");
        auto buffer = gInferencingCtx->createBuffer(down3Out.getBuffer(), down3Out.getCount()*sizeof(float),"down3Out");

        // 2. Load Expected Result (The input to UpBlock0)
        auto expected = loadRawFloats("debug_dump/up0_concat_output.bin");

        // 3. Execute Concat (Axis 2 = Channels)
        ConcatKernel concat(gInferencingCtx);
        auto task = gInferencingCtx->createTask();

        // In this specific model, down3 outputs [2, 2, 1024] (assuming 32x32 -> 2x2)
        // Concatenating two of them -> [2, 2, 2048]
        int shape[] = { 2, 2, 1024 };
        auto result = concat.queueExecute(task, buffer, makeArrayView(shape), buffer, makeArrayView(shape), 2);

        task.execute();
        TEST_CHECK("testBottleneckConcat", checkOutput(result, expected));
        return SLANG_OK;
    }

    SlangResult testUp0()
    {
        // 1. Input: The Concatenated tensor (2048 channels)
        auto inputData = loadRawFloats("debug_dump/up0_concat_output.bin");
        auto inputBuf = gInferencingCtx->createBuffer(inputData.getBuffer(), inputData.getCount()*sizeof(float));

        // 2. Time Input
        auto timeData = loadRawFloats("debug_dump/down0_time_proj_input.bin");
        auto timeBuf = gInferencingCtx->createBuffer(timeData.getBuffer(), timeData.getCount()*sizeof(float));

        // 3. Setup Block
        // InChannels=1024, OutChannels=512. 
        // Note: The block constructor automatically doubles inChannels for conv1 logic (1024*2 = 2048).
        UNetModel model = UNetModel(gInferencingCtx, 1, 1);
        TorchParamReader reader = TorchParamReader(resourceBase.resolveResource("model_weights.bin"));
        SLANG_RETURN_ON_FAIL(model.loadParams(reader));
        auto expectedOutput = loadRawFloats("debug_dump/down0_transform_output.bin");
        if (expectedOutput.getCount() == 0)
            return SLANG_FAIL;
        auto& upBlock = *(model.upBlocks[0]);

        // 4. Execute
        auto task = gInferencingCtx->createTask();
        // Input size: 2x2. Output should be 4x4.
        auto result = upBlock.forward(task, inputBuf, 2, 2, timeBuf);

        task.execute();

        // 5. Verify
        auto expected = loadRawFloats("debug_dump/up0_output_output.bin");
        TEST_CHECK("testUp0", checkOutput(result, expected));
        return SLANG_OK;
    }

    SlangResult testStep495()
    {
        // 1. Setup Model
        UNetModel model(gInferencingCtx, 1, 1);
        RefPtr<FileStream> fileStream = new FileStream();
        // Ensure this matches the .bin corresponding to unet_mnist.pth!
        SLANG_RETURN_ON_FAIL(fileStream->init(resourceBase.resolveResource("model_weights.bin"), FileMode::Open));
        TorchParamReader reader(fileStream);
        SLANG_RETURN_ON_FAIL(model.loadParams(reader));

        // 2. Load Inputs
        List<float> inputData = loadRawFloats("debug_dump/step495_input.bin");
        if (inputData.getCount() == 0) return SLANG_FAIL;

        auto inputImage = gInferencingCtx->createBuffer(
            inputData.getBuffer(),
            inputData.getCount() * sizeof(float),
            "step495_input"
        );

        // 3. Run Forward Pass
        int t = 495;
        auto task = gInferencingCtx->createTask();

        // Note: ensure input width/height matches Python (32)
        auto result = model.forward(task, inputImage, 32, 32, t);

        task.execute(); // Or just let debug mode handle it

        // 4. Verify
        auto expected = loadRawFloats("debug_dump/step495_output.bin");
        TEST_CHECK("testStep495", checkOutput(result, expected));

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
