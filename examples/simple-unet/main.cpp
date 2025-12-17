// An example showing how to build a simple unet model to produce
// 32x32 MNIST digit images unconditioned in a diffusion process.

#include "core/slang-basic.h"
#include "example-base/example-base.h"
#include "inference-context.h"
#include "kernels.h"
#include "simple-unet.h"
#include "torch-reader.h"

#include <chrono>
#include <random>

static const ExampleResources resourceBase("simple-unet");

struct SimpleUNetProgram : public TestBase
{
    ComPtr<rhi::IDevice> gDevice;

    RefPtr<InferencingContext> gInferencingCtx;

    SlangResult execute(int argc, char* argv[])
    {
        parseOption(argc, argv);
        rhi::DeviceDesc deviceDesc;
        deviceDesc.slang.targetProfile = "spirv_1_6";
        deviceDesc.deviceType = rhi::DeviceType::Vulkan;
        // rhi::getRHI()->enableDebugLayers();
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

#define TEST_CHECK(testName, condition) \
    SLANG_RETURN_ON_FAIL(testCheck((condition), (testName), #condition))

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

    SlangResult loadModel(UNetModel& model, String modelWeightsPath)
    {
        modelWeightsPath = resourceBase.resolveResource(modelWeightsPath.getBuffer());
        RefPtr<FileStream> fileStream = new FileStream();
        if (!File::exists(modelWeightsPath))
        {
            printf(
                "Model weights not found at %s, make sure to run train.py to generate the model!\n",
                modelWeightsPath.getBuffer());
            return SLANG_FAIL;
        }
        SLANG_RETURN_ON_FAIL(fileStream->init(modelWeightsPath, FileMode::Open));
        TorchParamReader reader(fileStream);
        SLANG_RETURN_ON_FAIL(model.loadParams(reader));
        return SLANG_OK;
    }

    SlangResult testUNetModel()
    {
        UNetModel model = UNetModel(gInferencingCtx, 1, 1);

        DDIMStepKernel diffusionKernel = DDIMStepKernel(gInferencingCtx);
        SLANG_RETURN_ON_FAIL(loadModel(model, "model_weights.bin"));

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
        auto imageAStorage = gInferencingCtx->createPersistentBuffer(
            inputImageData.getBuffer(),
            inputImageData.getCount() * sizeof(float),
            "imageA");
        auto imageA = BufferView(imageAStorage);
        auto imageB = gInferencingCtx->allocScratchBuffer(
            inputImageData.getCount() * sizeof(float),
            "imageB");
        auto outputImage = imageA;
        static const int largePrime = 15485863;
        renderDocBeginFrame();
        auto task = gInferencingCtx->createTask();
        gInferencingCtx->pushAllocScope();
        SLANG_DEFER(gInferencingCtx->popAllocScope());

        auto predictedNoise = gInferencingCtx->allocScratchBuffer(
            inputImageData.getCount() * sizeof(float),
            "predictedNoise");
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

            // Use 't' for the model, but 'step' for the loop logic

            model.queueExecute(task, predictedNoise, imageA, imageSize, imageSize, t, 1);

            auto noiseParam = noiseSchedule[t];
            diffusionKernel.queueExecute(
                task,
                imageA,
                predictedNoise,
                imageB,
                alphaBar_t,
                alphaBar_prev,
                imageSize,
                imageSize,
                outputChannelCount);
            outputImage = imageB;
            Swap(imageA, imageB);
        }
        auto startTime = std::chrono::high_resolution_clock::now();
        task.execute();
        auto endTime = std::chrono::high_resolution_clock::now();
        auto elapsedMs =
            std::chrono::duration_cast<std::chrono::milliseconds>(endTime - startTime).count();
        printf("Inference Time: %lld ms\n", elapsedMs);
        renderDocEndFrame();

        // Read back final image
        List<float> outputImageData = gInferencingCtx->readBuffer<float>(outputImage);

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
        writeImagePNG(
            "output.png",
            imageSize,
            imageSize,
            outputChannelCount,
            outputImageData8Bit.getBuffer());
        return SLANG_OK;
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

    bool checkOutput(BufferView outputBuffer, const List<float>& expectedOutput)
    {
        if (outputBuffer.size < expectedOutput.getCount() * sizeof(float))
            return false;
        auto outputData = gInferencingCtx->readBuffer<float>(outputBuffer);
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
        TorchParamReader reader =
            TorchParamReader(resourceBase.resolveResource("debug_dump/global_time_linear1.bin"));
        SLANG_RETURN_ON_FAIL(glboalTimeEmbedKernel.loadParams(reader));

        auto task = gInferencingCtx->createTask();
        auto output = glboalTimeEmbedKernel.allocResultBuffer(1);
        glboalTimeEmbedKernel.queueExecute(task, output, 495);
        task.execute();

        TEST_CHECK("testGlobalTimeEmbed", checkOutput(output, expectedOutput));
        return SLANG_OK;
    }

    SlangResult testInitialConv()
    {
        auto expectedOutput = loadRawFloats("debug_dump/conv0_output.bin");
        if (expectedOutput.getCount() == 0)
            return SLANG_FAIL;
        Conv2DKernel initialConvKernel =
            Conv2DKernel(gInferencingCtx, 16, 3, 1, 1, 64, ActivationFunction::None, "initialConv");
        TorchParamReader reader =
            TorchParamReader(resourceBase.resolveResource("debug_dump/conv0.bin"));
        SLANG_RETURN_ON_FAIL(initialConvKernel.loadParams(reader, false));
        List<float> inputImageData = loadRawFloats("debug_dump/initial_x_input.bin");
        auto inputImage = gInferencingCtx->createPersistentBuffer(
            inputImageData.getBuffer(),
            inputImageData.getCount() * sizeof(float),
            "inputImage");
        auto task = gInferencingCtx->createTask();
        auto outputImage = initialConvKernel.allocateResultBuffer(32, 32, 1, 1);
        initialConvKernel.queueExecute(task, outputImage, BufferView(inputImage), 32, 32, 1);
        task.execute();
        TEST_CHECK("testInitialConv", checkOutput(outputImage, expectedOutput));
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
        Conv2DKernel transformKernel = Conv2DKernel(
            gInferencingCtx,
            16,
            4,
            2,
            128,
            128,
            ActivationFunction::None,
            "down0Transform");

        // 3. Load Weights (No BatchNorm fusion for transform layers)
        TorchParamReader reader =
            TorchParamReader(resourceBase.resolveResource("debug_dump/down0_transform.bin"));
        SLANG_RETURN_ON_FAIL(transformKernel.loadParams(reader, false));

        // 4. Load Input
        // Input size should be 32x32x128
        List<float> inputImageData = loadRawFloats("debug_dump/down0_transform_input.bin");
        auto inputImage = gInferencingCtx->createPersistentBuffer(
            inputImageData.getBuffer(),
            inputImageData.getCount() * sizeof(float),
            "down0TransformInput");

        // 5. Execute
        // Input: 32x32. Stride: 2. Padding: 1.
        auto task = gInferencingCtx->createTask();
        auto output = transformKernel.allocateResultBuffer(32, 32, 1, 1);
        transformKernel.queueExecute(task, output, BufferView(inputImage), 32, 32, 1);

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
        Conv2DKernel conv1Kernel =
            Conv2DKernel(gInferencingCtx, 16, 3, 1, 64, 128, ActivationFunction::ReLU, "conv1");
        // 3. Load Weights (With BatchNorm fusion)
        Conv2DLayerParams convParams;
        {
            TorchParamReader reader =
                TorchParamReader(resourceBase.resolveResource("debug_dump/down0_conv1.bin"));
            reader.readConv2DLayer(64, 128, 3, convParams);
        }
        BatchNorm2DLayerParams bnParams;
        {
            TorchParamReader reader =
                TorchParamReader(resourceBase.resolveResource("debug_dump/down0_bn1.bin"));
            reader.readBatchNorm2DLayer(128, bnParams);
        }
        convParams.fuseBatchNorm(bnParams);
        SLANG_RETURN_ON_FAIL(conv1Kernel.loadParams(
            3,
            128,
            convParams.weights.getBuffer(),
            convParams.biases.getBuffer()));
        // 4. Load Input
        // Input size should be 32x32x64
        List<float> inputImageData = loadRawFloats("debug_dump/down0_conv1_input.bin");
        auto inputImage = gInferencingCtx->createPersistentBuffer(
            inputImageData.getBuffer(),
            inputImageData.getCount() * sizeof(float),
            "down0Conv1Input");
        // 5. Execute
        auto task = gInferencingCtx->createTask();
        auto output = conv1Kernel.allocateResultBuffer(32, 32, 1, 1);
        conv1Kernel.queueExecute(task, output, BufferView(inputImage), 32, 32, 1);
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
        auto inputABuffer = gInferencingCtx->createPersistentBuffer(
            inputAData.getBuffer(),
            inputAData.getCount() * sizeof(float),
            "broadcastAddInputA");
        List<float> inputBData = loadRawFloats("debug_dump/down0_time_proj_output.bin");
        auto inputBBuffer = gInferencingCtx->createPersistentBuffer(
            inputBData.getBuffer(),
            inputBData.getCount() * sizeof(float),
            "broadcastAddInputB");
        // 4. Execute
        auto task = gInferencingCtx->createTask();
        int shapeA[] = {32, 32, 128};
        int shapeB[] = {1, 1, 128};
        auto output =
            broadcastAddKernel.allocResultBuffer(makeArrayView(shapeA), makeArrayView(shapeB), 1);
        broadcastAddKernel.queueExecute(
            task,
            output,
            BufferView(inputABuffer),
            makeArrayView(shapeA),
            BufferView(inputBBuffer),
            makeArrayView(shapeB));
        task.execute();
        // 5. Verify
        TEST_CHECK("testBroadcastAdd", checkOutput(output, expectedOutput));
        return SLANG_OK;
    }

    SlangResult testDown0()
    {
        UNetModel model = UNetModel(gInferencingCtx, 1, 1);
        SLANG_RETURN_ON_FAIL(loadModel(model, "model_weights.bin"));

        auto expectedOutput = loadRawFloats("debug_dump/down0_transform_output.bin");
        if (expectedOutput.getCount() == 0)
            return SLANG_FAIL;

        // Verify loaded weights.
        {
            TorchParamReader reader1 =
                TorchParamReader(resourceBase.resolveResource("debug_dump/down0_time_proj.bin"));
            LinearLayerParams linearParams;
            reader1.readLinearLayer(32, 128, linearParams);
            TEST_CHECK(
                "testDown0_timeProjWeights",
                checkOutput(
                    BufferView(model.downBlocks[0]->timeEmbedTransform->weightsBuffer),
                    linearParams.weights));
        }

        List<float> imageInputData = loadRawFloats("debug_dump/down0_conv1_input.bin");
        auto inputImage = gInferencingCtx->createPersistentBuffer(
            imageInputData.getBuffer(),
            imageInputData.getCount() * sizeof(float),
            "inputImage");
        List<float> timeEmbedInputData = loadRawFloats("debug_dump/down0_time_proj_input.bin");
        auto timeEmbedInput = gInferencingCtx->createPersistentBuffer(
            timeEmbedInputData.getBuffer(),
            timeEmbedInputData.getCount() * sizeof(float),
            "timeEmbedInput");
        auto task = gInferencingCtx->createTask();
        auto result = model.downBlocks[0]->allocateResultBuffer(32, 32, 1);
        model.downBlocks[0]->queueExecute(
            task,
            result,
            BufferView(inputImage),
            32,
            32,
            1,
            BufferView(timeEmbedInput));
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
        auto buffer = gInferencingCtx->createPersistentBuffer(
            down3Out.getBuffer(),
            down3Out.getCount() * sizeof(float),
            "down3Out");

        // 2. Load Expected Result (The input to UpBlock0)
        auto expected = loadRawFloats("debug_dump/up0_concat_output.bin");

        // 3. Execute Concat (Axis 2 = Channels)
        ConcatKernel concat(gInferencingCtx, 2);
        auto task = gInferencingCtx->createTask();

        // In this specific model, down3 outputs [2, 2, 1024] (assuming 32x32 -> 2x2)
        // Concatenating two of them -> [2, 2, 2048]
        Shape shapes[] = {{2, 2, 1024}, {2, 2, 1024}};
        BufferView buffers[] = {BufferView(buffer), BufferView(buffer)};
        auto result = concat.allocResultBuffer(makeArrayView(shapes), 2);
        concat.queueExecute(task, result, makeArrayView(buffers), makeArrayView(shapes), 2);

        task.execute();
        TEST_CHECK("testBottleneckConcat", checkOutput(result, expected));
        return SLANG_OK;
    }

    SlangResult testUp0()
    {
        // 1. Input: The Concatenated tensor (2048 channels)
        auto inputData = loadRawFloats("debug_dump/up0_concat_output.bin");
        auto inputBuf = gInferencingCtx->createPersistentBuffer(
            inputData.getBuffer(),
            inputData.getCount() * sizeof(float));

        // 2. Time Input
        auto timeData = loadRawFloats("debug_dump/down0_time_proj_input.bin");
        auto timeBuf = gInferencingCtx->createPersistentBuffer(
            timeData.getBuffer(),
            timeData.getCount() * sizeof(float));

        // 3. Setup Block
        // InChannels=1024, OutChannels=512.
        // Note: The block constructor automatically doubles inChannels for conv1 logic (1024*2 =
        // 2048).
        UNetModel model = UNetModel(gInferencingCtx, 1, 1);
        SLANG_RETURN_ON_FAIL(loadModel(model, "model_weights.bin"));

        auto expectedOutput = loadRawFloats("debug_dump/down0_transform_output.bin");
        if (expectedOutput.getCount() == 0)
            return SLANG_FAIL;
        auto& upBlock = *(model.upBlocks[0]);

        // 4. Execute
        auto task = gInferencingCtx->createTask();
        // Input size: 2x2. Output should be 4x4.
        auto result = upBlock.allocateResultBuffer(2, 2, 1);
        upBlock.queueExecute(task, result, BufferView(inputBuf), 2, 2, 1, BufferView(timeBuf));

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
        SLANG_RETURN_ON_FAIL(loadModel(model, "model_weights.bin"));

        // 2. Load Inputs
        List<float> inputData = loadRawFloats("debug_dump/step495_input.bin");
        if (inputData.getCount() == 0)
            return SLANG_FAIL;

        auto inputImage = gInferencingCtx->createPersistentBuffer(inputData, "step495_input");

        // 3. Run Forward Pass
        int t = 495;
        auto task = gInferencingCtx->createTask();

        // Note: ensure input width/height matches Python (32)
        auto result = gInferencingCtx->createPersistentBuffer(inputData, "step495_output");
        model.queueExecute(task, BufferView(result), BufferView(inputImage), 32, 32, t, 1);

        task.execute(); // Or just let debug mode handle it

        // 4. Verify
        auto expected = loadRawFloats("debug_dump/step495_output.bin");
        TEST_CHECK("testStep495", checkOutput(result, expected));

        return SLANG_OK;
    }
};

int main(int argc, char** argv)
{
    SimpleUNetProgram app;
    if (SLANG_FAILED(app.execute(argc, argv)))
    {
        return -1;
    }
    return 0;
}
