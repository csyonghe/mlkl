// An example showing how to build a simple unet model to produce
// 32x32 MNIST digit images unconditioned in a diffusion process.

#include "../shared/ddim-sampler.h"
#include "core/slang-basic.h"
#include "example-base/example-base.h"
#include "inference-context.h"
#include "kernels.h"
#include "simple-unet.h"
#include "test-unet-model.h"
#include "torch-reader.h"

#include <chrono>
#include <cmath>
#include <cstring>
#include <random>

static const ExampleResources resourceBase("simple-unet");

struct SimpleUNetProgram : public TestBase
{
    ComPtr<rhi::IDevice> gDevice;

    RefPtr<InferencingContext> gInferencingCtx;

    SlangResult execute(int argc, char* argv[])
    {
        parseOption(argc, argv);

        gInferencingCtx = new InferencingContext();

        // Check for comparison test flags
        for (int i = 1; i < argc; i++)
        {
            if (strcmp(argv[i], "--compare-up0") == 0)
            {
                printf("Running up block 0 comparison test...\n\n");
                return testUpBlock0(gInferencingCtx);
            }
            if (strcmp(argv[i], "--compare-down0") == 0)
            {
                printf("Running down block 0 comparison test...\n\n");
                return testDownBlock0(gInferencingCtx);
            }
            if (strcmp(argv[i], "--compare-conv0") == 0)
            {
                printf("Running initial conv (conv0) comparison test...\n\n");
                return testInitialConv(gInferencingCtx);
            }
            if (strcmp(argv[i], "--compare-time-embed") == 0)
            {
                printf("Running time embedding comparison test...\n\n");
                return testTimeEmbedding(gInferencingCtx);
            }
            if (strcmp(argv[i], "--compare") == 0)
            {
                printf("Running full UNet PyTorch comparison test...\n\n");
                return testUNetModelAgainstPyTorch(gInferencingCtx);
            }
        }

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
        uint32_t seed = 171;
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
        SLANG_RETURN_ON_FAIL(loadModel(model, "model_weights.bin"));

        uint32_t imageSize = 32;
        int outputChannelCount = 1;
        int inputChannelCount = 1;

        List<float> inputImageData;
        initImage(inputImageData, imageSize, imageSize, inputChannelCount);

        int trainingSteps = 500;
        int inferenceSteps = 50;
        // Linear schedule with default beta range
        DiffusionSchedule schedule(trainingSteps, 0.0001f, 0.02f, /*scaled_linear=*/false);
        DDIMSampler sampler(gInferencingCtx, std::move(schedule), inferenceSteps);

        auto imageAStorage = gInferencingCtx->createTensor(
            ElementType::Float32,
            Shape(1, imageSize, imageSize, inputChannelCount),
            inputImageData.getCount() * sizeof(float),
            inputImageData.getBuffer(),
            "imageA");
        auto imageA = imageAStorage->getView();
        auto imageB = gInferencingCtx->allocScratchTensor(
            ElementType::Float32,
            Shape(1, imageSize, imageSize, inputChannelCount),
            "imageB");
        auto outputImage = imageA;
        static const int largePrime = 15485863;
        renderDocBeginFrame();
        auto task = gInferencingCtx->createTask();
        gInferencingCtx->pushAllocScope();
        SLANG_DEFER(gInferencingCtx->popAllocScope());

        auto predictedNoise = gInferencingCtx->allocScratchTensor(
            ElementType::Float32,
            Shape(1, imageSize, imageSize, inputChannelCount),
            "predictedNoise");
        for (int step = 0; step < inferenceSteps; step++)
        {
            // Use 't' for the model, but 'step' for the loop logic
            int t = sampler.timesteps[step];
            model.queueExecute(task, predictedNoise, imageA, t);

            sampler.step(task, imageB, imageA, predictedNoise, step);

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
        printf(
            "Output tensor shape: [%d, %d, %d, %d], element count: %zu\n",
            outputImage.shape[0],
            outputImage.shape[1],
            outputImage.shape[2],
            outputImage.shape[3],
            outputImage.shape.getElementCount());
        printf(
            "Expected: [1, %d, %d, %d], element count: %zu\n",
            imageSize,
            imageSize,
            outputChannelCount,
            (size_t)(imageSize * imageSize * outputChannelCount));

        List<float> outputImageData = gInferencingCtx->readBuffer<float>(outputImage);
        printf("Read %zu floats from output buffer\n", outputImageData.getCount());

        // Save to disk as png
        writeImagePNG("output.png", imageSize, imageSize, outputChannelCount, outputImageData);
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
