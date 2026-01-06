/**
 * Test UNet model by comparing C++ output against PyTorch reference.
 *
 * Usage:
 * 1. Run train.py to generate model_weights.bin and model_weights.pth
 * 2. Run dump_unet_reference.py to generate ref_input.bin, ref_output.bin, etc.
 * 3. Run this test to compare C++ output against PyTorch reference
 */

#include "test-unet-model.h"

#include "core/slang-basic.h"
#include "core/slang-io.h"
#include "example-base/example-base.h"
#include "inference-context.h"
#include "kernels.h"
#include "simple-unet.h"
#include "time-embed.h"
#include "torch-reader.h"

#include <cmath>
#include <fstream>
#include <vector>

// Helper to load a binary file of floats
std::vector<float> loadFloatBin(String inFilepath)
{
    const ExampleResources resourceBase("simple-unet");
    std::string filepath = resourceBase.resolveResource(inFilepath.getBuffer()).getBuffer();

    std::ifstream file(filepath, std::ios::binary | std::ios::ate);
    if (!file.is_open())
    {
        printf("ERROR: Could not open file: %s\n", filepath.c_str());
        return {};
    }

    std::streamsize size = file.tellg();
    file.seekg(0, std::ios::beg);

    std::vector<float> data(size / sizeof(float));
    if (!file.read(reinterpret_cast<char*>(data.data()), size))
    {
        printf("ERROR: Could not read file: %s\n", filepath.c_str());
        return {};
    }

    return data;
}

// Helper to load a binary file of int32
std::vector<int32_t> loadInt32Bin(String inFilepath)
{
    const ExampleResources resourceBase("simple-unet");
    std::string filepath = resourceBase.resolveResource(inFilepath.getBuffer()).getBuffer();

    std::ifstream file(filepath, std::ios::binary | std::ios::ate);
    if (!file.is_open())
    {
        printf("ERROR: Could not open file: %s\n", filepath.c_str());
        return {};
    }

    std::streamsize size = file.tellg();
    file.seekg(0, std::ios::beg);

    std::vector<int32_t> data(size / sizeof(int32_t));
    if (!file.read(reinterpret_cast<char*>(data.data()), size))
    {
        printf("ERROR: Could not read file: %s\n", filepath.c_str());
        return {};
    }

    return data;
}

// Compute statistics of a float array
void printStats(const char* name, const float* data, size_t count)
{
    if (count == 0)
        return;

    float minVal = data[0], maxVal = data[0], sum = 0;
    int nanCount = 0, infCount = 0;

    for (size_t i = 0; i < count; i++)
    {
        if (std::isnan(data[i]))
            nanCount++;
        else if (std::isinf(data[i]))
            infCount++;
        else
        {
            minVal = std::min(minVal, data[i]);
            maxVal = std::max(maxVal, data[i]);
            sum += data[i];
        }
    }

    printf("  %s: min=%.4f, max=%.4f, mean=%.4f", name, minVal, maxVal, sum / count);
    if (nanCount > 0)
        printf(", NaN=%d", nanCount);
    if (infCount > 0)
        printf(", Inf=%d", infCount);
    printf("\n");
}

// Compare two float arrays and report differences
bool compareArrays(
    const char* name,
    const float* ref,
    const float* test,
    size_t count,
    float tolerance = 1e-3f)
{
    float maxAbsDiff = 0;
    float maxRelDiff = 0;
    size_t maxDiffIdx = 0;
    int mismatchCount = 0;

    for (size_t i = 0; i < count; i++)
    {
        float absDiff = std::abs(ref[i] - test[i]);
        float relDiff = (std::abs(ref[i]) > 1e-6f) ? absDiff / std::abs(ref[i]) : absDiff;

        if (absDiff > maxAbsDiff)
        {
            maxAbsDiff = absDiff;
            maxRelDiff = relDiff;
            maxDiffIdx = i;
        }

        if (absDiff > tolerance)
        {
            mismatchCount++;
        }
    }

    printf("\n%s Comparison:\n", name);
    printf(
        "  Max absolute diff: %.6f at index %zu (ref=%.4f, test=%.4f)\n",
        maxAbsDiff,
        maxDiffIdx,
        ref[maxDiffIdx],
        test[maxDiffIdx]);
    printf("  Max relative diff: %.6f\n", maxRelDiff);
    printf(
        "  Mismatches (>%.1e): %d / %zu (%.2f%%)\n",
        tolerance,
        mismatchCount,
        count,
        100.0f * mismatchCount / count);

    bool passed = (maxAbsDiff < tolerance);
    printf("  Result: %s\n", passed ? "PASSED" : "FAILED");

    return passed;
}

SlangResult testUpBlock0(InferencingContext* ctx)
{
    const ExampleResources resourceBase("simple-unet");

    printf("========================================\n");
    printf("Up Block 0 Test - C++ vs PyTorch\n");
    printf("========================================\n\n");

    // Load reference data
    printf("Loading reference data...\n");
    auto refBeforeUp0 = loadFloatBin("ref_before_up0.bin");
    auto refUp0AfterConcat = loadFloatBin("ref_up0_after_concat.bin");
    auto refUp0AfterConv1 = loadFloatBin("ref_up0_after_conv1.bin");
    auto refAfterUp0 = loadFloatBin("ref_after_up0.bin");
    auto refTimeEmbed = loadFloatBin("ref_time_embed.bin");

    if (refBeforeUp0.empty() || refAfterUp0.empty())
    {
        printf("ERROR: Could not load reference files. Run dump_unet_reference.py first.\n");
        return SLANG_FAIL;
    }

    printf("  Ref before up0 (bottleneck): %zu floats\n", refBeforeUp0.size());
    printf("  Ref up0 after concat: %zu floats\n", refUp0AfterConcat.size());
    printf("  Ref up0 after conv1: %zu floats\n", refUp0AfterConv1.size());
    printf("  Ref after up0: %zu floats\n", refAfterUp0.size());

    printStats("Ref Before Up0", refBeforeUp0.data(), refBeforeUp0.size());
    printStats("Ref After Concat", refUp0AfterConcat.data(), refUp0AfterConcat.size());
    printStats("Ref After Up0", refAfterUp0.data(), refAfterUp0.size());

    // Calculate dimensions
    // After 4 down blocks: 32 -> 16 -> 8 -> 4 -> 2
    const int bottleneckSize = 2;
    const int bottleneckChannels = 1024;
    const int timeEmbedDim = 32;

    // Load full UNet model
    printf("\nLoading C++ UNet model...\n");
    UNetModel model(ctx, 1, 1);

    auto weightsPath = resourceBase.resolveResource("model_weights.bin");
    Slang::RefPtr<FileStream> fileStream = new FileStream();
    if (!File::exists(weightsPath))
    {
        printf("ERROR: Model weights not found at %s\n", weightsPath.getBuffer());
        return SLANG_FAIL;
    }
    SLANG_RETURN_ON_FAIL(fileStream->init(weightsPath, FileMode::Open));
    TorchParamReader reader(fileStream);
    SLANG_RETURN_ON_FAIL(model.loadParams(reader));
    printf("  Model loaded successfully.\n");

    // Create input tensors from reference data
    // Bottleneck has shape [1, 2, 2, 1024] in NHWC
    auto bottleneckTensor = ctx->createTensor(
        ElementType::Float32,
        Shape(1, bottleneckSize, bottleneckSize, bottleneckChannels),
        refBeforeUp0.size() * sizeof(float),
        refBeforeUp0.data(),
        "bottleneck");

    // Time embedding
    auto timeEmbedTensor = ctx->createTensor(
        ElementType::Float32,
        Shape(1, timeEmbedDim),
        refTimeEmbed.size() * sizeof(float),
        refTimeEmbed.data(),
        "timeEmbed");

    // First, test the concat operation
    printf("\nTesting concat operation...\n");

    // In the UNet, for up0, we concat x with skipConnections[3] which is the same as x
    Shape concatShape = {1, bottleneckSize, bottleneckSize, bottleneckChannels};
    Shape shapes[] = {concatShape, concatShape};
    TensorView buffers[] = {bottleneckTensor->getView(), bottleneckTensor->getView()};

    auto concatResult =
        model.concat->allocateResultBuffer(ElementType::Float32, makeArrayView(shapes), 3);
    
    auto task = ctx->createTask();
    model.concat->queueExecute(task, concatResult, makeArrayView(buffers), 3);
    task.execute();

    Slang::List<float> cppConcat = ctx->readBuffer<float>(concatResult);
    printf("  C++ concat output: %zu floats\n", cppConcat.getCount());
    printf(
        "  C++ concat shape: [%d, %d, %d, %d]\n",
        concatResult.shape[0],
        concatResult.shape[1],
        concatResult.shape[2],
        concatResult.shape[3]);
    printStats("C++ After Concat", cppConcat.getBuffer(), cppConcat.getCount());

    bool concatPassed = compareArrays(
        "Concat",
        refUp0AfterConcat.data(),
        cppConcat.getBuffer(),
        std::min(refUp0AfterConcat.size(), (size_t)cppConcat.getCount()),
        0.01f);

    if (!concatPassed)
    {
        printf("\nConcat failed! First 20 values:\n");
        printf("  Index |    PyTorch |       C++ |      Diff\n");
        printf("  ------|------------|-----------|----------\n");
        for (int i = 0; i < 20 && i < (int)refUp0AfterConcat.size(); i++)
        {
            float diff = cppConcat[i] - refUp0AfterConcat[i];
            printf("  %5d | %10.4f | %9.4f | %9.4f\n", i, refUp0AfterConcat[i], cppConcat[i], diff);
        }
    }

    // Now test the full up block
    printf("\nTesting full up block 0...\n");
    auto& upBlock0 = model.upBlocks[0];

    // Use the concat result as input to up block
    // But first, create a proper concat tensor from reference
    auto concatTensor = ctx->createTensor(
        ElementType::Float32,
        Shape(1, bottleneckSize, bottleneckSize, 2 * bottleneckChannels),
        refUp0AfterConcat.size() * sizeof(float),
        refUp0AfterConcat.data(),
        "up0ConcatInput");

    auto upOutput = upBlock0->allocateResultBuffer(
        ElementType::Float32,
        bottleneckSize, // width
        bottleneckSize, // height
        1);             // batchSize

    task = ctx->createTask();
    upBlock0->queueExecute(task, upOutput, concatTensor->getView(), timeEmbedTensor->getView());
    task.execute();

    Slang::List<float> cppUpOutput = ctx->readBuffer<float>(upOutput);
    printf("  C++ up0 output: %zu floats\n", cppUpOutput.getCount());
    printf(
        "  C++ up0 output shape: [%d, %d, %d, %d]\n",
        upOutput.shape[0],
        upOutput.shape[1],
        upOutput.shape[2],
        upOutput.shape[3]);
    printStats("C++ After Up0", cppUpOutput.getBuffer(), cppUpOutput.getCount());

    bool upBlockPassed = compareArrays(
        "Up Block 0",
        refAfterUp0.data(),
        cppUpOutput.getBuffer(),
        std::min(refAfterUp0.size(), (size_t)cppUpOutput.getCount()),
        0.01f);

    // Print first few values
    printf("\nFirst 20 up0 output values:\n");
    printf("  Index |    PyTorch |       C++ |      Diff\n");
    printf("  ------|------------|-----------|----------\n");
    for (int i = 0; i < 20 && i < (int)refAfterUp0.size(); i++)
    {
        float diff = cppUpOutput[i] - refAfterUp0[i];
        const char* marker = (std::abs(diff) > 0.01f) ? " <-- DIFF" : "";
        printf(
            "  %5d | %10.4f | %9.4f | %9.4f%s\n",
            i,
            refAfterUp0[i],
            cppUpOutput[i],
            diff,
            marker);
    }

    bool passed = concatPassed && upBlockPassed;
    printf("\n========================================\n");
    printf("Up Block 0 Test %s\n", passed ? "PASSED" : "FAILED");
    printf("  Concat: %s\n", concatPassed ? "PASSED" : "FAILED");
    printf("  Up Block: %s\n", upBlockPassed ? "PASSED" : "FAILED");
    printf("========================================\n");

    return passed ? SLANG_OK : SLANG_FAIL;
}

SlangResult testDownBlock0(InferencingContext* ctx)
{
    const ExampleResources resourceBase("simple-unet");

    printf("========================================\n");
    printf("Down Block 0 Test - C++ vs PyTorch\n");
    printf("========================================\n\n");

    // Load reference data
    printf("Loading reference data...\n");
    auto refAfterConv0 = loadFloatBin("ref_after_conv0.bin");
    auto refTimeEmbed = loadFloatBin("ref_time_embed.bin");
    auto refDown0AfterConv1 = loadFloatBin("ref_down0_after_conv1.bin");
    auto refDown0TimeProj = loadFloatBin("ref_down0_time_proj.bin");
    auto refDown0AfterAdd = loadFloatBin("ref_down0_after_add.bin");
    auto refDown0AfterConv2 = loadFloatBin("ref_down0_after_conv2.bin");
    auto refAfterDown0 = loadFloatBin("ref_after_down0.bin");

    if (refAfterConv0.empty() || refAfterDown0.empty())
    {
        printf("ERROR: Could not load reference files. Run dump_unet_reference.py first.\n");
        return SLANG_FAIL;
    }

    printf("  Ref after conv0: %zu floats\n", refAfterConv0.size());
    printf("  Ref time embed: %zu floats\n", refTimeEmbed.size());
    printf("  Ref down0 after conv1: %zu floats\n", refDown0AfterConv1.size());
    printf("  Ref down0 time proj: %zu floats\n", refDown0TimeProj.size());
    printf("  Ref down0 after add: %zu floats\n", refDown0AfterAdd.size());
    printf("  Ref down0 after conv2: %zu floats\n", refDown0AfterConv2.size());
    printf("  Ref after down0: %zu floats\n", refAfterDown0.size());

    printStats("Ref After Conv0", refAfterConv0.data(), refAfterConv0.size());
    printStats("Ref Time Embed", refTimeEmbed.data(), refTimeEmbed.size());
    printStats("Ref After Down0", refAfterDown0.data(), refAfterDown0.size());

    // Load full UNet model (we need it to load weights in correct order)
    printf("\nLoading C++ UNet model...\n");
    UNetModel model(ctx, 1, 1);

    auto weightsPath = resourceBase.resolveResource("model_weights.bin");
    Slang::RefPtr<FileStream> fileStream = new FileStream();
    if (!File::exists(weightsPath))
    {
        printf("ERROR: Model weights not found at %s\n", weightsPath.getBuffer());
        return SLANG_FAIL;
    }
    SLANG_RETURN_ON_FAIL(fileStream->init(weightsPath, FileMode::Open));
    TorchParamReader reader(fileStream);
    SLANG_RETURN_ON_FAIL(model.loadParams(reader));
    printf("  Model loaded successfully.\n");

    // Create input tensors from reference data
    const int imageSize = 32;
    const int conv0OutChannels = 64;
    const int timeEmbedDim = 32;

    // Input to down block 0 is output of conv0
    auto inputTensor = ctx->createTensor(
        ElementType::Float32,
        Shape(1, imageSize, imageSize, conv0OutChannels),
        refAfterConv0.size() * sizeof(float),
        refAfterConv0.data(),
        "down0Input");

    // Time embedding
    auto timeEmbedTensor = ctx->createTensor(
        ElementType::Float32,
        Shape(1, timeEmbedDim),
        refTimeEmbed.size() * sizeof(float),
        refTimeEmbed.data(),
        "timeEmbed");

    // Get the first down block
    auto& downBlock0 = model.downBlocks[0];

    // Allocate output and execute
    printf("\nRunning C++ down block 0...\n");
    auto outputTensor = downBlock0->allocateResultBuffer(
        ElementType::Float32,
        imageSize,  // width
        imageSize,  // height
        1);         // batchSize

    auto task = ctx->createTask();
    downBlock0->queueExecute(task, outputTensor, inputTensor->getView(), timeEmbedTensor->getView());
    task.execute();

    // Read back C++ output
    Slang::List<float> cppOutput = ctx->readBuffer<float>(outputTensor);
    printf("  C++ output: %zu floats\n", cppOutput.getCount());
    printf(
        "  C++ output shape: [%d, %d, %d, %d]\n",
        outputTensor.shape[0],
        outputTensor.shape[1],
        outputTensor.shape[2],
        outputTensor.shape[3]);
    printStats("C++ After Down0", cppOutput.getBuffer(), cppOutput.getCount());

    // Compare outputs
    printf("\n========================================\n");
    printf("Comparison Results\n");
    printf("========================================\n");

    bool passed = compareArrays(
        "Down Block 0 Output",
        refAfterDown0.data(),
        cppOutput.getBuffer(),
        std::min(refAfterDown0.size(), (size_t)cppOutput.getCount()),
        0.01f);

    // Print first few values
    printf("\nFirst 20 output values:\n");
    printf("  Index |    PyTorch |       C++ |      Diff\n");
    printf("  ------|------------|-----------|----------\n");
    for (int i = 0; i < 20 && i < (int)refAfterDown0.size(); i++)
    {
        float diff = cppOutput[i] - refAfterDown0[i];
        const char* marker = (std::abs(diff) > 0.01f) ? " <-- DIFF" : "";
        printf(
            "  %5d | %10.4f | %9.4f | %9.4f%s\n",
            i,
            refAfterDown0[i],
            cppOutput[i],
            diff,
            marker);
    }

    printf("\n========================================\n");
    printf("Down Block 0 Test %s\n", passed ? "PASSED" : "FAILED");
    printf("========================================\n");

    return passed ? SLANG_OK : SLANG_FAIL;
}

SlangResult testInitialConv(InferencingContext* ctx)
{
    const ExampleResources resourceBase("simple-unet");

    printf("========================================\n");
    printf("Initial Conv (conv0) Test - C++ vs PyTorch\n");
    printf("========================================\n\n");

    // Load reference data
    printf("Loading reference data...\n");
    auto refInput = loadFloatBin("ref_input.bin");
    auto refAfterConv0 = loadFloatBin("ref_after_conv0.bin");

    if (refInput.empty() || refAfterConv0.empty())
    {
        printf("ERROR: Could not load reference files. Run dump_unet_reference.py first.\n");
        return SLANG_FAIL;
    }

    const int imageSize = 32;
    const int inputChannels = 1;
    const int outputChannels = 64; // First channel size in UNet

    printf("  Ref input: %zu floats\n", refInput.size());
    printf("  Ref after conv0: %zu floats\n", refAfterConv0.size());
    printStats("Ref Input", refInput.data(), refInput.size());
    printStats("Ref After Conv0", refAfterConv0.data(), refAfterConv0.size());

    // Create input tensor
    auto inputTensor = ctx->createTensor(
        ElementType::Float32,
        Shape(1, imageSize, imageSize, inputChannels),
        refInput.size() * sizeof(float),
        refInput.data(),
        "testInput");

    // Create and load the initial conv kernel
    printf("\nLoading C++ Conv2DKernel (conv0)...\n");

    // Conv0 in UNet: 3x3 kernel, stride 1, padding 1, input->64 channels
    RefPtr<Conv2DKernel> conv0 = new Conv2DKernel(
        ctx,
        16,             // groupSize
        3,              // kernelSize
        1,              // stride
        inputChannels,  // inChannels
        outputChannels, // outChannels
        kernelOutput(), // no activation (PyTorch conv0 has no activation)
        "conv0");

    auto weightsPath = resourceBase.resolveResource("model_weights.bin");
    Slang::RefPtr<FileStream> fileStream = new FileStream();
    if (!File::exists(weightsPath))
    {
        printf("ERROR: Model weights not found at %s\n", weightsPath.getBuffer());
        return SLANG_FAIL;
    }
    SLANG_RETURN_ON_FAIL(fileStream->init(weightsPath, FileMode::Open));
    TorchParamReader reader(fileStream);

    // Skip time embedding params (they come first in the weight file)
    const int timeEmbedDim = 32;
    TimeEmbedingKernel timeEmbedKernel(ctx, timeEmbedDim);
    SLANG_RETURN_ON_FAIL(timeEmbedKernel.loadParams(reader));

    // Now load conv0 (no batch norm in PyTorch model)
    SLANG_RETURN_ON_FAIL(conv0->loadParams(reader, false)); // false = no batch norm
    printf("  Conv0 loaded successfully.\n");

    // Allocate output and execute
    printf("\nRunning C++ conv0...\n");
    auto outputTensor = conv0->allocateResultBuffer(
        ElementType::Float32,
        imageSize, // width
        imageSize, // height
        1,         // stride (padding multiplier)
        1);        // batchSize

    auto task = ctx->createTask();
    conv0->queueExecute(task, outputTensor, inputTensor->getView(), 1);
    task.execute();

    // Read back C++ output
    Slang::List<float> cppOutput = ctx->readBuffer<float>(outputTensor);
    printf("  C++ output: %zu floats\n", cppOutput.getCount());
    printf(
        "  C++ output shape: [%d, %d, %d, %d]\n",
        outputTensor.shape[0],
        outputTensor.shape[1],
        outputTensor.shape[2],
        outputTensor.shape[3]);
    printStats("C++ After Conv0", cppOutput.getBuffer(), cppOutput.getCount());

    // Compare outputs
    printf("\n========================================\n");
    printf("Comparison Results\n");
    printf("========================================\n");

    bool passed = compareArrays(
        "Conv0 Output",
        refAfterConv0.data(),
        cppOutput.getBuffer(),
        std::min(refAfterConv0.size(), (size_t)cppOutput.getCount()),
        0.01f);

    // Print first few values
    printf("\nFirst 20 output values:\n");
    printf("  Index |    PyTorch |       C++ |      Diff\n");
    printf("  ------|------------|-----------|----------\n");
    for (int i = 0; i < 20 && i < (int)refAfterConv0.size(); i++)
    {
        float diff = cppOutput[i] - refAfterConv0[i];
        const char* marker = (std::abs(diff) > 0.01f) ? " <-- DIFF" : "";
        printf(
            "  %5d | %10.4f | %9.4f | %9.4f%s\n",
            i,
            refAfterConv0[i],
            cppOutput[i],
            diff,
            marker);
    }

    printf("\n========================================\n");
    printf("Initial Conv Test %s\n", passed ? "PASSED" : "FAILED");
    printf("========================================\n");

    return passed ? SLANG_OK : SLANG_FAIL;
}

SlangResult testTimeEmbedding(InferencingContext* ctx)
{
    const ExampleResources resourceBase("simple-unet");

    printf("========================================\n");
    printf("Time Embedding Test - C++ vs PyTorch\n");
    printf("========================================\n\n");

    // Load reference data
    printf("Loading reference data...\n");
    auto refTimestep = loadInt32Bin("ref_timestep.bin");
    auto refSinusoidal = loadFloatBin("ref_sinusoidal_embed.bin");
    auto refAfterLinear = loadFloatBin("ref_time_embed_after_linear.bin");
    auto refTimeEmbed = loadFloatBin("ref_time_embed.bin");

    if (refTimestep.empty() || refTimeEmbed.empty())
    {
        printf("ERROR: Could not load reference files. Run dump_unet_reference.py first.\n");
        return SLANG_FAIL;
    }

    int timestep = refTimestep[0];
    printf("  Timestep: %d\n", timestep);
    printf("  Ref sinusoidal: %zu floats\n", refSinusoidal.size());
    printf("  Ref after linear: %zu floats\n", refAfterLinear.size());
    printf("  Ref time embed (after ReLU): %zu floats\n", refTimeEmbed.size());

    if (!refSinusoidal.empty())
        printStats("Ref Sinusoidal", refSinusoidal.data(), refSinusoidal.size());
    if (!refAfterLinear.empty())
        printStats("Ref After Linear", refAfterLinear.data(), refAfterLinear.size());
    printStats("Ref Time Embed", refTimeEmbed.data(), refTimeEmbed.size());

    // Create and load C++ TimeEmbedingKernel
    printf("\nLoading C++ TimeEmbedingKernel...\n");

    const int timeEmbedDim = 32; // From simple-unet model
    TimeEmbedingKernel timeEmbedKernel(ctx, timeEmbedDim);

    auto weightsPath = resourceBase.resolveResource("model_weights.bin");
    Slang::RefPtr<FileStream> fileStream = new FileStream();
    if (!File::exists(weightsPath))
    {
        printf("ERROR: Model weights not found at %s\n", weightsPath.getBuffer());
        return SLANG_FAIL;
    }
    SLANG_RETURN_ON_FAIL(fileStream->init(weightsPath, FileMode::Open));
    TorchParamReader reader(fileStream);
    SLANG_RETURN_ON_FAIL(timeEmbedKernel.loadParams(reader));
    printf("  TimeEmbedingKernel loaded successfully.\n");

    // Allocate output tensor and execute
    printf("\nRunning C++ time embedding...\n");
    auto outputTensor = timeEmbedKernel.allocateResultBuffer(ElementType::Float32, 1 /*batchSize*/);

    auto task = ctx->createTask();
    timeEmbedKernel.queueExecute(task, outputTensor, (uint32_t)timestep);
    task.execute();

    // Read back C++ output
    Slang::List<float> cppOutput = ctx->readBuffer<float>(outputTensor);
    printf("  C++ output: %zu floats\n", cppOutput.getCount());
    printStats("C++ Time Embed", cppOutput.getBuffer(), cppOutput.getCount());

    // Compare outputs
    printf("\n========================================\n");
    printf("Comparison Results\n");
    printf("========================================\n");

    bool passed = compareArrays(
        "Time Embedding",
        refTimeEmbed.data(),
        cppOutput.getBuffer(),
        std::min(refTimeEmbed.size(), (size_t)cppOutput.getCount()),
        0.001f); // Tighter tolerance for this simple operation

    // Print all values side by side (only 32 values for time embed dim)
    printf("\nAll time embedding values:\n");
    printf("  Index |    PyTorch |       C++ |      Diff\n");
    printf("  ------|------------|-----------|----------\n");
    for (size_t i = 0; i < refTimeEmbed.size() && i < (size_t)cppOutput.getCount(); i++)
    {
        float diff = cppOutput[(int)i] - refTimeEmbed[i];
        const char* marker = (std::abs(diff) > 0.001f) ? " <-- DIFF" : "";
        printf(
            "  %5zu | %10.4f | %9.4f | %9.4f%s\n",
            i,
            refTimeEmbed[i],
            cppOutput[(int)i],
            diff,
            marker);
    }

    printf("\n========================================\n");
    printf("Time Embedding Test %s\n", passed ? "PASSED" : "FAILED");
    printf("========================================\n");

    return passed ? SLANG_OK : SLANG_FAIL;
}

SlangResult testUNetModelAgainstPyTorch(InferencingContext* ctx)
{
    const ExampleResources resourceBase("simple-unet");

    printf("========================================\n");
    printf("UNet Model Test - C++ vs PyTorch\n");
    printf("========================================\n\n");

    // Load reference data from Python
    printf("Loading reference data from Python...\n");

    auto refInput = loadFloatBin("ref_input.bin");
    auto refOutput = loadFloatBin("ref_output.bin");
    auto refTimestep = loadInt32Bin("ref_timestep.bin");
    auto refTimeEmbed = loadFloatBin("ref_time_embed.bin");
    auto refAfterConv0 = loadFloatBin("ref_after_conv0.bin");
    auto refAfterDown0 = loadFloatBin("ref_after_down0.bin");

    if (refInput.empty() || refOutput.empty() || refTimestep.empty())
    {
        printf("ERROR: Could not load reference files. Run dump_unet_reference.py first.\n");
        return SLANG_FAIL;
    }

    printf("  Input: %zu floats\n", refInput.size());
    printf("  Output: %zu floats\n", refOutput.size());
    printf("  Timestep: %d\n", refTimestep[0]);
    printStats("Ref Input", refInput.data(), refInput.size());
    printStats("Ref Output", refOutput.data(), refOutput.size());

    // Load C++ model
    printf("\nLoading C++ UNet model...\n");
    UNetModel model(ctx, 1, 1); // 1 input channel, 1 output channel

    auto weightsPath = resourceBase.resolveResource("model_weights.bin");
    Slang::RefPtr<FileStream> fileStream = new FileStream();
    if (!File::exists(weightsPath))
    {
        printf("ERROR: Model weights not found at %s\n", weightsPath.getBuffer());
        return SLANG_FAIL;
    }
    SLANG_RETURN_ON_FAIL(fileStream->init(weightsPath, FileMode::Open));
    TorchParamReader reader(fileStream);
    SLANG_RETURN_ON_FAIL(model.loadParams(reader));
    printf("  Model loaded successfully.\n");

    // Create input tensor from reference data (already in NHWC format)
    const int imageSize = 32;
    const int channels = 1;

    auto inputTensor = ctx->createTensor(
        ElementType::Float32,
        Shape(1, imageSize, imageSize, channels),
        refInput.size() * sizeof(float),
        refInput.data(),
        "testInput");

    auto outputTensor = ctx->allocScratchTensor(
        ElementType::Float32,
        Shape(1, imageSize, imageSize, channels),
        "testOutput");

    // Run C++ model
    printf("\nRunning C++ UNet forward pass...\n");
    int timestep = refTimestep[0];

    auto task = ctx->createTask();
    model.queueExecute(task, outputTensor, inputTensor->getView(), timestep);
    task.execute();

    // Read back C++ output
    Slang::List<float> cppOutput = ctx->readBuffer<float>(outputTensor);
    printf("  C++ output: %zu floats\n", cppOutput.getCount());
    printStats("C++ Output", cppOutput.getBuffer(), cppOutput.getCount());

    // Compare outputs
    printf("\n========================================\n");
    printf("Comparison Results\n");
    printf("========================================\n");

    bool passed = compareArrays(
        "Full Output",
        refOutput.data(),
        cppOutput.getBuffer(),
        std::min(refOutput.size(), (size_t)cppOutput.getCount()),
        0.01f); // 1% tolerance

    // Print first few values for visual inspection
    printf("\nFirst 10 output values:\n");
    printf("  Index |    PyTorch |       C++ |      Diff\n");
    printf("  ------|------------|-----------|----------\n");
    for (int i = 0; i < 10 && i < (int)refOutput.size(); i++)
    {
        float diff = cppOutput[i] - refOutput[i];
        printf("  %5d | %10.4f | %9.4f | %9.4f\n", i, refOutput[i], cppOutput[i], diff);
    }

    printf("\n========================================\n");
    printf("Test %s\n", passed ? "PASSED" : "FAILED");
    printf("========================================\n");

    return passed ? SLANG_OK : SLANG_FAIL;
}
