// Unit test for kernels

#include "core/slang-basic.h"
#include "example-base/example-base.h"
#include "slang-rhi/shader-cursor.h"
#include "external/slang-rhi/include/slang-rhi.h"
#include "slang-com-ptr.h"
#include "slang.h"

#include <string>

using Slang::ComPtr;

static const ExampleResources resourceBase("unit-test");


struct Kernel
{
    ComPtr<rhi::IShaderProgram> program;
    ComPtr<rhi::IComputePipeline> pipeline;
    operator bool() { return program && pipeline; }
};

template<int kernelSize, int inChannels, int outChannels>
struct Conv2D
{
    float weights[kernelSize * kernelSize * inChannels * outChannels];
    float biases[outChannels];
};

template<int kernelSize, int inChannels, int outChannels>
struct ConvTransposed2D
{
    float weights[kernelSize * kernelSize * inChannels * outChannels];
    float biases[outChannels];
};

template<int kernelSize, int inChannels, int outChannels>
struct SimpleConvolutionParams
{
    rhi::DeviceAddress inputImage;
    rhi::DeviceAddress outputImage;
    int inputImageWidth;
    int inputImageHeight;
    int outputImageWidth;
    int stride;
    int padding;
    Conv2D<kernelSize, inChannels, outChannels> convLayer;
};

template<int kernelSize, int inChannels, int outChannels>
struct SimpleTransposedConvolutionParams
{
    rhi::DeviceAddress inputImage;
    rhi::DeviceAddress outputImage;
    int inputImageWidth;
    int inputImageHeight;
    int outputImageWidth;
    int stride;
    int padding;
    ConvTransposed2D<kernelSize, inChannels, outChannels> transposedConvLayer;
};

struct UnitTestProgram : public TestBase
{
    ComPtr<rhi::IDevice> gDevice;

    ComPtr<slang::ISession> gSlangSession;
    ComPtr<slang::IModule> gSlangModule;
    Kernel gConvolutionProgram;
    Kernel gTransposedConvolutionProgram;

    SlangResult execute(int argc, char* argv[])
    {
        parseOption(argc, argv);
        static const int v = sizeof(SimpleConvolutionParams<3, 1, 1>);
        rhi::DeviceDesc deviceDesc;
        deviceDesc.slang.targetProfile = "spirv_1_6";
        deviceDesc.deviceType = rhi::DeviceType::Vulkan;
        gDevice = rhi::getRHI()->createDevice(deviceDesc);
        if (!gDevice)
            return SLANG_FAIL;

        SLANG_RETURN_ON_FAIL(loadShaderKernels());

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
                           16,17,18,19,20 };
        auto readInput = [&](int x, int y) { return inputData[y * 5 + x]; };
        auto inputBuffer = createBuffer(5 * 5 * sizeof(float), inputData);
        auto outputBuffer = createBuffer(inputBuffer->getDesc().size);
        SimpleConvolutionParams<3, 1, 1> params;
        params.inputImage = inputBuffer->getDeviceAddress();
        params.outputImage = outputBuffer->getDeviceAddress();
        params.inputImageHeight = 5;
        params.inputImageWidth = 5;
        params.outputImageWidth = 5;
        params.padding = 1;
        params.stride = 1;
        params.convLayer.biases[0] = 1000.0f;
        float convWeights[9] = { 0.1, 0.5, 0.2,
                                0.5, 1.0, 0.5,
                                0.2, 0.5, 0.4 };
        auto readWeight = [&](int x, int y) {return convWeights[y * 3 + x]; };
        memcpy(params.convLayer.weights, convWeights, sizeof(convWeights));
        renderDocBeginFrame();
        dispatchKernel(gConvolutionProgram, params, 1, 1);
        float outputData[25];
        gDevice->readBuffer(outputBuffer, 0, sizeof(outputData), outputData);
        renderDocEndFrame();
        float v0 = outputData[0];
        float expectedV0 = readInput(0, 0) * readWeight(1, 1) + readInput(1, 0) * readWeight(2, 1) +
            readInput(0, 1) * readWeight(2, 1) + readInput(1, 1) * readWeight(2, 2) + params.convLayer.biases[0];
        TEST_CHECK("simpleConvolution", fabs(v0 - expectedV0) < 1e-3f);
        return SLANG_OK;
    }

    SlangResult testSimpleTransposedConvolution()
    {
        float inputData[] = { 1,2,3,4,5,
                           6,7,8,9,10,
                           11,12,13,14,15,
                           16,17,18,19,20 };
        auto readInput = [&](int x, int y) { return inputData[y * 5 + x]; };
        auto inputBuffer = createBuffer(5 * 5 * sizeof(float), inputData);
        auto outputBuffer = createBuffer(inputBuffer->getDesc().size);
        SimpleTransposedConvolutionParams<3, 1, 1> params;
        params.inputImage = inputBuffer->getDeviceAddress();
        params.outputImage = outputBuffer->getDeviceAddress();
        params.inputImageHeight = 5;
        params.inputImageWidth = 5;
        params.outputImageWidth = 5;
        params.padding = 1;
        params.stride = 1;
        params.transposedConvLayer.biases[0] = 1000.0f;
        float convWeights[9] = { 0.1, 0.5, 0.2,
                                0.5, 1.0, 0.5,
                                0.2, 0.5, 0.4 };
        auto readWeight = [&](int x, int y) {return convWeights[y * 3 + x]; };
        memcpy(params.transposedConvLayer.weights, convWeights, sizeof(convWeights));
        renderDocBeginFrame();
        dispatchKernel(gTransposedConvolutionProgram, params, 1, 1);
        float outputData[25];
        gDevice->readBuffer(outputBuffer, 0, sizeof(outputData), outputData);
        renderDocEndFrame();
        float v0 = outputData[0];
        float expectedV0 = readInput(1, 0) * readWeight(1, 1) + readInput(0, 1) * readWeight(1, 0) +
            readInput(1, 1) * readWeight(0, 0) + params.transposedConvLayer.biases[0];
        TEST_CHECK("simpleTransposedConvolution", fabs(v0 - expectedV0) < 1e-3f);
        return SLANG_OK;
    }

    template<typename Args>
    void dispatchKernel(Kernel& kernel, Args& args, size_t numWorkGroupsX, size_t numWorkGroupsY)
    {
        auto queue = gDevice->getQueue(rhi::QueueType::Graphics);
        ComPtr<rhi::ICommandEncoder> encoder;
        queue->createCommandEncoder(encoder.writeRef());
        {
            auto computeEncoder = encoder->beginComputePass();
            auto rootShaderObject = computeEncoder->bindPipeline(kernel.pipeline.get());
            rhi::ShaderCursor cursor(rootShaderObject->getEntryPoint(0));
            auto bufferObj = gDevice->createShaderObject(cursor["params"].getTypeLayout()->getElementTypeLayout()->getType());
            bufferObj->setData(rhi::ShaderOffset(), &args, sizeof(Args));
            cursor["params"].setObject(bufferObj);
            computeEncoder->dispatchCompute(numWorkGroupsX, numWorkGroupsY, 1);
            computeEncoder->end();
        }
        ComPtr<rhi::ICommandBuffer> commandBuffer;
        encoder->finish(commandBuffer.writeRef());
        queue->submit(commandBuffer);
    }

    // Create a buffer with the specified size and optional initial data.
    ComPtr<rhi::IBuffer> createBuffer(size_t size, void* initData = nullptr)
    {
        rhi::BufferDesc bufferDesc = {};
        bufferDesc.size = size;
        bufferDesc.defaultState = rhi::ResourceState::UnorderedAccess;
        bufferDesc.usage = rhi::BufferUsage::CopySource | rhi::BufferUsage::CopyDestination |
                           rhi::BufferUsage::UnorderedAccess;
        bufferDesc.memoryType = rhi::MemoryType::DeviceLocal;
        return gDevice->createBuffer(bufferDesc, initData);
    }

    void clearBuffer(rhi::IBuffer* buffer)
    {
        auto queue = gDevice->getQueue(rhi::QueueType::Graphics);
        auto encoder = queue->createCommandEncoder();
        encoder->clearBuffer(buffer);
        auto cmdBuffer = encoder->finish();
        queue->submit(cmdBuffer);
    }

    Kernel loadComputeProgram(slang::IModule* slangModule, char const* entryPointName, slang::SpecializationArg* specArgs, int specArgsCount)
    {
        ComPtr<ISlangBlob> diagnosticBlob;
        ComPtr<slang::IEntryPoint> entryPoint;
        slangModule->findAndCheckEntryPoint(entryPointName, SLANG_STAGE_COMPUTE, entryPoint.writeRef(), diagnosticBlob.writeRef());
        diagnoseIfNeeded(diagnosticBlob);

        ComPtr<slang::IComponentType> linkedProgram;
        ComPtr<slang::IComponentType> specializedComponentType;
        if (specArgsCount)
            entryPoint->specialize(specArgs, specArgsCount, specializedComponentType.writeRef(), diagnosticBlob.writeRef());
        else
            specializedComponentType = entryPoint;

        diagnoseIfNeeded(diagnosticBlob);

        specializedComponentType->link(linkedProgram.writeRef());

        if (isTestMode())
        {
            printEntrypointHashes(1, 1, linkedProgram);
        }

        Kernel result;

        rhi::ComputePipelineDesc desc;
        auto program = gDevice->createShaderProgram(linkedProgram);
        desc.program = program.get();
        result.program = program;
        result.pipeline = gDevice->createComputePipeline(desc);
        return result;
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

    ComPtr<slang::ISession> createSlangSession(rhi::IDevice* device)
    {
        ComPtr<slang::ISession> slangSession = device->getSlangSession();
        return slangSession;
    }

    ComPtr<slang::IModule> compileShaderModuleFromFile(
        slang::ISession* slangSession,
        char const* filePath)
    {
        ComPtr<slang::IModule> slangModule;
        ComPtr<slang::IBlob> diagnosticBlob;
        Slang::String path = resourceBase.resolveResource(filePath);
        slangModule = slangSession->loadModule(path.getBuffer(), diagnosticBlob.writeRef());
        diagnoseIfNeeded(diagnosticBlob);

        return slangModule;
    }

    SlangResult loadShaderKernels()
    {
        Slang::String path = resourceBase.resolveResource("diffusion.slang");

        gSlangSession = createSlangSession(gDevice);
        gSlangModule = compileShaderModuleFromFile(gSlangSession, path.getBuffer());
        if (!gSlangModule)
            return SLANG_FAIL;

        slang::SpecializationArg specArgs[] = {
            slang::SpecializationArg::fromExpr("4"), // tile size
            slang::SpecializationArg::fromExpr("3"),
            slang::SpecializationArg::fromExpr("1"),
            slang::SpecializationArg::fromExpr("1")};

        gConvolutionProgram = loadComputeProgram(gSlangModule, "simpleConvolution", specArgs,
            SLANG_COUNT_OF(specArgs));
        if (!gConvolutionProgram)
            return SLANG_FAIL;
        
        slang::SpecializationArg specArgs2[] = {
            slang::SpecializationArg::fromExpr("3"), // tile size
            slang::SpecializationArg::fromExpr("3"),
            slang::SpecializationArg::fromExpr("1"),
            slang::SpecializationArg::fromExpr("1"),
            slang::SpecializationArg::fromExpr("1") };
        gTransposedConvolutionProgram = loadComputeProgram(gSlangModule, "simpleTransposedConvolution", specArgs2,
            SLANG_COUNT_OF(specArgs2));
        if (!gTransposedConvolutionProgram)
            return SLANG_FAIL;
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
