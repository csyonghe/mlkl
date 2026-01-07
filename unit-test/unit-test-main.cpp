// Unit test for kernels

#include "core/slang-basic.h"
#include "example-base/example-base.h"
#include "inference-context.h"
#include "kernels.h"
#include "test-kernels.h"
#include "torch-reader.h"

#include <chrono>
#include <random>

static const ExampleResources resourceBase("unit-test");

struct UnitTestProgram : public TestBase
{
    ComPtr<rhi::IDevice> gDevice;

    RefPtr<InferencingContext> gInferencingCtx;

    SlangResult execute(int argc, char* argv[])
    {
        parseOption(argc, argv);
        rhi::DeviceDesc deviceDesc;
        deviceDesc.slang.targetProfile = "spirv_1_6";
        List<slang::CompilerOptionEntry> compilerOptionsEntries;
        const char* capabilities[] = {"spvGroupNonUniformBallot", "spvGroupNonUniformArithmetic"};
        for (auto cap : capabilities)
        {
            slang::CompilerOptionEntry entry;
            entry.name = slang::CompilerOptionName::Capability;
            slang::CompilerOptionValue value;
            value.kind = slang::CompilerOptionValueKind::String;
            value.stringValue0 = cap;
            entry.value = value;
            compilerOptionsEntries.add(entry);
        }
        deviceDesc.slang.compilerOptionEntries = compilerOptionsEntries.getBuffer();
        deviceDesc.slang.compilerOptionEntryCount = (uint32_t)compilerOptionsEntries.getCount();
        deviceDesc.deviceType = rhi::DeviceType::Vulkan;
        // rhi::getRHI()->enableDebugLayers();
        gDevice = rhi::getRHI()->createDevice(deviceDesc);
        if (!gDevice)
            return SLANG_FAIL;

        gInferencingCtx = new InferencingContext(gDevice);

        SLANG_RETURN_ON_FAIL(testLinear(gInferencingCtx));
        SLANG_RETURN_ON_FAIL(testLinearPartitioned(gInferencingCtx));
        SLANG_RETURN_ON_FAIL(testTranspose(gInferencingCtx));
        SLANG_RETURN_ON_FAIL(testLeakyReluComposite(gInferencingCtx));
        SLANG_RETURN_ON_FAIL(testMaterialize(gInferencingCtx));
        SLANG_RETURN_ON_FAIL(testReluNegSin(gInferencingCtx));
        SLANG_RETURN_ON_FAIL(testMultiConcat(gInferencingCtx));
        SLANG_RETURN_ON_FAIL(testBroadcastAdd(gInferencingCtx));
        SLANG_RETURN_ON_FAIL(testIdentityPermute(gInferencingCtx));
        SLANG_RETURN_ON_FAIL(testNonTrivialPermute(gInferencingCtx));
        SLANG_RETURN_ON_FAIL(testClassifierFreeGuidance(gInferencingCtx));
        SLANG_RETURN_ON_FAIL(testConv2D(gInferencingCtx));
        SLANG_RETURN_ON_FAIL(testTransposedConv2D(gInferencingCtx));
        SLANG_RETURN_ON_FAIL(testBatchGemm(gInferencingCtx));
        SLANG_RETURN_ON_FAIL(testFusedBatchGemm(gInferencingCtx));
        SLANG_RETURN_ON_FAIL(testSoftmax(gInferencingCtx));

        // Half precision tests
        SLANG_RETURN_ON_FAIL(testLinearHalf(gInferencingCtx));
        SLANG_RETURN_ON_FAIL(testConv2DHalf(gInferencingCtx));
        SLANG_RETURN_ON_FAIL(testTransposedConv2DHalf(gInferencingCtx));
        SLANG_RETURN_ON_FAIL(testBatchGemmHalf(gInferencingCtx));
        SLANG_RETURN_ON_FAIL(testSoftmaxHalf(gInferencingCtx));
        SLANG_RETURN_ON_FAIL(testFlashAttentionHalf(gInferencingCtx));

        // Integer tests
        SLANG_RETURN_ON_FAIL(testLinearInt(gInferencingCtx));
        SLANG_RETURN_ON_FAIL(testConv2DInt(gInferencingCtx));
        SLANG_RETURN_ON_FAIL(testFlashAttention(gInferencingCtx));
        SLANG_RETURN_ON_FAIL(testFlashAttentionInputPermutationOnly(gInferencingCtx));
        SLANG_RETURN_ON_FAIL(testFlashAttentionFusedPermutation(gInferencingCtx));
        SLANG_RETURN_ON_FAIL(testCrossAttentionFull(gInferencingCtx));
        printf("all tests passed!\n");
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
