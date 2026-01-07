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

        // SafeTensors tests (no GPU needed)
        printf("=== SafeTensors Reader Tests ===\n");
        SLANG_RETURN_ON_FAIL(testSafeTensorsLoad());
        SLANG_RETURN_ON_FAIL(testSafeTensorsLoadMissing());
        SLANG_RETURN_ON_FAIL(testSafeTensorsTensorInfo());
        SLANG_RETURN_ON_FAIL(testSafeTensorsReadBasic());
        SLANG_RETURN_ON_FAIL(testSafeTensorsTypeConversion());
        SLANG_RETURN_ON_FAIL(testSafeTensorsLinear());
        SLANG_RETURN_ON_FAIL(testSafeTensorsConv2DPermutation());
        SLANG_RETURN_ON_FAIL(testSafeTensorsTransposedConv2DPermutation());
        SLANG_RETURN_ON_FAIL(testSafeTensorsPermutation());
        SLANG_RETURN_ON_FAIL(testSafeTensorsMixedPrecision());
        SLANG_RETURN_ON_FAIL(testSafeTensorsEmbedding());
        SLANG_RETURN_ON_FAIL(testSafeTensorsNorm());
        printf("=== GPU Kernel Tests ===\n");

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

        // Reduction kernel tests
        SLANG_RETURN_ON_FAIL(testReduceLastDim(gInferencingCtx));
        SLANG_RETURN_ON_FAIL(testReduceGroupNorm(gInferencingCtx));
        SLANG_RETURN_ON_FAIL(testReduceAxis(gInferencingCtx));
        SLANG_RETURN_ON_FAIL(testReduceAxis4D(gInferencingCtx));
        SLANG_RETURN_ON_FAIL(testReduceLarge(gInferencingCtx));
        SLANG_RETURN_ON_FAIL(testReduceHalf(gInferencingCtx));

        // GroupNorm kernel tests
        SLANG_RETURN_ON_FAIL(testGroupNorm(gInferencingCtx));
        SLANG_RETURN_ON_FAIL(testGroupNormSingleGroup(gInferencingCtx));
        SLANG_RETURN_ON_FAIL(testGroupNormPerChannel(gInferencingCtx));
        SLANG_RETURN_ON_FAIL(testGroupNormLarge(gInferencingCtx));
        SLANG_RETURN_ON_FAIL(testGroupNormHalf(gInferencingCtx));
        SLANG_RETURN_ON_FAIL(testGroupNormStats(gInferencingCtx));

        // LayerNorm kernel tests
        SLANG_RETURN_ON_FAIL(testLayerNorm(gInferencingCtx));
        SLANG_RETURN_ON_FAIL(testLayerNormStats(gInferencingCtx));
        SLANG_RETURN_ON_FAIL(testLayerNormLarge(gInferencingCtx));
        SLANG_RETURN_ON_FAIL(testLayerNormHalf(gInferencingCtx));

        // RMSNorm kernel tests
        SLANG_RETURN_ON_FAIL(testRMSNorm(gInferencingCtx));
        SLANG_RETURN_ON_FAIL(testRMSNormIdentity(gInferencingCtx));
        SLANG_RETURN_ON_FAIL(testRMSNormLarge(gInferencingCtx));
        SLANG_RETURN_ON_FAIL(testRMSNormHalf(gInferencingCtx));

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
