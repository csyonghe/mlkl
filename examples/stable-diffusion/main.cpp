// Stable Diffusion Example
// Runs VAE decoder, CLIP encoder, and UNet tests

#include "clip-encoder-test.h"
#include "core/slang-basic.h"
#include "example-base/example-base.h"
#include "inference-context.h"
#include "unet-test.h"
#include "vae-decoder-test.h"

#include <cstdio>

using namespace Slang;

static const ExampleResources resourceBase("stable-diffusion");

struct StableDiffusionProgram : public TestBase
{
    ComPtr<rhi::IDevice> device;
    RefPtr<InferencingContext> ctx;

    SlangResult execute(int argc, char* argv[])
    {
        parseOption(argc, argv);

        printf("=== Stable Diffusion Tests ===\n\n");

        // Initialize GPU device
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

        device = rhi::getRHI()->createDevice(deviceDesc);
        if (!device)
        {
            printf("Failed to create GPU device\n");
            return SLANG_FAIL;
        }

        ctx = new InferencingContext(device);

        // Run UNet tests
        printf("\n--- UNet Tests ---\n");
        SLANG_RETURN_ON_FAIL(testSDUNet(ctx));
        SLANG_RETURN_ON_FAIL(testSDResNetBlock(ctx));
        SLANG_RETURN_ON_FAIL(testSDSelfAttention(ctx));
        SLANG_RETURN_ON_FAIL(testSDCrossAttention(ctx));
        SLANG_RETURN_ON_FAIL(testSDSpatialTransformer(ctx));

        // Run CLIP encoder tests
        printf("\n--- CLIP Encoder Tests ---\n");
        SLANG_RETURN_ON_FAIL(testCLIPEncoderSD15(ctx));

        // Run VAE decoder tests
        printf("\n--- VAE Decoder Tests ---\n");
        SLANG_RETURN_ON_FAIL(testVAEDecoderSD15(ctx));
        SLANG_RETURN_ON_FAIL(testVAEResNetBlock(ctx));
        SLANG_RETURN_ON_FAIL(testVAEAttentionBlock(ctx));
        SLANG_RETURN_ON_FAIL(testVAEUpBlock(ctx));
        SLANG_RETURN_ON_FAIL(testVAEDecoderSmall(ctx));


        printf("\n=== All tests passed! ===\n");
        return SLANG_OK;
    }
};

int main(int argc, char** argv)
{
    StableDiffusionProgram app;
    if (SLANG_FAILED(app.execute(argc, argv)))
    {
        return -1;
    }
    return 0;
}
