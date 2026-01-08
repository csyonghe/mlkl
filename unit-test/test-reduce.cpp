#include "reduce.h"
#include "test-kernels.h"

#include <cmath>

// =============================================================================
// Test: LastDimLayout - Reduce last dimension of 2D tensor
// This is useful for LayerNorm, RMSNorm where we reduce over the feature dim
// =============================================================================
SlangResult testReduceLastDim(InferencingContext* context)
{
    MLKL_TEST_BEGIN();

    // Parameters
    const int numRows = 8;
    const int numCols = 64;

    // Create the reduce kernel with LastDim layout
    Expr input = buffer();
    ReduceKernel reduceKernel(context, input, ReductionLayoutType::LastDim);

    // Prepare input data [numRows, numCols]
    List<float> hInput;
    hInput.setCount(numRows * numCols);
    for (Index i = 0; i < hInput.getCount(); ++i)
        hInput[i] = (float)(i % 10) * 0.1f + 0.1f;  // Values in [0.1, 1.0]

    auto inputBuffer = context->createTensor(
        ElementType::Float32,
        Shape(numRows, numCols),
        hInput);

    // Allocate stats output buffer
    int numGroups = numRows;
    auto statsBuffer = reduceKernel.allocateStatsBuffer(numGroups);

    // Execute
    auto task = context->createTask();
    LastDimLayoutParams layout{numRows, numCols};
    reduceKernel.queueExecute(task, statsBuffer, inputBuffer->getView(), layout);
    task.execute();

    // CPU Reference: compute (sum, sumSq) for each row
    List<float> expectedStats;
    expectedStats.setCount(numGroups * 2);
    for (int row = 0; row < numRows; ++row)
    {
        float sum = 0.0f;
        float sumSq = 0.0f;
        for (int col = 0; col < numCols; ++col)
        {
            float val = hInput[row * numCols + col];
            sum += val;
            sumSq += val * val;
        }
        expectedStats[row * 2] = sum;
        expectedStats[row * 2 + 1] = sumSq;
    }

    // Verify
    auto gpuResult = context->readBuffer<float>(statsBuffer);
    float maxDiff = 0.0f;
    for (Index i = 0; i < expectedStats.getCount(); ++i)
    {
        maxDiff = std::max(maxDiff, std::abs(gpuResult[i] - expectedStats[i]));
    }

    if (maxDiff > 1e-3f)
    {
        printf("[FAILED] %s: GPU and CPU results diverge.\n", __func__);
        printf("  - Max Difference: %e\n", maxDiff);
        for (int row = 0; row < std::min(4, numRows); ++row)
        {
            printf("  Row %d: GPU(sum=%f, sumSq=%f), CPU(sum=%f, sumSq=%f)\n",
                   row,
                   gpuResult[row * 2],
                   gpuResult[row * 2 + 1],
                   expectedStats[row * 2],
                   expectedStats[row * 2 + 1]);
        }
        return SLANG_FAIL;
    }

    MLKL_TEST_OK();
}

// =============================================================================
// Test: GroupNormLayout - Reduce (H, W, C/G) for NHWC GroupNorm
// =============================================================================
SlangResult testReduceGroupNorm(InferencingContext* context)
{
    MLKL_TEST_BEGIN();

    // Parameters (typical GroupNorm scenario)
    const int batchSize = 2;
    const int height = 8;
    const int width = 8;
    const int channels = 32;
    const int numGroups = 8;
    const int channelsPerGroup = channels / numGroups;

    // Create the reduce kernel with GroupNorm layout
    Expr input = buffer();
    ReduceKernel reduceKernel(context, input, ReductionLayoutType::GroupNorm);

    // Prepare input data in NHWC format
    List<float> hInput;
    hInput.setCount(batchSize * height * width * channels);
    for (Index i = 0; i < hInput.getCount(); ++i)
        hInput[i] = (float)((i * 7 + 3) % 20) * 0.05f;  // Values in [0, 0.95]

    auto inputBuffer = context->createTensor(
        ElementType::Float32,
        Shape(batchSize, height, width, channels),
        hInput);

    // Allocate stats output buffer
    int totalGroups = batchSize * numGroups;
    auto statsBuffer = reduceKernel.allocateStatsBuffer(totalGroups);

    // Execute
    auto task = context->createTask();
    GroupNormLayoutParams layout{batchSize, height, width, numGroups, channelsPerGroup};
    reduceKernel.queueExecute(task, statsBuffer, inputBuffer->getView(), layout);
    task.execute();

    // CPU Reference: compute (sum, sumSq) for each (batch, group) pair
    List<float> expectedStats;
    expectedStats.setCount(totalGroups * 2);

    for (int b = 0; b < batchSize; ++b)
    {
        for (int g = 0; g < numGroups; ++g)
        {
            float sum = 0.0f;
            float sumSq = 0.0f;

            int groupChannelStart = g * channelsPerGroup;
            for (int h = 0; h < height; ++h)
            {
                for (int w = 0; w < width; ++w)
                {
                    for (int lc = 0; lc < channelsPerGroup; ++lc)
                    {
                        int c = groupChannelStart + lc;
                        // NHWC indexing
                        int idx = ((b * height + h) * width + w) * channels + c;
                        float val = hInput[idx];
                        sum += val;
                        sumSq += val * val;
                    }
                }
            }

            int groupIdx = b * numGroups + g;
            expectedStats[groupIdx * 2] = sum;
            expectedStats[groupIdx * 2 + 1] = sumSq;
        }
    }

    // Verify
    auto gpuResult = context->readBuffer<float>(statsBuffer);
    float maxDiff = 0.0f;
    for (Index i = 0; i < expectedStats.getCount(); ++i)
    {
        maxDiff = std::max(maxDiff, std::abs(gpuResult[i] - expectedStats[i]));
    }

    if (maxDiff > 1e-2f)  // Larger tolerance for more elements
    {
        printf("[FAILED] %s: GPU and CPU results diverge.\n", __func__);
        printf("  - Max Difference: %e\n", maxDiff);
        for (int i = 0; i < std::min(4, totalGroups); ++i)
        {
            printf("  Group %d: GPU(sum=%f, sumSq=%f), CPU(sum=%f, sumSq=%f)\n",
                   i,
                   gpuResult[i * 2],
                   gpuResult[i * 2 + 1],
                   expectedStats[i * 2],
                   expectedStats[i * 2 + 1]);
        }
        return SLANG_FAIL;
    }

    MLKL_TEST_OK();
}

// =============================================================================
// Test: AxisLayout - Reduce over axis 1 of a 3D tensor
// =============================================================================
SlangResult testReduceAxis(InferencingContext* context)
{
    MLKL_TEST_BEGIN();

    // Parameters: reduce axis 1 of [4, 16, 8] tensor
    const int dim0 = 4;
    const int dim1 = 16;  // axis to reduce
    const int dim2 = 8;

    // Create the reduce kernel with Axis layout
    Expr input = buffer();
    ReduceKernel reduceKernel(context, input, ReductionLayoutType::Axis);

    // Prepare input data [dim0, dim1, dim2]
    List<float> hInput;
    hInput.setCount(dim0 * dim1 * dim2);
    for (Index i = 0; i < hInput.getCount(); ++i)
        hInput[i] = (float)((i * 13 + 5) % 17) * 0.1f;

    auto inputBuffer = context->createTensor(
        ElementType::Float32,
        Shape(dim0, dim1, dim2),
        hInput);

    // Allocate stats output buffer
    // numGroups = product of non-reduced dims = dim0 * dim2
    int numGroups = dim0 * dim2;
    auto statsBuffer = reduceKernel.allocateStatsBuffer(numGroups);

    // Execute
    auto task = context->createTask();
    AxisLayoutParams layout;
    layout.shape = Shape(dim0, dim1, dim2);
    layout.axis = 1;
    layout.elementsPerGroup = dim1;
    reduceKernel.queueExecute(task, statsBuffer, inputBuffer->getView(), layout);
    task.execute();

    // CPU Reference: reduce over axis 1
    // Output groups are indexed by (d0, d2) in row-major order
    List<float> expectedStats;
    expectedStats.setCount(numGroups * 2);

    for (int d0 = 0; d0 < dim0; ++d0)
    {
        for (int d2 = 0; d2 < dim2; ++d2)
        {
            float sum = 0.0f;
            float sumSq = 0.0f;

            for (int d1 = 0; d1 < dim1; ++d1)
            {
                int idx = (d0 * dim1 + d1) * dim2 + d2;
                float val = hInput[idx];
                sum += val;
                sumSq += val * val;
            }

            // Group index calculation matches AxisLayout::getCoord inverse
            // For axis=1, groupIdx encodes (d0, d2) in the order that
            // AxisLayout decodes them
            int groupIdx = d0 * dim2 + d2;
            expectedStats[groupIdx * 2] = sum;
            expectedStats[groupIdx * 2 + 1] = sumSq;
        }
    }

    // Verify
    auto gpuResult = context->readBuffer<float>(statsBuffer);
    float maxDiff = 0.0f;
    for (Index i = 0; i < expectedStats.getCount(); ++i)
    {
        maxDiff = std::max(maxDiff, std::abs(gpuResult[i] - expectedStats[i]));
    }

    if (maxDiff > 1e-3f)
    {
        printf("[FAILED] %s: GPU and CPU results diverge.\n", __func__);
        printf("  - Max Difference: %e\n", maxDiff);
        for (int i = 0; i < std::min(8, numGroups); ++i)
        {
            printf("  Group %d: GPU(sum=%f, sumSq=%f), CPU(sum=%f, sumSq=%f)\n",
                   i,
                   gpuResult[i * 2],
                   gpuResult[i * 2 + 1],
                   expectedStats[i * 2],
                   expectedStats[i * 2 + 1]);
        }
        return SLANG_FAIL;
    }

    MLKL_TEST_OK();
}

// =============================================================================
// Test: AxisLayout with higher-dimensional tensor (4D, reduce axis 2)
// =============================================================================
SlangResult testReduceAxis4D(InferencingContext* context)
{
    MLKL_TEST_BEGIN();

    // Parameters: reduce axis 2 of [2, 4, 8, 6] tensor
    const int dim0 = 2;
    const int dim1 = 4;
    const int dim2 = 8;  // axis to reduce
    const int dim3 = 6;

    // Create the reduce kernel with Axis layout
    Expr input = buffer();
    ReduceKernel reduceKernel(context, input, ReductionLayoutType::Axis);

    // Prepare input data
    List<float> hInput;
    hInput.setCount(dim0 * dim1 * dim2 * dim3);
    for (Index i = 0; i < hInput.getCount(); ++i)
        hInput[i] = (float)((i * 11 + 7) % 23) * 0.05f;

    auto inputBuffer = context->createTensor(
        ElementType::Float32,
        Shape(dim0, dim1, dim2, dim3),
        hInput);

    // numGroups = dim0 * dim1 * dim3 (all dims except axis 2)
    int numGroups = dim0 * dim1 * dim3;
    auto statsBuffer = reduceKernel.allocateStatsBuffer(numGroups);

    // Execute
    auto task = context->createTask();
    AxisLayoutParams layout;
    layout.shape = Shape(dim0, dim1, dim2, dim3);
    layout.axis = 2;
    layout.elementsPerGroup = dim2;
    reduceKernel.queueExecute(task, statsBuffer, inputBuffer->getView(), layout);
    task.execute();

    // CPU Reference: reduce over axis 2
    List<float> expectedStats;
    expectedStats.setCount(numGroups * 2);

    for (int d0 = 0; d0 < dim0; ++d0)
    {
        for (int d1 = 0; d1 < dim1; ++d1)
        {
            for (int d3 = 0; d3 < dim3; ++d3)
            {
                float sum = 0.0f;
                float sumSq = 0.0f;

                for (int d2 = 0; d2 < dim2; ++d2)
                {
                    int idx = ((d0 * dim1 + d1) * dim2 + d2) * dim3 + d3;
                    float val = hInput[idx];
                    sum += val;
                    sumSq += val * val;
                }

                // Group index: product of (d0, d1, d3) in row-major for non-reduced dims
                int groupIdx = (d0 * dim1 + d1) * dim3 + d3;
                expectedStats[groupIdx * 2] = sum;
                expectedStats[groupIdx * 2 + 1] = sumSq;
            }
        }
    }

    // Verify
    auto gpuResult = context->readBuffer<float>(statsBuffer);
    float maxDiff = 0.0f;
    for (Index i = 0; i < expectedStats.getCount(); ++i)
    {
        maxDiff = std::max(maxDiff, std::abs(gpuResult[i] - expectedStats[i]));
    }

    if (maxDiff > 1e-3f)
    {
        printf("[FAILED] %s: GPU and CPU results diverge.\n", __func__);
        printf("  - Max Difference: %e\n", maxDiff);
        return SLANG_FAIL;
    }

    MLKL_TEST_OK();
}

// =============================================================================
// Test: Large reduction to stress-test parallel reduction
// =============================================================================
SlangResult testReduceLarge(InferencingContext* context)
{
    MLKL_TEST_BEGIN();

    // Parameters: 4 rows, 4096 columns (larger than thread group size)
    const int numRows = 4;
    const int numCols = 4096;

    // Create the reduce kernel with LastDim layout
    Expr input = buffer();
    ReduceKernel reduceKernel(context, input, ReductionLayoutType::LastDim);

    // Prepare input data with varied values
    List<float> hInput;
    hInput.setCount(numRows * numCols);
    for (Index i = 0; i < hInput.getCount(); ++i)
    {
        // Use sine to get varied values in [-1, 1]
        hInput[i] = std::sin((float)i * 0.01f);
    }

    auto inputBuffer = context->createTensor(
        ElementType::Float32,
        Shape(numRows, numCols),
        hInput);

    int numGroups = numRows;
    auto statsBuffer = reduceKernel.allocateStatsBuffer(numGroups);

    // Execute
    auto task = context->createTask();
    LastDimLayoutParams layout{numRows, numCols};
    reduceKernel.queueExecute(task, statsBuffer, inputBuffer->getView(), layout);
    task.execute();

    // CPU Reference
    List<float> expectedStats;
    expectedStats.setCount(numGroups * 2);
    for (int row = 0; row < numRows; ++row)
    {
        double sum = 0.0;  // Use double for more accurate reference
        double sumSq = 0.0;
        for (int col = 0; col < numCols; ++col)
        {
            double val = hInput[row * numCols + col];
            sum += val;
            sumSq += val * val;
        }
        expectedStats[row * 2] = (float)sum;
        expectedStats[row * 2 + 1] = (float)sumSq;
    }

    // Verify with larger tolerance due to accumulation errors
    auto gpuResult = context->readBuffer<float>(statsBuffer);
    float maxDiff = 0.0f;
    for (Index i = 0; i < expectedStats.getCount(); ++i)
    {
        maxDiff = std::max(maxDiff, std::abs(gpuResult[i] - expectedStats[i]));
    }

    // Allow 0.1% relative error for large reductions
    float tolerance = std::max(1e-2f, std::abs(expectedStats[0]) * 0.001f);
    if (maxDiff > tolerance)
    {
        printf("[FAILED] %s: GPU and CPU results diverge.\n", __func__);
        printf("  - Max Difference: %e, Tolerance: %e\n", maxDiff, tolerance);
        for (int row = 0; row < numRows; ++row)
        {
            printf("  Row %d: GPU(sum=%f, sumSq=%f), CPU(sum=%f, sumSq=%f)\n",
                   row,
                   gpuResult[row * 2],
                   gpuResult[row * 2 + 1],
                   expectedStats[row * 2],
                   expectedStats[row * 2 + 1]);
        }
        return SLANG_FAIL;
    }

    MLKL_TEST_OK();
}

// =============================================================================
// Test: Half precision reduction
// =============================================================================
SlangResult testReduceHalf(InferencingContext* context)
{
    MLKL_TEST_BEGIN();

    // Parameters
    const int numRows = 4;
    const int numCols = 64;

    // Create the reduce kernel with half precision
    Expr input = buffer();
    ReduceKernel reduceKernel(context, ElementType::Float16, input, ReductionLayoutType::LastDim);

    // Prepare input data in float, then convert to half
    List<float> hInput;
    hInput.setCount(numRows * numCols);
    for (Index i = 0; i < hInput.getCount(); ++i)
        hInput[i] = (float)(i % 8) * 0.1f + 0.1f;

    List<uint16_t> hInputHalf;
    floatToHalf(hInput, hInputHalf);

    auto inputBuffer = context->createTensor(
        ElementType::Float16,
        Shape(numRows, numCols),
        hInputHalf.getCount() * sizeof(uint16_t),
        hInputHalf.getBuffer());

    int numGroups = numRows;
    auto statsBuffer = reduceKernel.allocateStatsBuffer(numGroups);

    // Execute
    auto task = context->createTask();
    LastDimLayoutParams layout{numRows, numCols};
    reduceKernel.queueExecute(task, statsBuffer, inputBuffer->getView(), layout);
    task.execute();

    // CPU Reference using float precision
    List<float> expectedStats;
    expectedStats.setCount(numGroups * 2);
    for (int row = 0; row < numRows; ++row)
    {
        float sum = 0.0f;
        float sumSq = 0.0f;
        for (int col = 0; col < numCols; ++col)
        {
            float val = hInput[row * numCols + col];
            sum += val;
            sumSq += val * val;
        }
        expectedStats[row * 2] = sum;
        expectedStats[row * 2 + 1] = sumSq;
    }

    // Verify with larger tolerance for half precision
    auto gpuResultHalf = context->readBuffer<uint16_t>(statsBuffer);
    List<float> gpuResult;
    halfToFloat(gpuResultHalf, gpuResult);

    float maxDiff = 0.0f;
    for (Index i = 0; i < expectedStats.getCount(); ++i)
    {
        maxDiff = std::max(maxDiff, std::abs(gpuResult[i] - expectedStats[i]));
    }

    // Half precision tolerance
    float tolerance = std::max(0.05f, std::abs(expectedStats[0]) * 0.01f);
    if (maxDiff > tolerance)
    {
        printf("[FAILED] %s: GPU and CPU results diverge.\n", __func__);
        printf("  - Max Difference: %e, Tolerance: %e\n", maxDiff, tolerance);
        return SLANG_FAIL;
    }

    MLKL_TEST_OK();
}


