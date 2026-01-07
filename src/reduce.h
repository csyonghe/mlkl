#pragma once
#include "elementwise.h"
#include "kernel-base.h"

// Layout types for reduction kernels
enum class ReductionLayoutType
{
    LastDim,    // Reduce last dimension of 2D tensor (LayerNorm, RMSNorm)
    GroupNorm,  // Reduce (H, W, C/G) for NHWC GroupNorm
    Axis,       // General single-axis reduction
};

// Parameters for LastDimLayout
struct LastDimLayoutParams
{
    int numRows;
    int numCols;
};

// Parameters for GroupNormLayout
struct GroupNormLayoutParams
{
    int batchSize;
    int height;
    int width;
    int numGroups;
    int channelsPerGroup;
};

// Parameters for AxisLayout
struct AxisLayoutParams
{
    Shape shape;  // Up to 8 dimensions
    int axis;
    int elementsPerGroup;
};

// Generic Reduction Kernel
// Computes (sum, sumSq) for each reduction group
class ReduceKernel : public RefObject
{
private:
    ComPtr<rhi::IComputePipeline> pipeline;
    InferencingContext* context;
    ElementType elementType;
    ReductionLayoutType layoutType;

    ProgramNode inputProgram;

public:
    ReduceKernel(
        InferencingContext* ctx,
        ElementType elementType,
        Expr inputExpr,
        ReductionLayoutType layoutType);

    // Convenience constructor defaulting to Float32
    ReduceKernel(
        InferencingContext* ctx,
        Expr inputExpr,
        ReductionLayoutType layoutType)
        : ReduceKernel(ctx, ElementType::Float32, inputExpr, layoutType)
    {
    }

    ElementType getElementType() const { return elementType; }

    // Allocate stats buffer: [numGroups * 2] elements for (sum, sumSq) pairs
    BufferView allocateStatsBuffer(int numGroups);

    // Execute reduction with LastDimLayout
    void queueExecute(
        InferencingTask& task,
        BufferView statsOutput,
        TensorView input,
        const LastDimLayoutParams& layout);

    // Execute reduction with GroupNormLayout
    void queueExecute(
        InferencingTask& task,
        BufferView statsOutput,
        TensorView input,
        const GroupNormLayoutParams& layout);

    // Execute reduction with AxisLayout
    void queueExecute(
        InferencingTask& task,
        BufferView statsOutput,
        TensorView input,
        const AxisLayoutParams& layout);
};

