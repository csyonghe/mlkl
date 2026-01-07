#pragma once
#include "elementwise.h"
#include "kernel-base.h"

// Standalone permutation kernel.
//
// FUSION OPPORTUNITY: Consider fusing into adjacent kernels instead!
// - For BatchGemm: use transpose(buffer(), dim1, dim2) in the input expression
// - For Conv2D/Linear: use permute(buffer(), {dims}) in input expression
// - For elementwise ops: combine with other ops in single ElementwiseKernel
//
// Only use this standalone kernel when:
// - The permutation result is used by multiple downstream kernels
// - No adjacent kernel supports input expression fusion
class PermuteKernel : public RefObject
{
private:
    RefPtr<ElementwiseKernel> kernel;
    Expr inputExpr; // Handle to the input buffer node in the expression graph
    Index rank;

public:
    // Initializes the kernel for a fixed permutation (e.g., {0, 2, 1, 3})
    PermuteKernel(InferencingContext* ctx, ArrayView<int> dims);
    PermuteKernel(InferencingContext* ctx, const std::initializer_list<int>& dims);
    void queueExecute(InferencingTask& task, TensorView output, TensorView input);
};