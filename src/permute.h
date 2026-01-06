#pragma once
#include "elementwise.h"
#include "kernel-base.h"

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