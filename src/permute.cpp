#include "permute.h"

PermuteKernel::PermuteKernel(InferencingContext* ctx, ArrayView<int> dims)
{
    // Construct the expression: Output = Permute(InputBuffer, dims)
    inputExpr = buffer();
    Expr root = permute(inputExpr, dims);

    // Compile the ElementwiseKernel
    kernel = new ElementwiseKernel(ctx, root);

    rank = dims.getCount();
}

PermuteKernel::PermuteKernel(InferencingContext* ctx, const std::initializer_list<int>& dims)
{
    // Construct the expression: Output = Permute(InputBuffer, dims)
    inputExpr = buffer();
    Expr root = permute(inputExpr, dims);
    // Compile the ElementwiseKernel
    kernel = new ElementwiseKernel(ctx, root);
    rank = (Index)dims.size();
}

void PermuteKernel::queueExecute(InferencingTask& task, TensorView output, TensorView input)
{
    // Validate input/output shapes.
    if (output.shape.getRank() != rank)
    {
        throw std::runtime_error("PermuteKernel::queueExecute: Output rank mismatch");
    }
    if (input.shape.getRank() != rank)
    {
        throw std::runtime_error("PermuteKernel::queueExecute: Input rank mismatch");
    }
    if (input.shape.getElementCount() != output.shape.getElementCount())
    {
        throw std::runtime_error(
            "PermuteKernel::queueExecute: Input/Output element count mismatch");
    }
    Dictionary<Expr, InputInfo> inputs;
    inputs.add(inputExpr, input);
    kernel->queueExecute(task, output, inputs);
}