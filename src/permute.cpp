#include "permute.h"

PermuteKernel::PermuteKernel(InferencingContext* ctx, ArrayView<int> dims)
{
    // Construct the expression: Output = Permute(InputBuffer, dims)
    inputExpr = buffer();
    Expr root = permute(inputExpr, dims);

    // Compile the ElementwiseKernel
    kernel = new ElementwiseKernel(ctx, root);
}

PermuteKernel::PermuteKernel(InferencingContext* ctx, const std::initializer_list<int>& dims)
{
    // Construct the expression: Output = Permute(InputBuffer, dims)
    inputExpr = buffer();
    Expr root = permute(inputExpr, dims);
    // Compile the ElementwiseKernel
    kernel = new ElementwiseKernel(ctx, root);
}

void PermuteKernel::queueExecute(
    InferencingTask& task,
    BufferView output,
    BufferView input,
    const Shape& inputShape)
{
    Dictionary<Expr, InputInfo> inputs;
    inputs.add(inputExpr, InputInfo(inputShape, input));

    kernel->eval(task, output, inputs);
}