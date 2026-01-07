#include "kernels.h"
#include "test-kernels.h"

// Local CPU Reference for 3D Transpose
static void cpuTranspose3D(
    const float* input,
    float* output,
    int d0,
    int d1,
    int d2,
    int swapA,
    int swapB)
{
    // Output dimensions
    int dims[3] = {d0, d1, d2};
    int outDims[3] = {dims[0], dims[1], dims[2]};
    std::swap(outDims[swapA], outDims[swapB]);

    // Input Strides
    int inStrides[3] = {d1 * d2, d2, 1};

    // Output Strides (for loop decomposition)
    int outStrides[3];
    outStrides[2] = 1;
    outStrides[1] = outDims[2];
    outStrides[0] = outDims[1] * outDims[2];

    int total = d0 * d1 * d2;
    for (int i = 0; i < total; i++)
    {
        // 1. Decompose linear index 'i' into Output Coordinates
        int temp = i;
        int c[3];
        c[0] = temp / outStrides[0];
        int rem = temp % outStrides[0];
        c[1] = rem / outStrides[1];
        c[2] = rem % outStrides[1];

        // 2. Map Output Coords -> Input Coords (Swap dimensions back)
        int inC[3] = {c[0], c[1], c[2]};
        std::swap(inC[swapA], inC[swapB]);

        // 3. Compute Input Linear Index
        int inIdx = inC[0] * inStrides[0] + inC[1] * inStrides[1] + inC[2] * inStrides[2];
        output[i] = input[inIdx];
    }
}

SlangResult testTranspose(InferencingContext* ctx)
{
    MLKL_TEST_BEGIN();

    // Case: Transpose [2, 3, 4] -> Swap(0, 2) -> [4, 3, 2]
    int D0 = 2, D1 = 3, D2 = 4;
    int total = D0 * D1 * D2;

    List<float> inputData;
    initRandom(inputData, total);

    // 1. CPU Reference
    List<float> expected;
    expected.setCount(total);
    cpuTranspose3D(inputData.getBuffer(), expected.getBuffer(), D0, D1, D2, 0, 2);

    // 2. GPU Setup
    auto bufIn =
        ctx->createTensor(ElementType::Float32, Shape(D0, D1, D2), inputData, "TransposeIn");

    // 3. Construct Expression: Transpose(Buffer, 0, 2)
    // Input Shape is implicitly provided via InputInfo later
    auto bufExpr = buffer();
    auto expr = transpose(bufExpr, 0, 2);

    // 4. Kernel
    ElementwiseKernel kernel(ctx, expr);
    auto task = ctx->createTask();

    // Bind Inputs
    Dictionary<Expr, InputInfo> inputs;
    // Note: expr->inner is the BufferNode created by buffer()
    // We bind the shape [2, 3, 4] to it.
    inputs.add(bufExpr, bufIn->getView());

    // Allocate Output
    // kernel.allocateResultBuffer resolves the shape automatically (should be [4, 3, 2])
    auto bufOut = kernel.allocateResultBuffer(ElementType::Float32, inputs);

    // 5. Execute
    kernel.queueExecute(task, bufOut, inputs);
    task.execute();

    // 6. Verify
    if (!checkOutput(ctx, bufOut, expected))
    {
        printf("testTranspose FAILED\n");
        return SLANG_FAIL;
    }

    MLKL_TEST_OK();
}

SlangResult testMaterialize(InferencingContext* ctx)
{
    MLKL_TEST_BEGIN();

    // 1. Setup Input Data
    // Create two 4x4 arrays
    const int width = 4;
    const int count = width * width;
    float dataA[count];
    float dataB[count];
    for (int i = 0; i < count; i++)
    {
        dataA[i] = (float)i; // 0, 1, 2...
        dataB[i] = 100.0f;   // 100, 100...
    }

    auto bufA =
        ctx->createTensor(ElementType::Float32, Shape(width, width), count * sizeof(float), dataA);
    auto bufB =
        ctx->createTensor(ElementType::Float32, Shape(width, width), count * sizeof(float), dataB);

    // 2. Build Expression Tree: (A + B) * 0.5
    auto a = buffer();
    auto b = buffer();
    auto p = a + b;
    // Expression: res = (a + b) * 0.5
    // Expected result[i] = (i + 100) * 0.5
    auto expr = (p * 0.3f + p * 0.7f) * 0.5f;

    // 3. Compile Pipeline
    RefPtr<ElementwiseKernel> kernel = new ElementwiseKernel(ctx, expr);

    // 4. Prepare Execution
    auto task = ctx->createTask();

    // 5. Eval
    // Effectively dispatches `materialize<Program<9,
    //  Eval<0, BufferView>,
    //  Eval<1, BufferView>,
    //  Eval<2, Add<Reg<0>,Reg<1>>>,
    //  Eval<3, ConstantView>,
    //  Eval<4, Mul<Reg<2>,Reg<3>>>,
    //  Eval<5, ConstantView>,
    //  Eval<6, Mul<Reg<2>,Reg<5>>>,
    //  Eval<7, Add<Reg<4>,Reg<6>>>,
    //  Eval<8, ConstantView>,
    //  Eval<9, Mul<Reg<7>,Reg<8>>>
    //  >>`.
    auto outputBuffer = ctx->allocScratchTensor(ElementType::Float32, Shape(width, width));
    kernel->queueExecute(task, outputBuffer, bufA->getView(), bufB->getView());

    // 6. Execute and Readback
    renderDocBeginFrame();
    task.execute();

    auto outputData = ctx->readBuffer<float>(outputBuffer);
    renderDocEndFrame();

    // 7. Verify Results
    for (int i = 0; i < count; i++)
    {
        float expected = (dataA[i] + dataB[i]) * 0.5f;
        TEST_CHECK("materialize", fabs(outputData[i] - expected) < 1e-3f);
    }

    MLKL_TEST_OK();
}

SlangResult testReluNegSin(InferencingContext* ctx)
{
    MLKL_TEST_BEGIN();

    // Scenario: Compute relu(-sin(x))
    // We pick values of x where sin(x) is positive, negative, and zero.

    // 0. Setup Data
    // x = 0        -> sin(0)=0          -> -0=0           -> relu(0) = 0
    // x = PI/2     -> sin(1.57)=1       -> -1             -> relu(-1) = 0
    // x = PI       -> sin(3.14)=~0      -> -0             -> relu(0) = 0
    // x = 3PI/2    -> sin(4.71)=-1      -> -(-1)=1        -> relu(1) = 1

    float PI = 3.14159265f;
    float inputData[] = {0.0f, PI / 2.0f, PI, 3.0f * PI / 2.0f};
    int count = 4;
    Shape shape = {count}; // [4]

    auto inputBuf = ctx->createTensor(ElementType::Float32, shape, sizeof(inputData), inputData);

    // 1. Build Expression Tree
    // Expr = relu( -sin(x) )
    auto x = buffer();
    auto resultExpr = relu(-sin(x)); // Uses operator- overload for neg()

    // 2. Create Kernel
    ElementwiseKernel kernel(ctx, resultExpr);
    auto task = ctx->createTask();

    // 3. Bind Inputs & Execute
    Dictionary<Expr, InputInfo> inputs;
    inputs.add(x, inputBuf->getView());

    // Execute
    // Since it's a simple elementwise op, output shape matches input shape
    auto outputBuffer = kernel.allocateResultBuffer(ElementType::Float32, inputs);
    kernel.queueExecute(task, outputBuffer, inputs);

    // 4. Readback
    renderDocBeginFrame();
    task.execute();

    auto output = ctx->readBuffer<float>(outputBuffer);
    renderDocEndFrame();

    // 5. Verify
    // Expected values calculation
    float expected[4];
    for (int i = 0; i < 4; i++)
    {
        float s = sin(inputData[i]);
        float n = -s;
        expected[i] = std::max(0.0f, n);
    }

    // Check (with tolerance for trig approximation)
    for (int i = 0; i < 4; i++)
    {
        if (fabs(output[i] - expected[i]) > 1e-3f)
        {
            printf(
                "ReluNegSin Mismatch at %d (Input %f): Got %f, Expected %f\n",
                i,
                inputData[i],
                output[i],
                expected[i]);
            return SLANG_FAIL;
        }
    }

    MLKL_TEST_OK();
}

SlangResult testLeakyReluComposite(InferencingContext* ctx)
{
    MLKL_TEST_BEGIN();

    // Test LeakyRelu(x, alpha=0.1)
    // x = 10.0  -> max(10, 1.0)  -> 10.0
    // x = -10.0 -> max(-10, -1.0) -> -1.0

    float inputData[] = {10.0f, -10.0f};
    float alpha = 0.1f;
    int count = 2;
    Shape shape = {count};

    auto inputBuf = ctx->createTensor(ElementType::Float32, shape, sizeof(inputData), inputData);

    auto x = buffer();
    // Use the C++ helper function
    auto resultExpr = leakyRelu(x, alpha);

    ElementwiseKernel kernel(ctx, resultExpr);
    auto task = ctx->createTask();

    Dictionary<Expr, InputInfo> inputs;
    inputs.add(x, inputBuf->getView());

    auto outputBuffer = kernel.allocateResultBuffer(ElementType::Float32, inputs);
    kernel.queueExecute(task, outputBuffer, inputs);

    // Readback
    renderDocBeginFrame();
    task.execute();

    auto output = ctx->readBuffer<float>(outputBuffer);
    renderDocEndFrame();

    // Verify
    float expected[] = {10.0f, -1.0f};
    for (int i = 0; i < 2; i++)
    {
        if (fabs(output[i] - expected[i]) > 1e-3f)
        {
            printf("LeakyRelu Mismatch at %d: Got %f, Expected %f\n", i, output[i], expected[i]);
            return SLANG_FAIL;
        }
    }
    MLKL_TEST_OK();
}

SlangResult testMultiConcat(InferencingContext* ctx)
{
    MLKL_TEST_BEGIN();

    // Scenario: Concat 3 vectors along axis 0
    // A: [1, 2]
    // B: [3, 4, 5]
    // C: [6]
    // Result: [1, 2, 3, 4, 5, 6] (Shape [6])

    float dataA[] = {1, 2};
    float dataB[] = {3, 4, 5};
    float dataC[] = {6};

    auto bufA = ctx->createTensor(ElementType::Float32, Shape(2), sizeof(dataA), dataA);
    auto bufB = ctx->createTensor(ElementType::Float32, Shape(3), sizeof(dataB), dataB);
    auto bufC = ctx->createTensor(ElementType::Float32, Shape(1), sizeof(dataC), dataC);

    TensorView inputs[] = {bufA->getView(), bufB->getView(), bufC->getView()};
    Shape shapes[] = {Shape({2}), Shape({3}), Shape({1})};

    ConcatKernel kernel(ctx, 3);
    auto task = ctx->createTask();
    auto outputBuffer = kernel.allocateResultBuffer(ElementType::Float32, makeArrayView(shapes), 0);

    kernel.queueExecute(task, outputBuffer, makeArrayView(inputs), 0);

    // Readback
    renderDocBeginFrame();
    task.execute();
    renderDocEndFrame();

    auto output = ctx->readBuffer<float>(outputBuffer);

    // Verify
    float expected[] = {1, 2, 3, 4, 5, 6};
    for (int i = 0; i < 6; i++)
    {
        if (output[i] != expected[i])
        {
            printf("Concat Mismatch at %d: %f != %f\n", i, output[i], expected[i]);
            return SLANG_FAIL;
        }
    }
    MLKL_TEST_OK();
}

SlangResult testIdentityPermute(InferencingContext* ctx)
{
    MLKL_TEST_BEGIN();

    // Create a 4D buffer [1, 2, 2, 2] = 8 elements
    Expr eBuf = buffer();
    Expr ePerm = permute(eBuf, {0, 1, 2, 3}); // Identity

    RefPtr<ElementwiseKernel> kernel = new ElementwiseKernel(ctx, ePerm);

    List<float> data = {0, 1, 2, 3, 4, 5, 6, 7};
    auto bufIn = ctx->createTensor(ElementType::Float32, Shape{1, 2, 2, 2}, data, "In");
    auto bufOut = ctx->allocScratchTensor(ElementType::Float32, Shape{1, 2, 2, 2}, "Out");

    Dictionary<Expr, InputInfo> inputs;
    // Note: The physical shape MUST be provided to the node
    inputs.add(eBuf, bufIn->getView());

    auto task = ctx->createTask();
    kernel->queueExecute(task, bufOut, inputs);
    task.execute();

    auto result = ctx->readBuffer<float>(bufOut);
    // If result is {0, 1, 2, 3, 4, 5, 6, 7}, indexing is correct.
    for (int i = 0; i < 8; i++)
    {
        if (result[i] != data[i])
        {
            printf("Identity Permute Mismatch at %d: %f != %f\n", i, result[i], data[i]);
            return SLANG_FAIL;
        }
    }
    MLKL_TEST_OK();
}

SlangResult testNonTrivialPermute(InferencingContext* ctx)
{
    MLKL_TEST_BEGIN();

    // Test 1: 2D Transpose [1, 0]
    // Input shape [2, 3] -> Output shape [3, 2]
    // Input:  [[0, 1, 2],      Output: [[0, 3],
    //          [3, 4, 5]]               [1, 4],
    //                                   [2, 5]]
    {
        Expr eBuf = buffer();
        Expr ePerm = permute(eBuf, {1, 0}); // Swap dimensions

        RefPtr<ElementwiseKernel> kernel = new ElementwiseKernel(ctx, ePerm);

        // Input data in row-major order: [2, 3]
        List<float> inputData = {0, 1, 2, 3, 4, 5};
        auto bufIn = ctx->createTensor(ElementType::Float32, Shape{2, 3}, inputData, "In2D");
        auto bufOut = ctx->allocScratchTensor(ElementType::Float32, Shape{3, 2}, "Out2D");

        Dictionary<Expr, InputInfo> inputs;
        inputs.add(eBuf, bufIn->getView());

        auto task = ctx->createTask();
        kernel->queueExecute(task, bufOut, inputs);
        task.execute();

        auto result = ctx->readBuffer<float>(bufOut);

        // Expected output in row-major [3, 2]: [[0, 3], [1, 4], [2, 5]]
        List<float> expected = {0, 3, 1, 4, 2, 5};
        for (int i = 0; i < 6; i++)
        {
            if (result[i] != expected[i])
            {
                return SLANG_FAIL;
            }
        }
    }

    // Test 2: 3D Permute [2, 0, 1]
    // Input shape [2, 3, 4] -> Output shape [4, 2, 3]
    // This is a cyclic permutation: dim0 -> dim1, dim1 -> dim2, dim2 -> dim0
    {
        Expr eBuf = buffer();
        Expr ePerm = permute(eBuf, {2, 0, 1}); // Cyclic permutation

        RefPtr<ElementwiseKernel> kernel = new ElementwiseKernel(ctx, ePerm);

        // Input data: shape [2, 3, 4] = 24 elements
        List<float> inputData;
        for (int i = 0; i < 24; i++)
            inputData.add((float)i);

        auto bufIn = ctx->createTensor(ElementType::Float32, Shape{2, 3, 4}, inputData, "In3D");
        auto bufOut = ctx->allocScratchTensor(ElementType::Float32, Shape{4, 2, 3}, "Out3D");

        Dictionary<Expr, InputInfo> inputs;
        inputs.add(eBuf, bufIn->getView());

        auto task = ctx->createTask();
        kernel->queueExecute(task, bufOut, inputs);
        task.execute();

        auto result = ctx->readBuffer<float>(bufOut);

        // Verify: output[d, b, c] = input[b, c, d]
        // where dims = [2, 0, 1] means:
        //   output dim 0 <- input dim 2
        //   output dim 1 <- input dim 0
        //   output dim 2 <- input dim 1
        // So at output coord (d, b, c), we read input at (b, c, d)
        bool allMatch = true;
        for (int d = 0; d < 4; d++)
        {
            for (int b = 0; b < 2; b++)
            {
                for (int c = 0; c < 3; c++)
                {
                    // Output index in row-major [4, 2, 3]
                    int outIdx = d * (2 * 3) + b * 3 + c;
                    // Input index in row-major [2, 3, 4]: input[b, c, d]
                    int inIdx = b * (3 * 4) + c * 4 + d;
                    float expected = inputData[inIdx];
                    if (result[outIdx] != expected)
                    {
                        allMatch = false;
                    }
                }
            }
        }
        if (!allMatch)
            return SLANG_FAIL;
    }

    // Test 3: 4D Permute [0, 2, 1, 3] (swap middle dimensions - like cross-attention)
    // Input shape [2, 3, 4, 5] -> Output shape [2, 4, 3, 5]
    {
        Expr eBuf = buffer();
        Expr ePerm = permute(eBuf, {0, 2, 1, 3}); // Swap dims 1 and 2

        RefPtr<ElementwiseKernel> kernel = new ElementwiseKernel(ctx, ePerm);

        // Input data: shape [2, 3, 4, 5] = 120 elements
        List<float> inputData;
        for (int i = 0; i < 120; i++)
            inputData.add((float)i);

        auto bufIn = ctx->createTensor(ElementType::Float32, Shape{2, 3, 4, 5}, inputData, "In4D");
        auto bufOut = ctx->allocScratchTensor(ElementType::Float32, Shape{2, 4, 3, 5}, "Out4D");

        Dictionary<Expr, InputInfo> inputs;
        inputs.add(eBuf, bufIn->getView());

        auto task = ctx->createTask();
        kernel->queueExecute(task, bufOut, inputs);
        task.execute();

        auto result = ctx->readBuffer<float>(bufOut);

        // Verify: output[a, c, b, d] = input[a, b, c, d]
        // dims = [0, 2, 1, 3] means:
        //   output dim 0 <- input dim 0
        //   output dim 1 <- input dim 2
        //   output dim 2 <- input dim 1
        //   output dim 3 <- input dim 3
        bool allMatch = true;
        int errorCount = 0;
        for (int a = 0; a < 2; a++)
        {
            for (int c = 0; c < 4; c++)
            { // output dim 1
                for (int b = 0; b < 3; b++)
                { // output dim 2
                    for (int d = 0; d < 5; d++)
                    {
                        // Output index in row-major [2, 4, 3, 5]
                        int outIdx = a * (4 * 3 * 5) + c * (3 * 5) + b * 5 + d;
                        // Input index in row-major [2, 3, 4, 5]: input[a, b, c, d]
                        int inIdx = a * (3 * 4 * 5) + b * (4 * 5) + c * 5 + d;
                        float expected = inputData[inIdx];
                        if (result[outIdx] != expected)
                        {
                            allMatch = false;
                            errorCount++;
                        }
                    }
                }
            }
        }
        if (!allMatch)
        {
            return SLANG_FAIL;
        }
    }

    MLKL_TEST_OK();
}