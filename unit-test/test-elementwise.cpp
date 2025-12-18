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
    auto bufIn = ctx->createPersistentBuffer(inputData, "TransposeIn");

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
    inputs.add(bufExpr, InputInfo(Shape{D0, D1, D2}, BufferView(bufIn)));

    // Allocate Output
    // kernel.allocResultBuffer resolves the shape automatically (should be [4, 3, 2])
    auto bufOut = kernel.allocResultBuffer(inputs);

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
    const int count = 16;
    float dataA[count];
    float dataB[count];
    for (int i = 0; i < count; i++)
    {
        dataA[i] = (float)i; // 0, 1, 2...
        dataB[i] = 100.0f;   // 100, 100...
    }

    auto bufA = ctx->createPersistentBuffer(dataA, count * sizeof(float));
    auto bufB = ctx->createPersistentBuffer(dataB, count * sizeof(float));

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

    Dictionary<Expr, InputInfo> inputs;
    inputs.add(a, {Shape(4, 4), bufA});
    inputs.add(b, {Shape(4, 4), bufB});

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
    auto outputBuffer = kernel->allocResultBuffer(inputs);
    kernel->queueExecute(task, outputBuffer, inputs);

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

    auto inputBuf = ctx->createPersistentBuffer(inputData, sizeof(inputData));

    // 1. Build Expression Tree
    // Expr = relu( -sin(x) )
    auto x = buffer();
    auto resultExpr = relu(-sin(x)); // Uses operator- overload for neg()

    // 2. Create Kernel
    ElementwiseKernel kernel(ctx, resultExpr);
    auto task = ctx->createTask();

    // 3. Bind Inputs & Execute
    Dictionary<Expr, InputInfo> inputs;
    inputs.add(x, InputInfo(shape, inputBuf));

    // Execute
    // Since it's a simple elementwise op, output shape matches input shape
    auto outputBuffer = kernel.allocResultBuffer(inputs);
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

    auto inputBuf = ctx->createPersistentBuffer(inputData, sizeof(inputData));

    auto x = buffer();
    // Use the C++ helper function
    auto resultExpr = leakyRelu(x, alpha);

    ElementwiseKernel kernel(ctx, resultExpr);
    auto task = ctx->createTask();

    Dictionary<Expr, InputInfo> inputs;
    inputs.add(x, InputInfo(shape, inputBuf));

    auto outputBuffer = kernel.allocResultBuffer(inputs);
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

    auto bufA = ctx->createPersistentBuffer(dataA, sizeof(dataA));
    auto bufB = ctx->createPersistentBuffer(dataB, sizeof(dataB));
    auto bufC = ctx->createPersistentBuffer(dataC, sizeof(dataC));

    BufferView inputs[] = {bufA, bufB, bufC};
    Shape shapes[] = {Shape({2}), Shape({3}), Shape({1})};

    ConcatKernel kernel(ctx, 3);
    auto task = ctx->createTask();
    auto outputBuffer = kernel.allocResultBuffer(makeArrayView(shapes), 0);

    kernel.queueExecute(task, outputBuffer, makeArrayView(inputs), makeArrayView(shapes), 0);

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