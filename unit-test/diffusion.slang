interface ITile<T : __BuiltinArithmeticType>
{
    property int originX {get;}
    property int originY {get;}
    property int stride {get;}
    int getLinearIndex(int x, int y, int c);
    T read(int linearIndex);
    vector<T, N> readVector<int N>(int linearIndex);
    void write(int linearIndex, T value);
    void writeVector<int N>(int linearIndex, vector<T,N> value);
}

// Represent a 2d-slice of iamge in device or shared memory.
struct DeviceTile<T : __BuiltinArithmeticType, int channels> : ITile<T>
{
    // Origin in the larger image.
    int originX;
    int originY;

    // Stride (in elements) to go to next row.
    int stride;

    T* data;

    __init(int2 origin, int stride, T* data)
    {
        this.originX = origin.x;
        this.originY = origin.y;
        this.stride = stride;
        this.data = data;
    }

    int getLinearIndex(int x, int y, int c)
    {
        return (originY + y) * stride + (originX + x) * channels + c;
    }
    vector<T,N> readVector<int N>(int linearIndex)
    {
        return *(bit_cast<vector<T,N>*>(data + linearIndex));
    }
    void writeVector<int N>(int linearIndex, vector<T,N> value)
    {
        *(bit_cast<vector<T,N>*>(data + linearIndex)) = value;
    }
    
    T read(int linearIndex)
    {
        return data[linearIndex];
    }
    void write(int linearIndex, T value)
    {
        data[linearIndex] = value;
    }
}

static const int kSharedTensorSize = 8192;

struct SharedTile<T:__BuiltinArithmeticType, int channels> : ITile<T>
{
    // Origin in the larger image.
    int originX;
    int originY;

    // Stride (in elements) to go to next row.
    int stride;

    int dataIndex;
    static groupshared T sharedData[kSharedTensorSize];
    
    __init(int2 origin, int stride, int dataIndex)
    {
        this.originX = origin.x;
        this.originY = origin.y;
        this.stride = stride;
        this.dataIndex = dataIndex;
    }
    int getLinearIndex(int x, int y, int c)
    {
        return (originY + y) * stride + (originX + x) * channels + c;
    }
    vector<T,N> readVector<int N>(int linearIndex)
    {
        vector<T,N> v;
        [ForceUnroll]
        for (int i = 0; i < N; i++)
        {
            v[i] = sharedData[dataIndex + linearIndex + i];
        }
        return v;
    }
    void writeVector<int N>(int linearIndex, vector<T,N> value)
    {
        [ForceUnroll]
        for (int i = 0; i < N; i++)
        {
            sharedData[dataIndex + linearIndex + i] = value[i];
        }
    }
    T read(int linearIndex)
    {
        return sharedData[dataIndex + linearIndex];
    }
    void write(int linearIndex, T value)
    {
        sharedData[dataIndex + linearIndex] = value;
    }
}

void copyTileElement<int channels, T : __BuiltinArithmeticType, SrcTile:ITile<T>, DestTile:ITile<T>>(DestTile dest, int linearIndexInDest, SrcTile src, int linearIndexInSrc)
{
    if (channels % 4 == 0)
    {
        for (int c = 0; c < channels; c+=4)
        {
            // TODO: use loadAligned when it supports arbitrary address spaces.
            let vecValue =src.readVector<4>(linearIndexInSrc + c);
            dest.writeVector<4>(linearIndexInDest + c, vecValue);
        }
    }
    else
    {
        for (int c = 0; c < channels; c++)
        {
            dest.write(linearIndexInDest + c, src.read(linearIndexInSrc + c));
        }
    }
}

// A 2D convolution layer.
struct Conv2D<int kernelSize, int inChannels, int outChannels>
{
    float weights[kernelSize * kernelSize * inChannels * outChannels];
    float biases[outChannels];

    // Compute convolution of src, and store the result into dest.
    // Use inter as shared memory buffer to speed up memory access.
    // This function is supposed to be called from a compute shader with a 2D thread group that matches
    // the output tile size.
    void apply<
    SrcTile : ITile<float>,
    DestTile : ITile<float>,
    InterTile : ITile<float>>(
        SrcTile src,
        DestTile dest,
        InterTile inter,
        int outTileSize,
        int stride,
        int padding,
        int2 srcImageSize,
        int2 localThreadId // RENAMED: This is SV_GroupThreadID (local), not GroupIndex
    )
    {
        // --- 1. COLLABORATIVE LOADING ---
        // Note: This relies on 'src.originX/Y' being set to the 
        // top-left of the current TILE by the caller.
        
        int groupThreadIdx = outTileSize * localThreadId.y + localThreadId.x;
        int inputTileSize = outTileSize * stride; 
        
        // Calculate the full size of the required patch (Input Tile + Halo)
        int haloSize = kernelSize - 1;
        int totalPixelsToLoad = (inputTileSize + haloSize) * (inputTileSize + haloSize);

        for (int p = groupThreadIdx; p < totalPixelsToLoad; p += outTileSize * outTileSize)
        {
            int patchWidth = inputTileSize + haloSize;
            int loadY = p / patchWidth;
            int loadX = p % patchWidth;

            // Calculate global coordinates for bounds checking
            // Note: src.originX must be the top-left of the tile in Global space
            int absX = loadX + src.originX - padding;
            int absY = loadY + src.originY - padding;

            int interDestIndex = inter.getLinearIndex(loadX, loadY, 0);

            if (absX < 0 || absY < 0 || absX >= srcImageSize.x || absY >= srcImageSize.y)
            {
                for (int c = 0; c < inChannels; c++) 
                    inter.write(interDestIndex + c, 0.0);
            }
            else
            {
                // We use (loadX - padding) because src.getLinearIndex adds originX back in.
                // Effectively: GlobalX = src.originX + loadX - padding
                copyTileElement<inChannels>(inter, interDestIndex, src, src.getLinearIndex(loadX - padding, loadY - padding, 0));
            }
        }
        
        GroupMemoryBarrierWithGroupSync();

        // --- 2. CONVOLUTION ---

        // Pre-calculate the destination index for this thread
        let destLinearIndex = dest.getLinearIndex(localThreadId.x, localThreadId.y, 0);

        // Optimization: Loop over Output Channels FIRST.
        // This allows us to accumulate 'sum' in a register and write to VRAM once.
        for (int oc = 0; oc < outChannels; oc++)
        {
            // Initialize with Bias (Done once per pixel per channel)
            float sum = biases[oc];

            // Now loop over the Spatial Kernel
            for (int ky = 0; ky < kernelSize; ky++)
            {
                for (int kx = 0; kx < kernelSize; kx++)
                {
                    // Calculate index in Shared Memory (The cached tile)
                    // Note: threadId * stride gives us the input center
                    int smemX = kx + localThreadId.x * stride;
                    int smemY = ky + localThreadId.y * stride;
                    
                    int interLinearIndex = inter.getLinearIndex(smemX, smemY, 0);

                    // Accumulate Input Channels
                    for (int ic = 0; ic < inChannels; ic++)
                    {
                        // Compute flat weight index
                        // Layout: [OutCh][InCh][Ky][Kx]
                        int wIndex = oc * (inChannels * kernelSize * kernelSize) +
                                     ic * (kernelSize * kernelSize) +
                                     ky * kernelSize +
                                     kx;

                        float w = weights[wIndex];
                        float val = inter.read(interLinearIndex + ic);
                        
                        sum += w * val;
                    }
                }
            }

            // Write to global memory only ONCE per channel
            dest.write(destLinearIndex + oc, sum);
        }
    }
}

struct ConvTransposed2D<int kernelSize, int inChannels, int outChannels>
{
    float weights[kernelSize * kernelSize * inChannels * outChannels];
    float biases[outChannels];

    // Compute Transposed Convolution using Output-Centric approach.
    // 
    // groupThreadId: The SV_GroupThreadID (0..15, 0..15)
    // dest: Represents the slice for the *current tile* in global memory. 
    //       (dest.originX/Y must represent the Global Top-Left of this tile)
    void apply<
        SrcTile : ITile<float>,
        DestTile : ITile<float>,
        InterTile : ITile<float>>
    (
        SrcTile srcGlobal,
        DestTile dest,
        InterTile inter,
        int outTileSize,
        int stride,
        int padding,
        int2 srcImageSize,
        int2 groupThreadId) 
    {
        // ==========================================================
        // 1. CALCULATE INPUT BOUNDS
        // ==========================================================
        
        // We need to find the top-leftmost Input pixel that contributes 
        // to the top-leftmost Output pixel of this tile.
        
        // Inverse Formula: Input = floor((Output + Padding - KernelIndex) / Stride)
        // To find the lowest possible Input index, we maximize KernelIndex (kernelSize - 1).
        
        // Global Output Origin for this tile
        int outOriginX = dest.originX;
        int outOriginY = dest.originY;

        // Calculate Start Input X (Handle negative numerator correctly for floor division)
        // Numerator: outOriginX + padding - (kernelSize - 1)
        int numX = outOriginX + padding - (kernelSize - 1);
        int numY = outOriginY + padding - (kernelSize - 1);

        // Safe floor division for (num / stride)
        int startInputX = numX < 0 ? (numX - stride + 1) / stride : numX / stride;
        int startInputY = numY < 0 ? (numY - stride + 1) / stride : numY / stride;

        // Determine patch size required
        // A rough upper bound is (TileSize + KernelSize) / Stride + 2
        int inputPatchWidth = (outTileSize + kernelSize + stride - 1) / stride + 2;
        int totalPixelsToLoad = inputPatchWidth * inputPatchWidth;

        // Flatten thread ID for loading loop
        int tid = outTileSize * groupThreadId.y + groupThreadId.x;

        // ==========================================================
        // 2. COLLABORATIVE LOADING
        // ==========================================================
        
        for (int p = tid; p < totalPixelsToLoad; p += outTileSize * outTileSize)
        {
            int localLoadY = p / inputPatchWidth;
            int localLoadX = p % inputPatchWidth;
            
            // Global Input Coordinates to read from
            int globalInX = startInputX + localLoadX;
            int globalInY = startInputY + localLoadY;

            // Shared Memory Index (linear)
            int interIndex = inter.getLinearIndex(localLoadX, localLoadY, 0);

            // Bounds Check
            // Using a basic negative check for safety:
            if (globalInX < 0 || globalInY < 0 || globalInX >= srcImageSize.x || globalInY >= srcImageSize.y) 
            {
                for(int c = 0; c < inChannels; c++) inter.write(interIndex + c, 0.0);
            }
            else
            {
                // Read from Global Memory
                // Adjusting index relative to src.origin if src is not 0-based
                int srcLinear = srcGlobal.getLinearIndex(globalInX, globalInY, 0);
                copyTileElement<inChannels>(inter, interIndex, srcGlobal, srcLinear);
            }
        }

        GroupMemoryBarrierWithGroupSync();

        // ==========================================================
        // 3. COMPUTE (Accumulate Output)
        // ==========================================================
        
        // Output Global Coordinate
        int globalOutX = outOriginX + groupThreadId.x;
        int globalOutY = outOriginY + groupThreadId.y;
        
        int destLinearIndex = dest.getLinearIndex(groupThreadId.x, groupThreadId.y, 0);

        for (int oc = 0; oc < outChannels; oc++)
        {
            float sum = biases[oc];

            // Iterate over Kernel
            for (int ky = 0; ky < kernelSize; ky++)
            {
                for (int kx = 0; kx < kernelSize; kx++)
                {
                    // ---------------------------------------------------------
                    // DERIVATION:
                    // Forward Transposed Conv formula:
                    // OutputPos = (InputPos * Stride) + KernelPos - Padding
                    //
                    // Rearranging to solve for InputPos:
                    // InputPos * Stride = OutputPos - KernelPos + Padding
                    // ---------------------------------------------------------

                    // 1. Calculate the Right Hand Side (The "Strided Input Coordinate")
                    int stridedInX = globalOutX - kx + padding;
                    int stridedInY = globalOutY - ky + padding;

                    // 2. Stride Alignment Check (The "Integer Solution" Check)
                    // If the result isn't divisible by stride, then the InputPos 
                    // would be a fraction, which implies no input pixel aligns 
                    // with this output pixel via this specific kernel weight.
                    bool alignedX = (stridedInX % stride == 0);
                    bool alignedY = (stridedInY % stride == 0);

                    // CHECK 1: Stride Alignment
                    if (alignedX && alignedY)
                    {
                        // CHECK 2: Is the resulting Input Index within our loaded Shared Memory?
                        int globalInX = stridedInX / stride;
                        int globalInY = stridedInY / stride;
                        
                        int smemX = globalInX - startInputX;
                        int smemY = globalInY - startInputY;

                        if (smemX >= 0 && smemX < inputPatchWidth && 
                            smemY >= 0 && smemY < inputPatchWidth)
                        {
                            int interIdx = inter.getLinearIndex(smemX, smemY, 0);
                            
                            // Weight Index: [Out][In][Ky][Kx]
                            int wIndex = oc * (inChannels * kernelSize * kernelSize);

                            for (int ic = 0; ic < inChannels; ic++)
                            {
                                int currentWIndex = wIndex + 
                                                    ic * (kernelSize * kernelSize) + 
                                                    ky * kernelSize + kx;
                                
                                sum += weights[currentWIndex] * inter.read(interIdx + ic);
                            }
                        }
                    }
                }
            }
            dest.write(destLinearIndex + oc, sum);
        }
    }
};

struct SimpleConvolutionParams<int kernelSize, int inChannels, int outChannels>
{
    float* inputImage;
    float* outputImage;
    int inputImageWidth;
    int inputImageHeight;
    int outputImageWidth;
    int stride;
    int padding;
    Conv2D<kernelSize, inChannels, outChannels> convLayer;
}

[numthreads(tileSize,tileSize,1)]
void simpleConvolution<int tileSize, int kernelSize, int inChannels, int outChannels>(
    ConstantBuffer<SimpleConvolutionParams<kernelSize, inChannels, outChannels>, ScalarDataLayout> params,
    int3 groupThreadId : SV_GroupThreadID,
    int3 groupId : SV_GroupID)
{
    static groupshared float sharedMem[(tileSize + kernelSize - 1)*(tileSize + kernelSize - 1) * inChannels];
    int2 srcOrigin = groupId.xy * tileSize * params.stride;
    let srcTile = DeviceTile<float, inChannels>(srcOrigin, params.inputImageWidth*inChannels, params.inputImage);

    int2 destOrigin = groupId.xy * tileSize;
    int outputStride = params.outputImageWidth * outChannels;
    let destTile = DeviceTile<float, outChannels>(destOrigin, outputStride, params.outputImage);
    let interTile = SharedTile<float, inChannels>(
        0,
        (tileSize + kernelSize - 1) * inChannels,
        0);

    params.convLayer.apply(
        srcTile,
        destTile,
        interTile,
        tileSize,
        params.stride,
        params.padding,
        int2(params.inputImageWidth, params.inputImageHeight),
        groupThreadId.xy);
}

struct SimpleTransposedConvolutionParams<int kernelSize, int inChannels, int outChannels>
{
    float* inputImage;
    float* outputImage;
    int inputImageWidth;
    int inputImageHeight;
    int outputImageWidth;
    int stride;
    int padding;
    ConvTransposed2D<kernelSize, inChannels, outChannels> transposedConvLayer;
}

[numthreads(tileSize, tileSize, 1)]
void simpleTransposedConvolution<int tileSize, int kernelSize, int stride, int inChannels, int outChannels>(
    ConstantBuffer<SimpleTransposedConvolutionParams<kernelSize, inChannels, outChannels>, CDataLayout> params,
    int3 groupThreadId : SV_GroupThreadID,
    int3 groupId : SV_GroupID,
    int3 tid : SV_DispatchThreadID)
{
    // 1. Calculate Shared Memory Size (Compile-Time)
    // The input patch is roughly (OutputSize / Stride).
    // Formula matches 'inputPatchWidth' in the struct: (Tile + Kernel + Stride - 1) / Stride + 2
    static const int smemWidth = (tileSize + kernelSize + stride - 1) / stride + 2;
    static groupshared float sharedMem[smemWidth * smemWidth * inChannels];

    // 2. Source Tile (Input)
    // Strategy: Pass the GLOBAL view of the input image (Origin 0,0).
    // Why? The 'apply' function calculates precise input offsets using integer division 
    // that handles negative padding. Passing a local tile origin here is risky because 
    // the required input pixels might shift slightly "left" of a simple block boundary.
    let srcTile = DeviceTile<float, inChannels>(
        int2(0, 0), 
        params.inputImageWidth * inChannels, 
        params.inputImage
    );

    // 3. Dest Tile (Output)
    // Origin: The top-left of the OUTPUT tile for this group.
    // Note: We need to calculate the output stride properly.
    int outputStride = params.outputImageWidth * outChannels; 
    
    let destTile = DeviceTile<float, outChannels>(
        groupId.xy * tileSize, 
        outputStride, 
        params.outputImage
    );

    // 4. Shared Memory Tile (Intermediate)
    let interTile = SharedTile<float, inChannels>(
        int2(0, 0),
        smemWidth * inChannels,
        0
    );

    // 5. Run Logic
    params.transposedConvLayer.apply(
        srcTile,
        destTile,
        interTile,
        tileSize,
        stride,
        params.padding,
        int2(params.inputImageWidth, params.inputImageHeight),
        groupThreadId.xy
    );
}