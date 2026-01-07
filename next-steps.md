[ X ] Add `TensorView` C++ class that wraps both shape/stride and a BufferView.
[ X ] Refactor the library to create and accept TensorView everywhere for validation.
[ X ] Add `TensorView.slice` to allow writing to slices of output buffers.
[ ] Make `IExpr` work with `Coord` instead of linear index.
[ ] Make `IExpr` generic on element type, and provide `evalVec` for vectorized loads.
[ ] Test that batching works for the simple unet model.
[ ] Implement classifier free guidance in the conditioned model to improve its results.
