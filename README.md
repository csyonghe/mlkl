# (Unofficial) Slang Machine Learning Kernel Library

This repository hosts a collection of machine learning kernels implemented in Slang,
a shading language designed for high-performance computing on GPUs.
These kernels can be used for various machine learning inferencing tasks.

**Disclaimer**: This work is exploratory, and is not a part of the Slang project. Use at your own risk.

## File Structure

- `/src/mlkl.slang` is the main Slang module file containing the implementations of various machine learning kernels.
- `/src/kernels.h` is a C++ header file that provides declarations for the Slang kernels, allowing them to be called from C++ code. The C++ host logic
   for launching and managing these kernels is built on top of `slang-rhi`, a graphics hardware abstraction layer that tightly integrates the Slang
   shading language for GPU programming.
- Each kernel implementation is separated into three files, `<kernel-name>.slang`, `<kernel-name>.h` and `<kernel-name>.cpp`, all located in the `/src/`
  directory. The `.slang` file contains the Slang code for the kernel, the `.h` file contains the C++ declarations, and the `.cpp` file contains the C++ host logic for launching the kernel.

## Building the Library

To build the Slang Machine Learning Kernel Library, follow these steps:
1. Clone the repository.
1. Run `setup.bat` or `setup.sh` to initiate the environment setup. This only needs to be done once.
1. Run `build.bat` or `build.sh` to compile the library.

## Testing Your Build

After building the library, you can run the provided test cases to verify that everything is working correctly. Use the command:

```
<build_diretory>/unit-test.exe
```

You should see `All tests passed!` if everything is functioning as expected.
