# CUDAECC

**CUDAECC** is a high-performance benchmarking suite that compares **Reed-Solomon (RS) error correction encoding and decoding** performance between **x86 CPUs utilizing GFNI** (Galois Field New Instructions) and **NVIDIA GPUs using CUDA**. 

This project demonstrates the raw throughput differences between hardware-accelerated CPU implementations and massively parallel GPU implementations, showcasing exactly how much faster CUDA executes Reed-Solomon calculations compared to modern x86 extensions.

## Core Features
* **GFNI SIMD Acceleration:** High-throughput x86 implementations utilizing AVX-512 / AVX2 with GFNI (`_mm512_gf2p8affine_epi64_epi8`).
* **Massively Parallel CUDA:** Fine-grained GPU optimization leveraging high-bandwidth memory.
* **Side-by-Side Validation:** End-to-end verification ensuring both implementations produce mathematically identical parity and reconstructed data.
* **Automated Benchmarking:** Built-in telemetry to measure throughput (GB/s).

## Data Structure
**CUDAECC** manages Reed Solomon encoding and decoding across a group of up to 255 buffers. This provides
error detection, correction and device fault tolerance at very high performance levels (>300GBytes/Second). **CUDAECC** contains ECC solutions in both CUDA and GFNI environments.

The buffers in this code are allocated and freed in kernel.cu. They are stored in pointer arrays.

**HOSTBUFS** is a pointer vector defined in kernel.cu, and holds the x86 addresses of buffers used for GFNI processing.

**GPUBUFS** is a pointer vector defined in kernel.cu, and holds the GPU addresses of buffers used for CUDA processing.

**CWP** is a pointer vector defined in DECODER_CUDA.cuh, is directly accessible by the GPU, and contains a copy of **GPUBUFS** 

## File Structure
* **kernel.cu** Kernel code that allocates buffers, calls encoder, injects errors, calls decoder, frees buffers and exits.
* **PLFSRSEQ_CUDA.cu** Accelerated encoder and decoder for GPU based on Parallel LFSR Sequencers
* **PLFSRSEQ_GFNI.c** Accelerated encoder and decoder for GFNI based on Parallel LFSR Sequencers
* **DECODER_CUDA.cu** Error decoder for CUDA
* **DECODER_GFNI.c** Error decoder for GFNI
* **CUDAECC.vcxproj** Project file for Visual Studio

## Performance Optimizations
All encoding and decoding starts with Parallel Linear Feedback Shift Registers. Instead of using an Encoding Matrix and a 
Decoding Matrix, GPUECC uses an LFSR structure to encode both original data (for encoding) and received data (for decoding). CUDAECC uses only Palindromic
Generator Polynomials, hence both the GFNI and CUDA versions cut the number of required multiplies, as well as the constant count, in half. 


**Parallel LFSR Sequencers** are provided in PLFSRSEQ_CUDA.cu and PLFSRSEQ_GFNI.c. For each of the 31 possible parity configurations (from 2-32 parity symbols), 
a unique function is produced for both the CUDA and GFNI encoders and decoders. That allows each configuration to allocate the
minimum amount of memory for the error correction polynomials Lambda (the Error Locator Polynomial), Omega (the Error Evaluator Polynomial), and
the other support polynomials. Each configuration from 2-32 parity symbols uses different constants and different sizes for the
error decoding polynomials, allowing small configurations to use a very small number of resources rather than being burdened by the worst case load.

For the GFNI implementation, single cycle GF multiplication is done with _mm512_gf2p8affine_epi64_epi8 and an affine table named PCAffTab shown in PLFSRSEQ_GFNI.h. 

A **GF Sparse Parallel Multiplier** is used for GF multiplication in CUDA encoding and decoding:

Here's how it works: Imagine you want to multiply a number 3 in GF space. The way you do it is first you add the number you want to multiply by 3 to the result, left shift the number by 1, and add it again to the result. Presto, you have multiplied by 3 if GF space. 3 * 3 = 3  * 2 + 3 = 6+3=110+011=101=5 (addition is xor in GF space). That's a Sparse multiply. To make it a Sparse Parallel Multiply, you do the same thing for 4 numbers in parallel. The key is that you know one of the multiply operands is fixed in advance, that lets you focus on just one operand instead of 2. To do 4 multiplies by the value 3 in parallel, load a 4-byte value into your GPU register. Add it to your 4-byte result. Left shift it by one, add that to the result, and Presto, 4 bytes multiplied by 3. In real life, you have to account for the potential of a bit shifting out, but that's pretty easy. Look in PLFSRSEQ_CUDA.CU for many examples.  Here's one as an example, this is a multiply by 0x49:

```
        // Process bit 0
        T0  = OV;                      // Account for bit 0 of constant 49
        MSK = OV & 0x80808080 ;        // Create a mask of the MSBs
        OF = (MSK >> 7) * 0x1d ;       // Account for Overflow
        OV <<= 1 ;                     // Multiply the original data by 2
        OV &= 0xfefefefe ;             // Clear the MSBs
        OV ^= OF ;                     // Add in the overflow mask

        // Process bit 1
        MSK = OV & 0x80808080 ;        // Create a mask of the MSBs
        OF = (MSK >> 7) * 0x1d ;       // Account for Overflow
        OV <<= 1 ;                     // Multiply the original data by 2
        OV &= 0xfefefefe ;             // Clear the MSBs
        OV ^= OF ;                     // Add in the overflow mask

        // Process bit 2
        MSK = OV & 0x80808080 ;        // Create a mask of the MSBs
        OF = (MSK >> 7) * 0x1d ;       // Account for Overflow
        OV <<= 1 ;                     // Multiply the original data by 2
        OV &= 0xfefefefe ;             // Clear the MSBs
        OV ^= OF ;                     // Add in the overflow mask

        // Process bit 3
        T0 ^= OV;                      // Account for bit 3 of constant 49
        MSK = OV & 0x80808080 ;        // Create a mask of the MSBs
        OF = (MSK >> 7) * 0x1d ;       // Account for Overflow
        OV <<= 1 ;                     // Multiply the original data by 2
        OV &= 0xfefefefe ;             // Clear the MSBs
        OV ^= OF ;                     // Add in the overflow mask

        // Process bit 4
        MSK = OV & 0x80808080 ;        // Create a mask of the MSBs
        OF = (MSK >> 7) * 0x1d ;       // Account for Overflow
        OV <<= 1 ;                     // Multiply the original data by 2
        OV &= 0xfefefefe ;             // Clear the MSBs
        OV ^= OF ;                     // Add in the overflow mask

        // Process bit 5
        MSK = OV & 0x80808080 ;        // Create a mask of the MSBs
        OF = (MSK >> 7) * 0x1d ;       // Account for Overflow
        OV <<= 1 ;                     // Multiply the original data by 2
        OV &= 0xfefefefe ;             // Clear the MSBs
        OV ^= OF ;                     // Add in the overflow mask

        // Process bit 6
        T0 ^= OV;                      // Account for bit 6 of constant 49
        MSK = OV & 0x80808080 ;        // Create a mask of the MSBs
        OF = (MSK >> 7) * 0x1d ;       // Account for Overflow
        OV <<= 1 ;                     // Multiply the original data by 2
        OV &= 0xfefefefe ;             // Clear the MSBs
        OV ^= OF ;                     // Add in the overflow mask

        // Process bit 7


```

For error decoding, there's no avoiding general purpose two argument multiplies. A power table and log table (PCPowTab and PCLogTab) for GF(2^8) PP 0x11d is expressed in DECODER_GFNI.h. These are copied to GPU 
memory by the kernel and held in gf_exp_CUDA and gf_log_CUDA, declared in DECODER_CUDA.cuh. Note that you don't want to use these tables for encoding or decoding because they tend to collide with each other in a GPU when shared by many threads, degrading the performance to a fraction of the original.

## Typical Benchmark Results
```text
Processor Brand: AMD Ryzen AI 9 HX 375 w/ Radeon 890M
Number of Logical Processors: 24
Processor Architecture: x64
Approximate Clock Speed: 1996 MHz
GPU Device Name: NVIDIA GeForce RTX 5060 Laptop GPU
K=247 p=8
Threads: 256 Blocks:16384
CUDA Encode Execution time: 12399200 ns
Bytes: 4080 MBytes
Rate: 345.038 GB/s
GFNI Encode Execution time: 498482200 ns
Bytes: 4080 MBytes
Rate: 8.58243 GB/s
Injected error 5a into parity buffer 0 offset 0
Injected error 5a into parity buffer 0 offset 1
GPU Correcting buffer 0 offset 16777214 with 5a
GPU Correcting buffer 0 offset 16777215 with 5a
CUDA Decode Execution time: 14086200 ns
Bytes: 4080 MBytes
Rate: 303.715 GB/s
CUDA Error count: 2
GFNI Correcting buffer 0 offset 16777214 with 5a
GFNI Correcting buffer 0 offset 16777215 with 5a
GFNIcode Execution time: 426948700 ns
Bytes: 4080 MBytes
Rate: 10.0204 GB/s
GFNI Error Count: 2
```
## Speed of Light Analysis by NVIDIA Nsight
<img width="1866" height="981" alt="NSIGHT" src="https://github.com/user-attachments/assets/5592aec7-f5e0-4294-bf10-62fd2dc2b60e" />

If you look near the top you will see GPU Speed of Light Throughput - which is a great summary line. This measurement of CUDAECC shows a compute load of 82.65% and a memory load of 89.27%, both very good. But what's really helpful is near the bottom under Memory Workload Analysis. The Memory Throughput [Gbytes/s] is 342.52. That corresponds exactly to my wall clock measurement time. There's nothing quite as good as 2 independent sources to validate performance claims. Other interesting things? At the top of the screen under "Attributes", you can see how many registers this solution used - 48. Too many registers and you won't fit nicely into the Streaming Multiprocessors, 48 seems fine. It also lists the # of SMs: 26, which is the spec but also nice to see. Finally, second section from the bottom under "Compute Workload Analysis" is "Executed Ipc Elapsed [inst.cycle], with a value of 2.54. I never thought about it before with respect to GPUs, but that's pretty good for a GPU with all those threads, a little better than 2.5 instructions per cycle on the average. 

## What is it that makes the CUDA solution so much faster?

I'd say it's one thing, and it's why x86/ARM will never catch up in their current form. That "one thing" is giving the outermost control loop to the GPU. As an example, take ECC encoding/decoding. Here's how it works: In x86/GFNI or ARM/NEON, you're going to create an outermost control loop, usually a "for" loop that starts at zero and increments until it hits the end of the buffer. In CUDA, you won't write this "for" loop at all. Instead, you describe the outermost control loop to the GPU, and he schedules this for you. There's a huge advantage in that. For GFNI, the Codec sweeps across the codeword 64 bytes at a time, and that's what the memory controller sees, (apparently) random 64 byte reads. In contrast, for CUDA, you describe "thread groups" and "blocks" at the highest level. For example, in my CUDA ECC codec, I group 256 threads together, each one handles generation of 4 bytes of ECC. 256 threads as a group means a 4 byte (32 bit) thread is now a 1024 byte group (4 * 256). Then, all 26 of the Streaming Multiprocessors in my RTX 5060 Mobile load up 256 threads each. That's (26*1k) = 26k that's all synchronized across 26 SMs each with 256 threads. I'm not sure if CUDA optimizes across "waves" (the 26 sets of 256 threads) or not, but let's say they don't. GFNI presents what looks like random 64 byte requests to the memory controller, while CUDA presents 26k requests, an advantage of over 400:1. So as a CUDA programmer, you need to think of yourself as small, and feed the GPU scheduler the parameters he needs to organize huge memory accesses. As an x86/ARM programmer, there is no way to achieve this level of organization at the memory controller. Tacking on hardware matrix multiplication (as some propose) won't change the outcome. GPUs will dominate the performance equation for AI, ECC, and most scientific calculations for the foreseeable future because of this outermost control loop and the effect it has on memory accesses.
