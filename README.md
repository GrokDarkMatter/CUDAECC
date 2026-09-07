# CUDAECC

**CUDAECC** is a high-performance benchmarking suite that compares **Reed-Solomon (RS) error correction encoding and decoding** performance between **x86 CPUs utilizing GFNI** (Galois Field New Instructions) and **NVIDIA GPUs using CUDA**. 

This project demonstrates the raw throughput differences between hardware-accelerated CPU implementations and massively parallel GPU implementations, showcasing exactly how much faster CUDA executes Reed-Solomon calculations compared to modern x86 extensions.

## Core Features
* **GFNI SIMD Acceleration:** High-throughput x86 implementations utilizing AVX-512 / AVX2 with GFNI (`_mm512_vgf2p8affineinv_epi8`).
* **Massively Parallel CUDA:** Fine-grained GPU optimization leveraging high-bandwidth memory.
* **Side-by-Side Validation:** End-to-end verification ensuring both implementations produce mathematically identical parity and reconstructed data.
* **Automated Benchmarking:** Built-in telemetry to measure throughput (GB/s).

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
GPU Correcting buffer 0 offset 16777214 with 5a
GPU Correcting buffer 0 offset 16777215 with 5a
Injected error 5a into parity buffer 0 offset 0
Injected error 5a into parity buffer 0 offset 1
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
<img width="1866" height="981" alt="NSIGHT" src="https://github.com/user-attachments/assets/5592aec7-f5e0-4294-bf10-62fd2dc2b60e" />
