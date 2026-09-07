//***************************************************************************************
// Copyright 2026 by StreamScale Inc. All rights reserved.
// This software is free to use for non-commercial or evaluation purposes, but may not 
// be redistributed or sold for any commercial purpose without the express written
// permission of StreamScale Inc.
// 
// In other words, this code is provided solely for the purposes of
// evaluation and is not licensed or intended to be licensed or used as part of
// or in connection with any commercial or non - commercial use other than evaluation
// of the potential for a license from StreamScale Inc. Neither StreamScale Inc. 
// nor any affiliated person grants any express or implied rights under any patents,
// copyrights, trademarks, or trade secret information. 
// 
// This software includes contributions protected by 
// U.S. Patents 11,848,686 and 12,341,532
//***************************************************************************************

#include <stdio.h>
#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include "kernel.cuh"

/* ------------------------------------------------------------------------------------ */
/*                               Note CUDA code is inlined                              */
/* ------------------------------------------------------------------------------------ */
#include "PLFSRSEQ_CUDA.cu"

/* ------------------------------------------------------------------------------------ */
/*                                  GFNI Section                                        */
/* ------------------------------------------------------------------------------------ */
// Host side Codeword buffer pointers
unsigned int* GPUBUFS[255];                // Host side copy of GPU buffers
unsigned char* HOSTBUFS[255];              // Host side buffers for encoding/decoding
int herror_count;                          // Host side error count for GPU decoder errors

/* ------------------------------------------------------------------------------------ */
/*         Allocate host and device buffers for testing, copy exp and log tables        */
/* ------------------------------------------------------------------------------------ */
int PCECCMalloc(int k, int p, int size)
{
    cudaError_t cudaStatus;
	int totBuf = k + p;                     // Total buffers for original data and check

    // Create and assign each buffer of the codeword
    for (int i = 0; i < totBuf; i++)
    {
		// Allocate 1 buffer on the GPU and 1 buffer on the host
        cudaStatus = cudaMalloc (&GPUBUFS[ i ], size);
        if (cudaStatus != cudaSuccess)
        {
            fprintf (stderr, "cudaMalloc GPUBUFSfailed!");
            return 1;
        }

        cudaStatus = cudaMallocHost (&HOSTBUFS[ i ], size);
        if (cudaStatus != cudaSuccess)
        {
            fprintf (stderr, "cudaMallocHost HOSTBUFS failed!");
            return 1;
        }

		// Initialize the host buffer with some data
        cudaStatus = cudaMemset(HOSTBUFS[i], 0x5a, BUFFER_SIZE);
        if (cudaStatus != cudaSuccess)
        {
            fprintf(stderr, "cudaMemset HOSTBUFS failed!");
            return 1;
        }
    
        cudaStatus = cudaMemcpy (GPUBUFS[ i ], HOSTBUFS[ i ], BUFFER_SIZE, cudaMemcpyHostToDevice);
        if (cudaStatus != cudaSuccess)
        {
            fprintf (stderr, "cudaMemcpy GPUBUFS failed!");
            return 1;
        }
    }

    // Copy the pointer array directly to the constant memory symbol
    cudaStatus = cudaMemcpyToSymbol(CWP, GPUBUFS, 255 * sizeof(float*));
    if (cudaStatus != cudaSuccess)
    {
        fprintf(stderr, "cudaMemcpyToSymbol CWP failed!");
        return 1;
    }

    // Copy the 0 error count to the gpu
    herror_count = 0;
    cudaStatus = cudaMemcpyToSymbol(error_count, &herror_count, sizeof (int));
    if (cudaStatus != cudaSuccess)
    {
        fprintf(stderr, "cudaMemcpyToSymbol error_count failed!");
        return 1;
    }

    // Copy the power table to the gpu
    cudaStatus = cudaMemcpyToSymbol(gf_exp_CUDA, &PCPowTab, sizeof(PCPowTab));
    if (cudaStatus != cudaSuccess)
    {
        fprintf(stderr, "cudaMemcpyToSymbol gf_exp_CUDA failed!");
        return 1;
    }

    // Copy the log table to the gpu
    cudaStatus = cudaMemcpyToSymbol(gf_log_CUDA, &PCLogTab, sizeof(PCLogTab));
    if (cudaStatus != cudaSuccess)
    {
        fprintf(stderr, "cudaMemcpyToSymbol gf_log_CUDA failed!");
        return 1;
    }

    return 0;
}

/* ------------------------------------------------------------------------------------ */
/*         Free buffers post testing                                                    */
/* ------------------------------------------------------------------------------------ */
void PCECCFree (int k, int p)
{
    for (int i = 0; i < (k + p + p); ++i)
    {
        cudaFree (&GPUBUFS[ i ]);
        cudaFreeHost (&HOSTBUFS[ i ]);
    }
}

/* ------------------------------------------------------------------------------------ */
/*   Inject errors for testing                                                          */
/* ------------------------------------------------------------------------------------ */
void InjectErrors()
{
	// Inject errors into the codeword buffers
	for (int i = 0; i < 2; i++)
	{
		// Inject an error into the first byte of each parity buffer
		HOSTBUFS[0][i] ^= 0x5a;
		printf("Injected error %x into parity buffer %d\n", 0xff, i);
	}
	// Copy the modified parity buffers back to the GPU
	for (int i = 0; i < 1; ++i)
	{
		cudaMemcpy(GPUBUFS[i], HOSTBUFS[i], BUFFER_SIZE, cudaMemcpyHostToDevice);
	}
}

/* ------------------------------------------------------------------------------------ */
/*   Print performance stats                                                            */
/* ------------------------------------------------------------------------------------ */
void ReportRate(float bytes, const char * str, std::chrono::steady_clock::time_point start, 
    std::chrono::steady_clock::time_point end)
{
	// Cast the duration straight to nanoseconds
	auto elapsed_ns = std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count();
	std::cout << str << "Execution time: " << elapsed_ns << " ns" << std::endl;
	std::cout << "Bytes: " << bytes / (1024 * 1024) << " MBytes" << std::endl;
	float ens = (float)elapsed_ns;
	float rate = bytes / ens;
	std::cout << "Rate: " << rate << " GB/s" << std::endl;
}

/* ------------------------------------------------------------------------------------ */
/* Check arguments, allocate buffers, encode and decode with CUDA and GFNI, free, exit  */
/* ------------------------------------------------------------------------------------ */
int main(int argc, char** argv)
{
    cudaError_t cudaStatus;
    int p = 8;
    int k = 247;

    /* Parse arguments */
    for (int i = 1; i < argc; i++)
    {
        if (strcmp(argv[i], "-k") == 0)
        {
            k = atoi(argv[++i]);
        }
        else if (strcmp(argv[i], "-p") == 0)
        {
            p = atoi(argv[++i]);
        }
        else
        {
            return -1;
        }
    }

    if (k <= 0)
    {
        printf("Number of source buffers (%d) must be > 0\n", k);
        return -1;
    }

    if (p <= 0)
    {
        printf("Number of parity buffers (%d) must be > 0\n", p);
        return -1;
    }

    if (p > 32)
    {
		printf("Number of parity buffers (%d) must be <= 32\n", p);
		return -1;
    }

	if (k + p > 255)
	{
		printf("Total number of buffers (%d) must be <= 255\n", k + p);
		return -1;
	}

	printf("K=%d p=%d\n", k, p);

	// Allocate buffers for testing
    PCECCMalloc (k, p, BUFFER_SIZE);

    // Print high level metrics and compute bytes for encoding/decoding
    std::cout << "Threads: " << THDS << " Blocks:" << BLKS << std::endl;
    float bytes = (float)STRD * THDS * BLKS * (k+p);

    // Capture start timestamp
    auto start = std::chrono::steady_clock::now();

    // --------------------------------------------------------------------------
    // CUDA Encoder test
    // --------------------------------------------------------------------------
    ParallelLFSRSequencer_CUDA(BLKS, THDS, k, p, 0);

    cudaStatus = cudaDeviceSynchronize();
    if (cudaStatus != cudaSuccess)
    {
        fprintf(stderr, "cudaDeviceSynchronize returned error code %d after launching Kernel!\n", cudaStatus);
        return 1;
    }

    // Capture end timestamp
    auto end = std::chrono::steady_clock::now();

    // Print rates for user
    ReportRate(bytes, "CUDA Encode ", start, end);

    start = std::chrono::steady_clock::now();

    // --------------------------------------------------------------------------
    // GFNI Encoder test
    // --------------------------------------------------------------------------
    ParallelLFSRSequencer_GFNI(BUFFER_SIZE, k, p, (unsigned char **)HOSTBUFS, 0);

    end = std::chrono::steady_clock::now();

    ReportRate(bytes, "GFNI Encode ", start, end);

    // --------------------------------------------------------------------------
	// Inject errors into the codeword buffers
    // --------------------------------------------------------------------------
    InjectErrors();

    start = std::chrono::steady_clock::now();

    // --------------------------------------------------------------------------
    // CUDA Decoder test
    // --------------------------------------------------------------------------
    ParallelLFSRSequencer_CUDA(BLKS, THDS, k, p, 1);

    cudaStatus = cudaDeviceSynchronize();
    if (cudaStatus != cudaSuccess)
    {
        fprintf(stderr, "cudaDeviceSynchronize returned error code %d after launching Kernel!\n", cudaStatus);
        return 1;
    }

    // Capture end timestamp
    end = std::chrono::steady_clock::now();

    ReportRate(bytes, "CUDA Decode ", start, end);

    // Copy the pointer array directly to the constant memory symbol
    cudaStatus = cudaMemcpyFromSymbol(&herror_count, error_count, sizeof(int));
    if (cudaStatus != cudaSuccess)
    {
        fprintf(stderr, "cudaMemcpyFromSymbol CWP failed!");
        return 1;
    }
	printf("CUDA Error count: %d\n", herror_count);

    start = std::chrono::steady_clock::now();

    // --------------------------------------------------------------------------
    // GFNI Decoder test
    // --------------------------------------------------------------------------
    ParallelLFSRSequencer_GFNI(BUFFER_SIZE, k, p, (unsigned char**)HOSTBUFS, 1);

    // Capture end timestamp
    end = std::chrono::steady_clock::now();

    ReportRate(bytes, "GFNIcode ", start, end);

    printf("GFNI Error Count: %d\n", herror_count);

    // --------------------------------------------------------------------------
	// All test done, free memory and reset device
    // --------------------------------------------------------------------------
    PCECCFree (k, p);

    // cudaDeviceReset must be called before exiting in order for profiling and
    // tracing tools such as Nsight and Visual Profiler to show complete traces.
    cudaStatus = cudaDeviceReset();
    if (cudaStatus != cudaSuccess) 
    {
        fprintf(stderr, "cudaDeviceReset failed!");
        return 1;
    }

    return 0;
}
