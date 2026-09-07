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

#ifndef DECODER_CUDA_CUH
#define DECODER_CUDA_CUH

#define gf_mul_CUDA(a, b) (((a) == 0 || (b) == 0) ? 0 : \
        gf_exp_CUDA[(gf_log_CUDA[a] + gf_log_CUDA[b]) % 255])

#define gf_div_CUDA(a, b) (((a) == 0 || (b) == 0) ? 0 : \
        gf_mul_CUDA(gf_exp_CUDA[255 - gf_log_CUDA[b]], (a)))

__constant__ unsigned char gf_exp_CUDA[256];  // Device side copy of GF power table
__constant__ unsigned char gf_log_CUDA[256];  // Device side copy of GF log table
/* ------------------------------------------------------------------------------------ */
/*                                  CUDA Section                                        */
/* ------------------------------------------------------------------------------------ */

__constant__ unsigned int* CWP[255];  // Device side copy of GPU buffers

__device__ int error_count = 0;               // Device side error count for decoder

#endif // DECODER_CUDA_CUH