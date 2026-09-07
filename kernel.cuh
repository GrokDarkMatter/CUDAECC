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

#ifndef KERNEL_CUH
#define KERNEL_CUH

#include <stdio.h>
#include <iostream>
#include <chrono>

// GFNI includes
#include "DECODER_GFNI.h"

/* ------------------------------------------------------------------------------------ */
/*                                  Global defines                                      */
/* ------------------------------------------------------------------------------------ */
// Size of each buffer in megabytes. Adjust this value to change the buffer size.
#define MEG (16)

// Total number of bytes in each buffer. This is calculated as BUFFER_SIZE = 1024 * 1024 * MEG.
#define BUFFER_SIZE (1024 * 1024 * MEG)

// Size of thread group - 256 * 4 bytes per stride = 1024 bytes per thread group.
#define THDS 256

// Total number of blocks to launch. 
#define BLKS (MEG * 1024)

// CUDA uses 32 bit registers, stride is 4
#define STRD 4

// GFNI decoder in C language for testing
extern "C" void ParallelLFSRSequencer_GFNI(int size, int k, int p, unsigned char** buffers, int decode);

#endif // KERNEL_CUH