#pragma nv_diag_suppress 68

// Inline the error decoder
#include "DECODER_CUDA.cu"

// ----------------------------------------------
// Parallel LFSR Sequencer for P = 2 Codewords
// ----------------------------------------------
__global__ void ParallelLFSRSequencer2_CUDA(int k, int decoder)
{
    // Find out place in the grid of threads
    int i = blockIdx.x * blockDim.x + threadIdx.x;

    // Pointer to develop original data address
    unsigned int* loc ;

    // Create the parity values, initialize to zero
    unsigned int P0 = 0 ;
    unsigned int P1 = 0 ;

    // Temporary values for self reciprocol polynomial
    unsigned int T0;

    // Overflow value for LFSR and carry mask
    unsigned int OV, MSK, OF, OOV ;

    // Adjust K if we are decoding
    k += decoder ? 2 : 0 ;

    // Loop through each symbol
    for ( int sym = 0 ; sym < k ; sym ++ )
    {
        loc = CWP[sym] + i;
        OV = *loc ^ P0 ;
        OOV = OV ;

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

        // Now process the LFSR
        P0 = P1 ^ T0 ;
        P1 = OOV ;
    }

    if ( decoder == 0 )
    {
        // Now store computed check symbols back to memory
        loc = CWP[k+0] + i;
        *loc = P0 ;
        loc = CWP[k+1] + i;
        *loc = P1 ;
    }
    else
    {
        // Now check for parity equal to zero
        T0 = P0;
        T0 |= P1;
        if (T0 != 0)
        {
            unsigned char Sr[2], S[2], errLocs[2], errMags[2] ;
            unsigned char Lambda[2+1], B[2+1], T[2+1], Omega[2+1] ;
            
            if ( T0 & (0xff << 0) )
            {
                atomicAdd(&error_count, 1);
                Sr[0] = (P0>>0) & 0xff ;
                Sr[1] = (P1>>0) & 0xff ;
                rs_decode_palindromic_CUDA(Sr, S, 2, k, errLocs, errMags,
                                           0+(i<<2), Lambda, B, T, Omega) ;
            }
            if ( T0 & (0xff << 8) )
            {
                atomicAdd(&error_count, 1);
                Sr[0] = (P0>>8) & 0xff ;
                Sr[1] = (P1>>8) & 0xff ;
                rs_decode_palindromic_CUDA(Sr, S, 2, k, errLocs, errMags,
                                           1+(i<<2), Lambda, B, T, Omega) ;
            }
            if ( T0 & (0xff << 16) )
            {
                atomicAdd(&error_count, 1);
                Sr[0] = (P0>>16) & 0xff ;
                Sr[1] = (P1>>16) & 0xff ;
                rs_decode_palindromic_CUDA(Sr, S, 2, k, errLocs, errMags,
                                           2+(i<<2), Lambda, B, T, Omega) ;
            }
            if ( T0 & (0xff << 24) )
            {
                atomicAdd(&error_count, 1);
                Sr[0] = (P0>>24) & 0xff ;
                Sr[1] = (P1>>24) & 0xff ;
                rs_decode_palindromic_CUDA(Sr, S, 2, k, errLocs, errMags,
                                           3+(i<<2), Lambda, B, T, Omega) ;
            }
        }
    }
}

// ----------------------------------------------
// Parallel LFSR Sequencer for P = 3 Codewords
// ----------------------------------------------
__global__ void ParallelLFSRSequencer3_CUDA(int k, int decoder)
{
    // Find out place in the grid of threads
    int i = blockIdx.x * blockDim.x + threadIdx.x;

    // Pointer to develop original data address
    unsigned int* loc ;

    // Create the parity values, initialize to zero
    unsigned int P0 = 0 ;
    unsigned int P1 = 0 ;
    unsigned int P2 = 0 ;

    // Temporary values for self reciprocol polynomial
    unsigned int T0;

    // Overflow value for LFSR and carry mask
    unsigned int OV, MSK, OF, OOV ;

    // Adjust K if we are decoding
    k += decoder ? 3 : 0 ;

    // Loop through each symbol
    for ( int sym = 0 ; sym < k ; sym ++ )
    {
        loc = CWP[sym] + i;
        OV = *loc ^ P0 ;
        OOV = OV ;

        // Process bit 0
        T0  = OV;                      // Account for bit 0 of constant 8d
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
        T0 ^= OV;                      // Account for bit 2 of constant 8d
        MSK = OV & 0x80808080 ;        // Create a mask of the MSBs
        OF = (MSK >> 7) * 0x1d ;       // Account for Overflow
        OV <<= 1 ;                     // Multiply the original data by 2
        OV &= 0xfefefefe ;             // Clear the MSBs
        OV ^= OF ;                     // Add in the overflow mask

        // Process bit 3
        T0 ^= OV;                      // Account for bit 3 of constant 8d
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
        MSK = OV & 0x80808080 ;        // Create a mask of the MSBs
        OF = (MSK >> 7) * 0x1d ;       // Account for Overflow
        OV <<= 1 ;                     // Multiply the original data by 2
        OV &= 0xfefefefe ;             // Clear the MSBs
        OV ^= OF ;                     // Add in the overflow mask

        // Process bit 7
        T0 ^= OV;                      // Account for bit 7 of constant 8d

        // Now process the LFSR
        P0 = P1 ^ T0 ;
        P1 = P2 ^ T0 ;
        P2 = OOV ;
    }

    if ( decoder == 0 )
    {
        // Now store computed check symbols back to memory
        loc = CWP[k+0] + i;
        *loc = P0 ;
        loc = CWP[k+1] + i;
        *loc = P1 ;
        loc = CWP[k+2] + i;
        *loc = P2 ;
    }
    else
    {
        // Now check for parity equal to zero
        T0 = P0;
        T0 |= P1;
        T0 |= P2;
        if (T0 != 0)
        {
            unsigned char Sr[3], S[3], errLocs[3], errMags[3] ;
            unsigned char Lambda[3+1], B[3+1], T[3+1], Omega[3+1] ;
            
            if ( T0 & (0xff << 0) )
            {
                atomicAdd(&error_count, 1);
                Sr[0] = (P0>>0) & 0xff ;
                Sr[1] = (P1>>0) & 0xff ;
                Sr[2] = (P2>>0) & 0xff ;
                rs_decode_palindromic_CUDA(Sr, S, 3, k, errLocs, errMags,
                                           0+(i<<2), Lambda, B, T, Omega) ;
            }
            if ( T0 & (0xff << 8) )
            {
                atomicAdd(&error_count, 1);
                Sr[0] = (P0>>8) & 0xff ;
                Sr[1] = (P1>>8) & 0xff ;
                Sr[2] = (P2>>8) & 0xff ;
                rs_decode_palindromic_CUDA(Sr, S, 3, k, errLocs, errMags,
                                           1+(i<<2), Lambda, B, T, Omega) ;
            }
            if ( T0 & (0xff << 16) )
            {
                atomicAdd(&error_count, 1);
                Sr[0] = (P0>>16) & 0xff ;
                Sr[1] = (P1>>16) & 0xff ;
                Sr[2] = (P2>>16) & 0xff ;
                rs_decode_palindromic_CUDA(Sr, S, 3, k, errLocs, errMags,
                                           2+(i<<2), Lambda, B, T, Omega) ;
            }
            if ( T0 & (0xff << 24) )
            {
                atomicAdd(&error_count, 1);
                Sr[0] = (P0>>24) & 0xff ;
                Sr[1] = (P1>>24) & 0xff ;
                Sr[2] = (P2>>24) & 0xff ;
                rs_decode_palindromic_CUDA(Sr, S, 3, k, errLocs, errMags,
                                           3+(i<<2), Lambda, B, T, Omega) ;
            }
        }
    }
}

// ----------------------------------------------
// Parallel LFSR Sequencer for P = 4 Codewords
// ----------------------------------------------
__global__ void ParallelLFSRSequencer4_CUDA(int k, int decoder)
{
    // Find out place in the grid of threads
    int i = blockIdx.x * blockDim.x + threadIdx.x;

    // Pointer to develop original data address
    unsigned int* loc ;

    // Create the parity values, initialize to zero
    unsigned int P0 = 0 ;
    unsigned int P1 = 0 ;
    unsigned int P2 = 0 ;
    unsigned int P3 = 0 ;

    // Temporary values for self reciprocol polynomial
    unsigned int T0;
    unsigned int T1;

    // Overflow value for LFSR and carry mask
    unsigned int OV, MSK, OF, OOV ;

    // Adjust K if we are decoding
    k += decoder ? 4 : 0 ;

    // Loop through each symbol
    for ( int sym = 0 ; sym < k ; sym ++ )
    {
        loc = CWP[sym] + i;
        OV = *loc ^ P0 ;
        OOV = OV ;

        // Process bit 0
        T1  = OV;                      // Account for bit 0 of constant cf
        MSK = OV & 0x80808080 ;        // Create a mask of the MSBs
        OF = (MSK >> 7) * 0x1d ;       // Account for Overflow
        OV <<= 1 ;                     // Multiply the original data by 2
        OV &= 0xfefefefe ;             // Clear the MSBs
        OV ^= OF ;                     // Add in the overflow mask

        // Process bit 1
        T1 ^= OV;                      // Account for bit 1 of constant cf
        MSK = OV & 0x80808080 ;        // Create a mask of the MSBs
        OF = (MSK >> 7) * 0x1d ;       // Account for Overflow
        OV <<= 1 ;                     // Multiply the original data by 2
        OV &= 0xfefefefe ;             // Clear the MSBs
        OV ^= OF ;                     // Add in the overflow mask

        // Process bit 2
        T1 ^= OV;                      // Account for bit 2 of constant cf
        MSK = OV & 0x80808080 ;        // Create a mask of the MSBs
        OF = (MSK >> 7) * 0x1d ;       // Account for Overflow
        OV <<= 1 ;                     // Multiply the original data by 2
        OV &= 0xfefefefe ;             // Clear the MSBs
        OV ^= OF ;                     // Add in the overflow mask

        // Process bit 3
        T0  = OV;                      // Account for bit 3 of constant 38
        T1 ^= OV;                      // Account for bit 3 of constant cf
        MSK = OV & 0x80808080 ;        // Create a mask of the MSBs
        OF = (MSK >> 7) * 0x1d ;       // Account for Overflow
        OV <<= 1 ;                     // Multiply the original data by 2
        OV &= 0xfefefefe ;             // Clear the MSBs
        OV ^= OF ;                     // Add in the overflow mask

        // Process bit 4
        T0 ^= OV;                      // Account for bit 4 of constant 38
        MSK = OV & 0x80808080 ;        // Create a mask of the MSBs
        OF = (MSK >> 7) * 0x1d ;       // Account for Overflow
        OV <<= 1 ;                     // Multiply the original data by 2
        OV &= 0xfefefefe ;             // Clear the MSBs
        OV ^= OF ;                     // Add in the overflow mask

        // Process bit 5
        T0 ^= OV;                      // Account for bit 5 of constant 38
        MSK = OV & 0x80808080 ;        // Create a mask of the MSBs
        OF = (MSK >> 7) * 0x1d ;       // Account for Overflow
        OV <<= 1 ;                     // Multiply the original data by 2
        OV &= 0xfefefefe ;             // Clear the MSBs
        OV ^= OF ;                     // Add in the overflow mask

        // Process bit 6
        T1 ^= OV;                      // Account for bit 6 of constant cf
        MSK = OV & 0x80808080 ;        // Create a mask of the MSBs
        OF = (MSK >> 7) * 0x1d ;       // Account for Overflow
        OV <<= 1 ;                     // Multiply the original data by 2
        OV &= 0xfefefefe ;             // Clear the MSBs
        OV ^= OF ;                     // Add in the overflow mask

        // Process bit 7
        T1 ^= OV;                      // Account for bit 7 of constant cf

        // Now process the LFSR
        P0 = P1 ^ T0 ;
        P1 = P2 ^ T1 ;
        P2 = P3 ^ T0 ;
        P3 = OOV ;
    }

    if ( decoder == 0 )
    {
        // Now store computed check symbols back to memory
        loc = CWP[k+0] + i;
        *loc = P0 ;
        loc = CWP[k+1] + i;
        *loc = P1 ;
        loc = CWP[k+2] + i;
        *loc = P2 ;
        loc = CWP[k+3] + i;
        *loc = P3 ;
    }
    else
    {
        // Now check for parity equal to zero
        T0 = P0;
        T0 |= P1;
        T0 |= P2;
        T0 |= P3;
        if (T0 != 0)
        {
            unsigned char Sr[4], S[4], errLocs[4], errMags[4] ;
            unsigned char Lambda[4+1], B[4+1], T[4+1], Omega[4+1] ;
            
            if ( T0 & (0xff << 0) )
            {
                atomicAdd(&error_count, 1);
                Sr[0] = (P0>>0) & 0xff ;
                Sr[1] = (P1>>0) & 0xff ;
                Sr[2] = (P2>>0) & 0xff ;
                Sr[3] = (P3>>0) & 0xff ;
                rs_decode_palindromic_CUDA(Sr, S, 4, k, errLocs, errMags,
                                           0+(i<<2), Lambda, B, T, Omega) ;
            }
            if ( T0 & (0xff << 8) )
            {
                atomicAdd(&error_count, 1);
                Sr[0] = (P0>>8) & 0xff ;
                Sr[1] = (P1>>8) & 0xff ;
                Sr[2] = (P2>>8) & 0xff ;
                Sr[3] = (P3>>8) & 0xff ;
                rs_decode_palindromic_CUDA(Sr, S, 4, k, errLocs, errMags,
                                           1+(i<<2), Lambda, B, T, Omega) ;
            }
            if ( T0 & (0xff << 16) )
            {
                atomicAdd(&error_count, 1);
                Sr[0] = (P0>>16) & 0xff ;
                Sr[1] = (P1>>16) & 0xff ;
                Sr[2] = (P2>>16) & 0xff ;
                Sr[3] = (P3>>16) & 0xff ;
                rs_decode_palindromic_CUDA(Sr, S, 4, k, errLocs, errMags,
                                           2+(i<<2), Lambda, B, T, Omega) ;
            }
            if ( T0 & (0xff << 24) )
            {
                atomicAdd(&error_count, 1);
                Sr[0] = (P0>>24) & 0xff ;
                Sr[1] = (P1>>24) & 0xff ;
                Sr[2] = (P2>>24) & 0xff ;
                Sr[3] = (P3>>24) & 0xff ;
                rs_decode_palindromic_CUDA(Sr, S, 4, k, errLocs, errMags,
                                           3+(i<<2), Lambda, B, T, Omega) ;
            }
        }
    }
}

// ----------------------------------------------
// Parallel LFSR Sequencer for P = 5 Codewords
// ----------------------------------------------
__global__ void ParallelLFSRSequencer5_CUDA(int k, int decoder)
{
    // Find out place in the grid of threads
    int i = blockIdx.x * blockDim.x + threadIdx.x;

    // Pointer to develop original data address
    unsigned int* loc ;

    // Create the parity values, initialize to zero
    unsigned int P0 = 0 ;
    unsigned int P1 = 0 ;
    unsigned int P2 = 0 ;
    unsigned int P3 = 0 ;
    unsigned int P4 = 0 ;

    // Temporary values for self reciprocol polynomial
    unsigned int T0;
    unsigned int T1;

    // Overflow value for LFSR and carry mask
    unsigned int OV, MSK, OF, OOV ;

    // Adjust K if we are decoding
    k += decoder ? 5 : 0 ;

    // Loop through each symbol
    for ( int sym = 0 ; sym < k ; sym ++ )
    {
        loc = CWP[sym] + i;
        OV = *loc ^ P0 ;
        OOV = OV ;

        // Process bit 0
        MSK = OV & 0x80808080 ;        // Create a mask of the MSBs
        OF = (MSK >> 7) * 0x1d ;       // Account for Overflow
        OV <<= 1 ;                     // Multiply the original data by 2
        OV &= 0xfefefefe ;             // Clear the MSBs
        OV ^= OF ;                     // Add in the overflow mask

        // Process bit 1
        T0  = OV;                      // Account for bit 1 of constant ce
        T1  = OV;                      // Account for bit 1 of constant e6
        MSK = OV & 0x80808080 ;        // Create a mask of the MSBs
        OF = (MSK >> 7) * 0x1d ;       // Account for Overflow
        OV <<= 1 ;                     // Multiply the original data by 2
        OV &= 0xfefefefe ;             // Clear the MSBs
        OV ^= OF ;                     // Add in the overflow mask

        // Process bit 2
        T0 ^= OV;                      // Account for bit 2 of constant ce
        T1 ^= OV;                      // Account for bit 2 of constant e6
        MSK = OV & 0x80808080 ;        // Create a mask of the MSBs
        OF = (MSK >> 7) * 0x1d ;       // Account for Overflow
        OV <<= 1 ;                     // Multiply the original data by 2
        OV &= 0xfefefefe ;             // Clear the MSBs
        OV ^= OF ;                     // Add in the overflow mask

        // Process bit 3
        T0 ^= OV;                      // Account for bit 3 of constant ce
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
        T1 ^= OV;                      // Account for bit 5 of constant e6
        MSK = OV & 0x80808080 ;        // Create a mask of the MSBs
        OF = (MSK >> 7) * 0x1d ;       // Account for Overflow
        OV <<= 1 ;                     // Multiply the original data by 2
        OV &= 0xfefefefe ;             // Clear the MSBs
        OV ^= OF ;                     // Add in the overflow mask

        // Process bit 6
        T0 ^= OV;                      // Account for bit 6 of constant ce
        T1 ^= OV;                      // Account for bit 6 of constant e6
        MSK = OV & 0x80808080 ;        // Create a mask of the MSBs
        OF = (MSK >> 7) * 0x1d ;       // Account for Overflow
        OV <<= 1 ;                     // Multiply the original data by 2
        OV &= 0xfefefefe ;             // Clear the MSBs
        OV ^= OF ;                     // Add in the overflow mask

        // Process bit 7
        T0 ^= OV;                      // Account for bit 7 of constant ce
        T1 ^= OV;                      // Account for bit 7 of constant e6

        // Now process the LFSR
        P0 = P1 ^ T0 ;
        P1 = P2 ^ T1 ;
        P2 = P3 ^ T1 ;
        P3 = P4 ^ T0 ;
        P4 = OOV ;
    }

    if ( decoder == 0 )
    {
        // Now store computed check symbols back to memory
        loc = CWP[k+0] + i;
        *loc = P0 ;
        loc = CWP[k+1] + i;
        *loc = P1 ;
        loc = CWP[k+2] + i;
        *loc = P2 ;
        loc = CWP[k+3] + i;
        *loc = P3 ;
        loc = CWP[k+4] + i;
        *loc = P4 ;
    }
    else
    {
        // Now check for parity equal to zero
        T0 = P0;
        T0 |= P1;
        T0 |= P2;
        T0 |= P3;
        T0 |= P4;
        if (T0 != 0)
        {
            unsigned char Sr[5], S[5], errLocs[5], errMags[5] ;
            unsigned char Lambda[5+1], B[5+1], T[5+1], Omega[5+1] ;
            
            if ( T0 & (0xff << 0) )
            {
                atomicAdd(&error_count, 1);
                Sr[0] = (P0>>0) & 0xff ;
                Sr[1] = (P1>>0) & 0xff ;
                Sr[2] = (P2>>0) & 0xff ;
                Sr[3] = (P3>>0) & 0xff ;
                Sr[4] = (P4>>0) & 0xff ;
                rs_decode_palindromic_CUDA(Sr, S, 5, k, errLocs, errMags,
                                           0+(i<<2), Lambda, B, T, Omega) ;
            }
            if ( T0 & (0xff << 8) )
            {
                atomicAdd(&error_count, 1);
                Sr[0] = (P0>>8) & 0xff ;
                Sr[1] = (P1>>8) & 0xff ;
                Sr[2] = (P2>>8) & 0xff ;
                Sr[3] = (P3>>8) & 0xff ;
                Sr[4] = (P4>>8) & 0xff ;
                rs_decode_palindromic_CUDA(Sr, S, 5, k, errLocs, errMags,
                                           1+(i<<2), Lambda, B, T, Omega) ;
            }
            if ( T0 & (0xff << 16) )
            {
                atomicAdd(&error_count, 1);
                Sr[0] = (P0>>16) & 0xff ;
                Sr[1] = (P1>>16) & 0xff ;
                Sr[2] = (P2>>16) & 0xff ;
                Sr[3] = (P3>>16) & 0xff ;
                Sr[4] = (P4>>16) & 0xff ;
                rs_decode_palindromic_CUDA(Sr, S, 5, k, errLocs, errMags,
                                           2+(i<<2), Lambda, B, T, Omega) ;
            }
            if ( T0 & (0xff << 24) )
            {
                atomicAdd(&error_count, 1);
                Sr[0] = (P0>>24) & 0xff ;
                Sr[1] = (P1>>24) & 0xff ;
                Sr[2] = (P2>>24) & 0xff ;
                Sr[3] = (P3>>24) & 0xff ;
                Sr[4] = (P4>>24) & 0xff ;
                rs_decode_palindromic_CUDA(Sr, S, 5, k, errLocs, errMags,
                                           3+(i<<2), Lambda, B, T, Omega) ;
            }
        }
    }
}

// ----------------------------------------------
// Parallel LFSR Sequencer for P = 6 Codewords
// ----------------------------------------------
__global__ void ParallelLFSRSequencer6_CUDA(int k, int decoder)
{
    // Find out place in the grid of threads
    int i = blockIdx.x * blockDim.x + threadIdx.x;

    // Pointer to develop original data address
    unsigned int* loc ;

    // Create the parity values, initialize to zero
    unsigned int P0 = 0 ;
    unsigned int P1 = 0 ;
    unsigned int P2 = 0 ;
    unsigned int P3 = 0 ;
    unsigned int P4 = 0 ;
    unsigned int P5 = 0 ;

    // Temporary values for self reciprocol polynomial
    unsigned int T0;
    unsigned int T1;
    unsigned int T2;

    // Overflow value for LFSR and carry mask
    unsigned int OV, MSK, OF, OOV ;

    // Adjust K if we are decoding
    k += decoder ? 6 : 0 ;

    // Loop through each symbol
    for ( int sym = 0 ; sym < k ; sym ++ )
    {
        loc = CWP[sym] + i;
        OV = *loc ^ P0 ;
        OOV = OV ;

        // Process bit 0
        T0  = OV;                      // Account for bit 0 of constant 25
        MSK = OV & 0x80808080 ;        // Create a mask of the MSBs
        OF = (MSK >> 7) * 0x1d ;       // Account for Overflow
        OV <<= 1 ;                     // Multiply the original data by 2
        OV &= 0xfefefefe ;             // Clear the MSBs
        OV ^= OF ;                     // Add in the overflow mask

        // Process bit 1
        T2  = OV;                      // Account for bit 1 of constant 8e
        MSK = OV & 0x80808080 ;        // Create a mask of the MSBs
        OF = (MSK >> 7) * 0x1d ;       // Account for Overflow
        OV <<= 1 ;                     // Multiply the original data by 2
        OV &= 0xfefefefe ;             // Clear the MSBs
        OV ^= OF ;                     // Add in the overflow mask

        // Process bit 2
        T0 ^= OV;                      // Account for bit 2 of constant 25
        T1  = OV;                      // Account for bit 2 of constant 6c
        T2 ^= OV;                      // Account for bit 2 of constant 8e
        MSK = OV & 0x80808080 ;        // Create a mask of the MSBs
        OF = (MSK >> 7) * 0x1d ;       // Account for Overflow
        OV <<= 1 ;                     // Multiply the original data by 2
        OV &= 0xfefefefe ;             // Clear the MSBs
        OV ^= OF ;                     // Add in the overflow mask

        // Process bit 3
        T1 ^= OV;                      // Account for bit 3 of constant 6c
        T2 ^= OV;                      // Account for bit 3 of constant 8e
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
        T0 ^= OV;                      // Account for bit 5 of constant 25
        T1 ^= OV;                      // Account for bit 5 of constant 6c
        MSK = OV & 0x80808080 ;        // Create a mask of the MSBs
        OF = (MSK >> 7) * 0x1d ;       // Account for Overflow
        OV <<= 1 ;                     // Multiply the original data by 2
        OV &= 0xfefefefe ;             // Clear the MSBs
        OV ^= OF ;                     // Add in the overflow mask

        // Process bit 6
        T1 ^= OV;                      // Account for bit 6 of constant 6c
        MSK = OV & 0x80808080 ;        // Create a mask of the MSBs
        OF = (MSK >> 7) * 0x1d ;       // Account for Overflow
        OV <<= 1 ;                     // Multiply the original data by 2
        OV &= 0xfefefefe ;             // Clear the MSBs
        OV ^= OF ;                     // Add in the overflow mask

        // Process bit 7
        T2 ^= OV;                      // Account for bit 7 of constant 8e

        // Now process the LFSR
        P0 = P1 ^ T0 ;
        P1 = P2 ^ T1 ;
        P2 = P3 ^ T2 ;
        P3 = P4 ^ T1 ;
        P4 = P5 ^ T0 ;
        P5 = OOV ;
    }

    if ( decoder == 0 )
    {
        // Now store computed check symbols back to memory
        loc = CWP[k+0] + i;
        *loc = P0 ;
        loc = CWP[k+1] + i;
        *loc = P1 ;
        loc = CWP[k+2] + i;
        *loc = P2 ;
        loc = CWP[k+3] + i;
        *loc = P3 ;
        loc = CWP[k+4] + i;
        *loc = P4 ;
        loc = CWP[k+5] + i;
        *loc = P5 ;
    }
    else
    {
        // Now check for parity equal to zero
        T0 = P0;
        T0 |= P1;
        T0 |= P2;
        T0 |= P3;
        T0 |= P4;
        T0 |= P5;
        if (T0 != 0)
        {
            unsigned char Sr[6], S[6], errLocs[6], errMags[6] ;
            unsigned char Lambda[6+1], B[6+1], T[6+1], Omega[6+1] ;
            
            if ( T0 & (0xff << 0) )
            {
                atomicAdd(&error_count, 1);
                Sr[0] = (P0>>0) & 0xff ;
                Sr[1] = (P1>>0) & 0xff ;
                Sr[2] = (P2>>0) & 0xff ;
                Sr[3] = (P3>>0) & 0xff ;
                Sr[4] = (P4>>0) & 0xff ;
                Sr[5] = (P5>>0) & 0xff ;
                rs_decode_palindromic_CUDA(Sr, S, 6, k, errLocs, errMags,
                                           0+(i<<2), Lambda, B, T, Omega) ;
            }
            if ( T0 & (0xff << 8) )
            {
                atomicAdd(&error_count, 1);
                Sr[0] = (P0>>8) & 0xff ;
                Sr[1] = (P1>>8) & 0xff ;
                Sr[2] = (P2>>8) & 0xff ;
                Sr[3] = (P3>>8) & 0xff ;
                Sr[4] = (P4>>8) & 0xff ;
                Sr[5] = (P5>>8) & 0xff ;
                rs_decode_palindromic_CUDA(Sr, S, 6, k, errLocs, errMags,
                                           1+(i<<2), Lambda, B, T, Omega) ;
            }
            if ( T0 & (0xff << 16) )
            {
                atomicAdd(&error_count, 1);
                Sr[0] = (P0>>16) & 0xff ;
                Sr[1] = (P1>>16) & 0xff ;
                Sr[2] = (P2>>16) & 0xff ;
                Sr[3] = (P3>>16) & 0xff ;
                Sr[4] = (P4>>16) & 0xff ;
                Sr[5] = (P5>>16) & 0xff ;
                rs_decode_palindromic_CUDA(Sr, S, 6, k, errLocs, errMags,
                                           2+(i<<2), Lambda, B, T, Omega) ;
            }
            if ( T0 & (0xff << 24) )
            {
                atomicAdd(&error_count, 1);
                Sr[0] = (P0>>24) & 0xff ;
                Sr[1] = (P1>>24) & 0xff ;
                Sr[2] = (P2>>24) & 0xff ;
                Sr[3] = (P3>>24) & 0xff ;
                Sr[4] = (P4>>24) & 0xff ;
                Sr[5] = (P5>>24) & 0xff ;
                rs_decode_palindromic_CUDA(Sr, S, 6, k, errLocs, errMags,
                                           3+(i<<2), Lambda, B, T, Omega) ;
            }
        }
    }
}

// ----------------------------------------------
// Parallel LFSR Sequencer for P = 7 Codewords
// ----------------------------------------------
__global__ void ParallelLFSRSequencer7_CUDA(int k, int decoder)
{
    // Find out place in the grid of threads
    int i = blockIdx.x * blockDim.x + threadIdx.x;

    // Pointer to develop original data address
    unsigned int* loc ;

    // Create the parity values, initialize to zero
    unsigned int P0 = 0 ;
    unsigned int P1 = 0 ;
    unsigned int P2 = 0 ;
    unsigned int P3 = 0 ;
    unsigned int P4 = 0 ;
    unsigned int P5 = 0 ;
    unsigned int P6 = 0 ;

    // Temporary values for self reciprocol polynomial
    unsigned int T0;
    unsigned int T1;
    unsigned int T2;

    // Overflow value for LFSR and carry mask
    unsigned int OV, MSK, OF, OOV ;

    // Adjust K if we are decoding
    k += decoder ? 7 : 0 ;

    // Loop through each symbol
    for ( int sym = 0 ; sym < k ; sym ++ )
    {
        loc = CWP[sym] + i;
        OV = *loc ^ P0 ;
        OOV = OV ;

        // Process bit 0
        T0  = OV;                      // Account for bit 0 of constant 6b
        T1  = OV;                      // Account for bit 0 of constant 9
        MSK = OV & 0x80808080 ;        // Create a mask of the MSBs
        OF = (MSK >> 7) * 0x1d ;       // Account for Overflow
        OV <<= 1 ;                     // Multiply the original data by 2
        OV &= 0xfefefefe ;             // Clear the MSBs
        OV ^= OF ;                     // Add in the overflow mask

        // Process bit 1
        T0 ^= OV;                      // Account for bit 1 of constant 6b
        T2  = OV;                      // Account for bit 1 of constant 9e
        MSK = OV & 0x80808080 ;        // Create a mask of the MSBs
        OF = (MSK >> 7) * 0x1d ;       // Account for Overflow
        OV <<= 1 ;                     // Multiply the original data by 2
        OV &= 0xfefefefe ;             // Clear the MSBs
        OV ^= OF ;                     // Add in the overflow mask

        // Process bit 2
        T2 ^= OV;                      // Account for bit 2 of constant 9e
        MSK = OV & 0x80808080 ;        // Create a mask of the MSBs
        OF = (MSK >> 7) * 0x1d ;       // Account for Overflow
        OV <<= 1 ;                     // Multiply the original data by 2
        OV &= 0xfefefefe ;             // Clear the MSBs
        OV ^= OF ;                     // Add in the overflow mask

        // Process bit 3
        T0 ^= OV;                      // Account for bit 3 of constant 6b
        T1 ^= OV;                      // Account for bit 3 of constant 9
        T2 ^= OV;                      // Account for bit 3 of constant 9e
        MSK = OV & 0x80808080 ;        // Create a mask of the MSBs
        OF = (MSK >> 7) * 0x1d ;       // Account for Overflow
        OV <<= 1 ;                     // Multiply the original data by 2
        OV &= 0xfefefefe ;             // Clear the MSBs
        OV ^= OF ;                     // Add in the overflow mask

        // Process bit 4
        T2 ^= OV;                      // Account for bit 4 of constant 9e
        MSK = OV & 0x80808080 ;        // Create a mask of the MSBs
        OF = (MSK >> 7) * 0x1d ;       // Account for Overflow
        OV <<= 1 ;                     // Multiply the original data by 2
        OV &= 0xfefefefe ;             // Clear the MSBs
        OV ^= OF ;                     // Add in the overflow mask

        // Process bit 5
        T0 ^= OV;                      // Account for bit 5 of constant 6b
        MSK = OV & 0x80808080 ;        // Create a mask of the MSBs
        OF = (MSK >> 7) * 0x1d ;       // Account for Overflow
        OV <<= 1 ;                     // Multiply the original data by 2
        OV &= 0xfefefefe ;             // Clear the MSBs
        OV ^= OF ;                     // Add in the overflow mask

        // Process bit 6
        T0 ^= OV;                      // Account for bit 6 of constant 6b
        MSK = OV & 0x80808080 ;        // Create a mask of the MSBs
        OF = (MSK >> 7) * 0x1d ;       // Account for Overflow
        OV <<= 1 ;                     // Multiply the original data by 2
        OV &= 0xfefefefe ;             // Clear the MSBs
        OV ^= OF ;                     // Add in the overflow mask

        // Process bit 7
        T2 ^= OV;                      // Account for bit 7 of constant 9e

        // Now process the LFSR
        P0 = P1 ^ T0 ;
        P1 = P2 ^ T1 ;
        P2 = P3 ^ T2 ;
        P3 = P4 ^ T2 ;
        P4 = P5 ^ T1 ;
        P5 = P6 ^ T0 ;
        P6 = OOV ;
    }

    if ( decoder == 0 )
    {
        // Now store computed check symbols back to memory
        loc = CWP[k+0] + i;
        *loc = P0 ;
        loc = CWP[k+1] + i;
        *loc = P1 ;
        loc = CWP[k+2] + i;
        *loc = P2 ;
        loc = CWP[k+3] + i;
        *loc = P3 ;
        loc = CWP[k+4] + i;
        *loc = P4 ;
        loc = CWP[k+5] + i;
        *loc = P5 ;
        loc = CWP[k+6] + i;
        *loc = P6 ;
    }
    else
    {
        // Now check for parity equal to zero
        T0 = P0;
        T0 |= P1;
        T0 |= P2;
        T0 |= P3;
        T0 |= P4;
        T0 |= P5;
        T0 |= P6;
        if (T0 != 0)
        {
            unsigned char Sr[7], S[7], errLocs[7], errMags[7] ;
            unsigned char Lambda[7+1], B[7+1], T[7+1], Omega[7+1] ;
            
            if ( T0 & (0xff << 0) )
            {
                atomicAdd(&error_count, 1);
                Sr[0] = (P0>>0) & 0xff ;
                Sr[1] = (P1>>0) & 0xff ;
                Sr[2] = (P2>>0) & 0xff ;
                Sr[3] = (P3>>0) & 0xff ;
                Sr[4] = (P4>>0) & 0xff ;
                Sr[5] = (P5>>0) & 0xff ;
                Sr[6] = (P6>>0) & 0xff ;
                rs_decode_palindromic_CUDA(Sr, S, 7, k, errLocs, errMags,
                                           0+(i<<2), Lambda, B, T, Omega) ;
            }
            if ( T0 & (0xff << 8) )
            {
                atomicAdd(&error_count, 1);
                Sr[0] = (P0>>8) & 0xff ;
                Sr[1] = (P1>>8) & 0xff ;
                Sr[2] = (P2>>8) & 0xff ;
                Sr[3] = (P3>>8) & 0xff ;
                Sr[4] = (P4>>8) & 0xff ;
                Sr[5] = (P5>>8) & 0xff ;
                Sr[6] = (P6>>8) & 0xff ;
                rs_decode_palindromic_CUDA(Sr, S, 7, k, errLocs, errMags,
                                           1+(i<<2), Lambda, B, T, Omega) ;
            }
            if ( T0 & (0xff << 16) )
            {
                atomicAdd(&error_count, 1);
                Sr[0] = (P0>>16) & 0xff ;
                Sr[1] = (P1>>16) & 0xff ;
                Sr[2] = (P2>>16) & 0xff ;
                Sr[3] = (P3>>16) & 0xff ;
                Sr[4] = (P4>>16) & 0xff ;
                Sr[5] = (P5>>16) & 0xff ;
                Sr[6] = (P6>>16) & 0xff ;
                rs_decode_palindromic_CUDA(Sr, S, 7, k, errLocs, errMags,
                                           2+(i<<2), Lambda, B, T, Omega) ;
            }
            if ( T0 & (0xff << 24) )
            {
                atomicAdd(&error_count, 1);
                Sr[0] = (P0>>24) & 0xff ;
                Sr[1] = (P1>>24) & 0xff ;
                Sr[2] = (P2>>24) & 0xff ;
                Sr[3] = (P3>>24) & 0xff ;
                Sr[4] = (P4>>24) & 0xff ;
                Sr[5] = (P5>>24) & 0xff ;
                Sr[6] = (P6>>24) & 0xff ;
                rs_decode_palindromic_CUDA(Sr, S, 7, k, errLocs, errMags,
                                           3+(i<<2), Lambda, B, T, Omega) ;
            }
        }
    }
}

// ----------------------------------------------
// Parallel LFSR Sequencer for P = 8 Codewords
// ----------------------------------------------
__global__ void ParallelLFSRSequencer8_CUDA(int k, int decoder)
{
    // Find out place in the grid of threads
    int i = blockIdx.x * blockDim.x + threadIdx.x;

    // Pointer to develop original data address
    unsigned int* loc ;

    // Create the parity values, initialize to zero
    unsigned int P0 = 0 ;
    unsigned int P1 = 0 ;
    unsigned int P2 = 0 ;
    unsigned int P3 = 0 ;
    unsigned int P4 = 0 ;
    unsigned int P5 = 0 ;
    unsigned int P6 = 0 ;
    unsigned int P7 = 0 ;

    // Temporary values for self reciprocol polynomial
    unsigned int T0;
    unsigned int T1;
    unsigned int T2;
    unsigned int T3;

    // Overflow value for LFSR and carry mask
    unsigned int OV, MSK, OF, OOV ;

    // Adjust K if we are decoding
    k += decoder ? 8 : 0 ;

    // Loop through each symbol
    for ( int sym = 0 ; sym < k ; sym ++ )
    {
        loc = CWP[sym] + i;
        OV = *loc ^ P0 ;
        OOV = OV ;

        // Process bit 0
        T1  = OV;                      // Account for bit 0 of constant f5
        T3  = OV;                      // Account for bit 0 of constant eb
        MSK = OV & 0x80808080 ;        // Create a mask of the MSBs
        OF = (MSK >> 7) * 0x1d ;       // Account for Overflow
        OV <<= 1 ;                     // Multiply the original data by 2
        OV &= 0xfefefefe ;             // Clear the MSBs
        OV ^= OF ;                     // Add in the overflow mask

        // Process bit 1
        T0  = OV;                      // Account for bit 1 of constant ee
        T2  = OV;                      // Account for bit 1 of constant 5e
        T3 ^= OV;                      // Account for bit 1 of constant eb
        MSK = OV & 0x80808080 ;        // Create a mask of the MSBs
        OF = (MSK >> 7) * 0x1d ;       // Account for Overflow
        OV <<= 1 ;                     // Multiply the original data by 2
        OV &= 0xfefefefe ;             // Clear the MSBs
        OV ^= OF ;                     // Add in the overflow mask

        // Process bit 2
        T0 ^= OV;                      // Account for bit 2 of constant ee
        T1 ^= OV;                      // Account for bit 2 of constant f5
        T2 ^= OV;                      // Account for bit 2 of constant 5e
        MSK = OV & 0x80808080 ;        // Create a mask of the MSBs
        OF = (MSK >> 7) * 0x1d ;       // Account for Overflow
        OV <<= 1 ;                     // Multiply the original data by 2
        OV &= 0xfefefefe ;             // Clear the MSBs
        OV ^= OF ;                     // Add in the overflow mask

        // Process bit 3
        T0 ^= OV;                      // Account for bit 3 of constant ee
        T2 ^= OV;                      // Account for bit 3 of constant 5e
        T3 ^= OV;                      // Account for bit 3 of constant eb
        MSK = OV & 0x80808080 ;        // Create a mask of the MSBs
        OF = (MSK >> 7) * 0x1d ;       // Account for Overflow
        OV <<= 1 ;                     // Multiply the original data by 2
        OV &= 0xfefefefe ;             // Clear the MSBs
        OV ^= OF ;                     // Add in the overflow mask

        // Process bit 4
        T1 ^= OV;                      // Account for bit 4 of constant f5
        T2 ^= OV;                      // Account for bit 4 of constant 5e
        MSK = OV & 0x80808080 ;        // Create a mask of the MSBs
        OF = (MSK >> 7) * 0x1d ;       // Account for Overflow
        OV <<= 1 ;                     // Multiply the original data by 2
        OV &= 0xfefefefe ;             // Clear the MSBs
        OV ^= OF ;                     // Add in the overflow mask

        // Process bit 5
        T0 ^= OV;                      // Account for bit 5 of constant ee
        T1 ^= OV;                      // Account for bit 5 of constant f5
        T3 ^= OV;                      // Account for bit 5 of constant eb
        MSK = OV & 0x80808080 ;        // Create a mask of the MSBs
        OF = (MSK >> 7) * 0x1d ;       // Account for Overflow
        OV <<= 1 ;                     // Multiply the original data by 2
        OV &= 0xfefefefe ;             // Clear the MSBs
        OV ^= OF ;                     // Add in the overflow mask

        // Process bit 6
        T0 ^= OV;                      // Account for bit 6 of constant ee
        T1 ^= OV;                      // Account for bit 6 of constant f5
        T2 ^= OV;                      // Account for bit 6 of constant 5e
        T3 ^= OV;                      // Account for bit 6 of constant eb
        MSK = OV & 0x80808080 ;        // Create a mask of the MSBs
        OF = (MSK >> 7) * 0x1d ;       // Account for Overflow
        OV <<= 1 ;                     // Multiply the original data by 2
        OV &= 0xfefefefe ;             // Clear the MSBs
        OV ^= OF ;                     // Add in the overflow mask

        // Process bit 7
        T0 ^= OV;                      // Account for bit 7 of constant ee
        T1 ^= OV;                      // Account for bit 7 of constant f5
        T3 ^= OV;                      // Account for bit 7 of constant eb

        // Now process the LFSR
        P0 = P1 ^ T0 ;
        P1 = P2 ^ T1 ;
        P2 = P3 ^ T2 ;
        P3 = P4 ^ T3 ;
        P4 = P5 ^ T2 ;
        P5 = P6 ^ T1 ;
        P6 = P7 ^ T0 ;
        P7 = OOV ;
    }

    if ( decoder == 0 )
    {
        // Now store computed check symbols back to memory
        loc = CWP[k+0] + i;
        *loc = P0 ;
        loc = CWP[k+1] + i;
        *loc = P1 ;
        loc = CWP[k+2] + i;
        *loc = P2 ;
        loc = CWP[k+3] + i;
        *loc = P3 ;
        loc = CWP[k+4] + i;
        *loc = P4 ;
        loc = CWP[k+5] + i;
        *loc = P5 ;
        loc = CWP[k+6] + i;
        *loc = P6 ;
        loc = CWP[k+7] + i;
        *loc = P7 ;
    }
    else
    {
        // Now check for parity equal to zero
        T0 = P0;
        T0 |= P1;
        T0 |= P2;
        T0 |= P3;
        T0 |= P4;
        T0 |= P5;
        T0 |= P6;
        T0 |= P7;
        if (T0 != 0)
        {
            unsigned char Sr[8], S[8], errLocs[8], errMags[8] ;
            unsigned char Lambda[8+1], B[8+1], T[8+1], Omega[8+1] ;
            
            if ( T0 & (0xff << 0) )
            {
                atomicAdd(&error_count, 1);
                Sr[0] = (P0>>0) & 0xff ;
                Sr[1] = (P1>>0) & 0xff ;
                Sr[2] = (P2>>0) & 0xff ;
                Sr[3] = (P3>>0) & 0xff ;
                Sr[4] = (P4>>0) & 0xff ;
                Sr[5] = (P5>>0) & 0xff ;
                Sr[6] = (P6>>0) & 0xff ;
                Sr[7] = (P7>>0) & 0xff ;
                rs_decode_palindromic_CUDA(Sr, S, 8, k, errLocs, errMags,
                                           0+(i<<2), Lambda, B, T, Omega) ;
            }
            if ( T0 & (0xff << 8) )
            {
                atomicAdd(&error_count, 1);
                Sr[0] = (P0>>8) & 0xff ;
                Sr[1] = (P1>>8) & 0xff ;
                Sr[2] = (P2>>8) & 0xff ;
                Sr[3] = (P3>>8) & 0xff ;
                Sr[4] = (P4>>8) & 0xff ;
                Sr[5] = (P5>>8) & 0xff ;
                Sr[6] = (P6>>8) & 0xff ;
                Sr[7] = (P7>>8) & 0xff ;
                rs_decode_palindromic_CUDA(Sr, S, 8, k, errLocs, errMags,
                                           1+(i<<2), Lambda, B, T, Omega) ;
            }
            if ( T0 & (0xff << 16) )
            {
                atomicAdd(&error_count, 1);
                Sr[0] = (P0>>16) & 0xff ;
                Sr[1] = (P1>>16) & 0xff ;
                Sr[2] = (P2>>16) & 0xff ;
                Sr[3] = (P3>>16) & 0xff ;
                Sr[4] = (P4>>16) & 0xff ;
                Sr[5] = (P5>>16) & 0xff ;
                Sr[6] = (P6>>16) & 0xff ;
                Sr[7] = (P7>>16) & 0xff ;
                rs_decode_palindromic_CUDA(Sr, S, 8, k, errLocs, errMags,
                                           2+(i<<2), Lambda, B, T, Omega) ;
            }
            if ( T0 & (0xff << 24) )
            {
                atomicAdd(&error_count, 1);
                Sr[0] = (P0>>24) & 0xff ;
                Sr[1] = (P1>>24) & 0xff ;
                Sr[2] = (P2>>24) & 0xff ;
                Sr[3] = (P3>>24) & 0xff ;
                Sr[4] = (P4>>24) & 0xff ;
                Sr[5] = (P5>>24) & 0xff ;
                Sr[6] = (P6>>24) & 0xff ;
                Sr[7] = (P7>>24) & 0xff ;
                rs_decode_palindromic_CUDA(Sr, S, 8, k, errLocs, errMags,
                                           3+(i<<2), Lambda, B, T, Omega) ;
            }
        }
    }
}

// ----------------------------------------------
// Parallel LFSR Sequencer for P = 9 Codewords
// ----------------------------------------------
__global__ void ParallelLFSRSequencer9_CUDA(int k, int decoder)
{
    // Find out place in the grid of threads
    int i = blockIdx.x * blockDim.x + threadIdx.x;

    // Pointer to develop original data address
    unsigned int* loc ;

    // Create the parity values, initialize to zero
    unsigned int P0 = 0 ;
    unsigned int P1 = 0 ;
    unsigned int P2 = 0 ;
    unsigned int P3 = 0 ;
    unsigned int P4 = 0 ;
    unsigned int P5 = 0 ;
    unsigned int P6 = 0 ;
    unsigned int P7 = 0 ;
    unsigned int P8 = 0 ;

    // Temporary values for self reciprocol polynomial
    unsigned int T0;
    unsigned int T1;
    unsigned int T2;
    unsigned int T3;

    // Overflow value for LFSR and carry mask
    unsigned int OV, MSK, OF, OOV ;

    // Adjust K if we are decoding
    k += decoder ? 9 : 0 ;

    // Loop through each symbol
    for ( int sym = 0 ; sym < k ; sym ++ )
    {
        loc = CWP[sym] + i;
        OV = *loc ^ P0 ;
        OOV = OV ;

        // Process bit 0
        T0  = OV;                      // Account for bit 0 of constant a3
        T1  = OV;                      // Account for bit 0 of constant b
        T2  = OV;                      // Account for bit 0 of constant 33
        T3  = OV;                      // Account for bit 0 of constant ef
        MSK = OV & 0x80808080 ;        // Create a mask of the MSBs
        OF = (MSK >> 7) * 0x1d ;       // Account for Overflow
        OV <<= 1 ;                     // Multiply the original data by 2
        OV &= 0xfefefefe ;             // Clear the MSBs
        OV ^= OF ;                     // Add in the overflow mask

        // Process bit 1
        T0 ^= OV;                      // Account for bit 1 of constant a3
        T1 ^= OV;                      // Account for bit 1 of constant b
        T2 ^= OV;                      // Account for bit 1 of constant 33
        T3 ^= OV;                      // Account for bit 1 of constant ef
        MSK = OV & 0x80808080 ;        // Create a mask of the MSBs
        OF = (MSK >> 7) * 0x1d ;       // Account for Overflow
        OV <<= 1 ;                     // Multiply the original data by 2
        OV &= 0xfefefefe ;             // Clear the MSBs
        OV ^= OF ;                     // Add in the overflow mask

        // Process bit 2
        T3 ^= OV;                      // Account for bit 2 of constant ef
        MSK = OV & 0x80808080 ;        // Create a mask of the MSBs
        OF = (MSK >> 7) * 0x1d ;       // Account for Overflow
        OV <<= 1 ;                     // Multiply the original data by 2
        OV &= 0xfefefefe ;             // Clear the MSBs
        OV ^= OF ;                     // Add in the overflow mask

        // Process bit 3
        T1 ^= OV;                      // Account for bit 3 of constant b
        T3 ^= OV;                      // Account for bit 3 of constant ef
        MSK = OV & 0x80808080 ;        // Create a mask of the MSBs
        OF = (MSK >> 7) * 0x1d ;       // Account for Overflow
        OV <<= 1 ;                     // Multiply the original data by 2
        OV &= 0xfefefefe ;             // Clear the MSBs
        OV ^= OF ;                     // Add in the overflow mask

        // Process bit 4
        T2 ^= OV;                      // Account for bit 4 of constant 33
        MSK = OV & 0x80808080 ;        // Create a mask of the MSBs
        OF = (MSK >> 7) * 0x1d ;       // Account for Overflow
        OV <<= 1 ;                     // Multiply the original data by 2
        OV &= 0xfefefefe ;             // Clear the MSBs
        OV ^= OF ;                     // Add in the overflow mask

        // Process bit 5
        T0 ^= OV;                      // Account for bit 5 of constant a3
        T2 ^= OV;                      // Account for bit 5 of constant 33
        T3 ^= OV;                      // Account for bit 5 of constant ef
        MSK = OV & 0x80808080 ;        // Create a mask of the MSBs
        OF = (MSK >> 7) * 0x1d ;       // Account for Overflow
        OV <<= 1 ;                     // Multiply the original data by 2
        OV &= 0xfefefefe ;             // Clear the MSBs
        OV ^= OF ;                     // Add in the overflow mask

        // Process bit 6
        T3 ^= OV;                      // Account for bit 6 of constant ef
        MSK = OV & 0x80808080 ;        // Create a mask of the MSBs
        OF = (MSK >> 7) * 0x1d ;       // Account for Overflow
        OV <<= 1 ;                     // Multiply the original data by 2
        OV &= 0xfefefefe ;             // Clear the MSBs
        OV ^= OF ;                     // Add in the overflow mask

        // Process bit 7
        T0 ^= OV;                      // Account for bit 7 of constant a3
        T3 ^= OV;                      // Account for bit 7 of constant ef

        // Now process the LFSR
        P0 = P1 ^ T0 ;
        P1 = P2 ^ T1 ;
        P2 = P3 ^ T2 ;
        P3 = P4 ^ T3 ;
        P4 = P5 ^ T3 ;
        P5 = P6 ^ T2 ;
        P6 = P7 ^ T1 ;
        P7 = P8 ^ T0 ;
        P8 = OOV ;
    }

    if ( decoder == 0 )
    {
        // Now store computed check symbols back to memory
        loc = CWP[k+0] + i;
        *loc = P0 ;
        loc = CWP[k+1] + i;
        *loc = P1 ;
        loc = CWP[k+2] + i;
        *loc = P2 ;
        loc = CWP[k+3] + i;
        *loc = P3 ;
        loc = CWP[k+4] + i;
        *loc = P4 ;
        loc = CWP[k+5] + i;
        *loc = P5 ;
        loc = CWP[k+6] + i;
        *loc = P6 ;
        loc = CWP[k+7] + i;
        *loc = P7 ;
        loc = CWP[k+8] + i;
        *loc = P8 ;
    }
    else
    {
        // Now check for parity equal to zero
        T0 = P0;
        T0 |= P1;
        T0 |= P2;
        T0 |= P3;
        T0 |= P4;
        T0 |= P5;
        T0 |= P6;
        T0 |= P7;
        T0 |= P8;
        if (T0 != 0)
        {
            unsigned char Sr[9], S[9], errLocs[9], errMags[9] ;
            unsigned char Lambda[9+1], B[9+1], T[9+1], Omega[9+1] ;
            
            if ( T0 & (0xff << 0) )
            {
                atomicAdd(&error_count, 1);
                Sr[0] = (P0>>0) & 0xff ;
                Sr[1] = (P1>>0) & 0xff ;
                Sr[2] = (P2>>0) & 0xff ;
                Sr[3] = (P3>>0) & 0xff ;
                Sr[4] = (P4>>0) & 0xff ;
                Sr[5] = (P5>>0) & 0xff ;
                Sr[6] = (P6>>0) & 0xff ;
                Sr[7] = (P7>>0) & 0xff ;
                Sr[8] = (P8>>0) & 0xff ;
                rs_decode_palindromic_CUDA(Sr, S, 9, k, errLocs, errMags,
                                           0+(i<<2), Lambda, B, T, Omega) ;
            }
            if ( T0 & (0xff << 8) )
            {
                atomicAdd(&error_count, 1);
                Sr[0] = (P0>>8) & 0xff ;
                Sr[1] = (P1>>8) & 0xff ;
                Sr[2] = (P2>>8) & 0xff ;
                Sr[3] = (P3>>8) & 0xff ;
                Sr[4] = (P4>>8) & 0xff ;
                Sr[5] = (P5>>8) & 0xff ;
                Sr[6] = (P6>>8) & 0xff ;
                Sr[7] = (P7>>8) & 0xff ;
                Sr[8] = (P8>>8) & 0xff ;
                rs_decode_palindromic_CUDA(Sr, S, 9, k, errLocs, errMags,
                                           1+(i<<2), Lambda, B, T, Omega) ;
            }
            if ( T0 & (0xff << 16) )
            {
                atomicAdd(&error_count, 1);
                Sr[0] = (P0>>16) & 0xff ;
                Sr[1] = (P1>>16) & 0xff ;
                Sr[2] = (P2>>16) & 0xff ;
                Sr[3] = (P3>>16) & 0xff ;
                Sr[4] = (P4>>16) & 0xff ;
                Sr[5] = (P5>>16) & 0xff ;
                Sr[6] = (P6>>16) & 0xff ;
                Sr[7] = (P7>>16) & 0xff ;
                Sr[8] = (P8>>16) & 0xff ;
                rs_decode_palindromic_CUDA(Sr, S, 9, k, errLocs, errMags,
                                           2+(i<<2), Lambda, B, T, Omega) ;
            }
            if ( T0 & (0xff << 24) )
            {
                atomicAdd(&error_count, 1);
                Sr[0] = (P0>>24) & 0xff ;
                Sr[1] = (P1>>24) & 0xff ;
                Sr[2] = (P2>>24) & 0xff ;
                Sr[3] = (P3>>24) & 0xff ;
                Sr[4] = (P4>>24) & 0xff ;
                Sr[5] = (P5>>24) & 0xff ;
                Sr[6] = (P6>>24) & 0xff ;
                Sr[7] = (P7>>24) & 0xff ;
                Sr[8] = (P8>>24) & 0xff ;
                rs_decode_palindromic_CUDA(Sr, S, 9, k, errLocs, errMags,
                                           3+(i<<2), Lambda, B, T, Omega) ;
            }
        }
    }
}

// ----------------------------------------------
// Parallel LFSR Sequencer for P = 10 Codewords
// ----------------------------------------------
__global__ void ParallelLFSRSequencer10_CUDA(int k, int decoder)
{
    // Find out place in the grid of threads
    int i = blockIdx.x * blockDim.x + threadIdx.x;

    // Pointer to develop original data address
    unsigned int* loc ;

    // Create the parity values, initialize to zero
    unsigned int P0 = 0 ;
    unsigned int P1 = 0 ;
    unsigned int P2 = 0 ;
    unsigned int P3 = 0 ;
    unsigned int P4 = 0 ;
    unsigned int P5 = 0 ;
    unsigned int P6 = 0 ;
    unsigned int P7 = 0 ;
    unsigned int P8 = 0 ;
    unsigned int P9 = 0 ;

    // Temporary values for self reciprocol polynomial
    unsigned int T0;
    unsigned int T1;
    unsigned int T2;
    unsigned int T3;
    unsigned int T4;

    // Overflow value for LFSR and carry mask
    unsigned int OV, MSK, OF, OOV ;

    // Adjust K if we are decoding
    k += decoder ? 10 : 0 ;

    // Loop through each symbol
    for ( int sym = 0 ; sym < k ; sym ++ )
    {
        loc = CWP[sym] + i;
        OV = *loc ^ P0 ;
        OOV = OV ;

        // Process bit 0
        T0  = OV;                      // Account for bit 0 of constant 93
        T1  = OV;                      // Account for bit 0 of constant 69
        T3  = OV;                      // Account for bit 0 of constant 77
        T4  = OV;                      // Account for bit 0 of constant 9
        MSK = OV & 0x80808080 ;        // Create a mask of the MSBs
        OF = (MSK >> 7) * 0x1d ;       // Account for Overflow
        OV <<= 1 ;                     // Multiply the original data by 2
        OV &= 0xfefefefe ;             // Clear the MSBs
        OV ^= OF ;                     // Add in the overflow mask

        // Process bit 1
        T0 ^= OV;                      // Account for bit 1 of constant 93
        T2  = OV;                      // Account for bit 1 of constant e6
        T3 ^= OV;                      // Account for bit 1 of constant 77
        MSK = OV & 0x80808080 ;        // Create a mask of the MSBs
        OF = (MSK >> 7) * 0x1d ;       // Account for Overflow
        OV <<= 1 ;                     // Multiply the original data by 2
        OV &= 0xfefefefe ;             // Clear the MSBs
        OV ^= OF ;                     // Add in the overflow mask

        // Process bit 2
        T2 ^= OV;                      // Account for bit 2 of constant e6
        T3 ^= OV;                      // Account for bit 2 of constant 77
        MSK = OV & 0x80808080 ;        // Create a mask of the MSBs
        OF = (MSK >> 7) * 0x1d ;       // Account for Overflow
        OV <<= 1 ;                     // Multiply the original data by 2
        OV &= 0xfefefefe ;             // Clear the MSBs
        OV ^= OF ;                     // Add in the overflow mask

        // Process bit 3
        T1 ^= OV;                      // Account for bit 3 of constant 69
        T4 ^= OV;                      // Account for bit 3 of constant 9
        MSK = OV & 0x80808080 ;        // Create a mask of the MSBs
        OF = (MSK >> 7) * 0x1d ;       // Account for Overflow
        OV <<= 1 ;                     // Multiply the original data by 2
        OV &= 0xfefefefe ;             // Clear the MSBs
        OV ^= OF ;                     // Add in the overflow mask

        // Process bit 4
        T0 ^= OV;                      // Account for bit 4 of constant 93
        T3 ^= OV;                      // Account for bit 4 of constant 77
        MSK = OV & 0x80808080 ;        // Create a mask of the MSBs
        OF = (MSK >> 7) * 0x1d ;       // Account for Overflow
        OV <<= 1 ;                     // Multiply the original data by 2
        OV &= 0xfefefefe ;             // Clear the MSBs
        OV ^= OF ;                     // Add in the overflow mask

        // Process bit 5
        T1 ^= OV;                      // Account for bit 5 of constant 69
        T2 ^= OV;                      // Account for bit 5 of constant e6
        T3 ^= OV;                      // Account for bit 5 of constant 77
        MSK = OV & 0x80808080 ;        // Create a mask of the MSBs
        OF = (MSK >> 7) * 0x1d ;       // Account for Overflow
        OV <<= 1 ;                     // Multiply the original data by 2
        OV &= 0xfefefefe ;             // Clear the MSBs
        OV ^= OF ;                     // Add in the overflow mask

        // Process bit 6
        T1 ^= OV;                      // Account for bit 6 of constant 69
        T2 ^= OV;                      // Account for bit 6 of constant e6
        T3 ^= OV;                      // Account for bit 6 of constant 77
        MSK = OV & 0x80808080 ;        // Create a mask of the MSBs
        OF = (MSK >> 7) * 0x1d ;       // Account for Overflow
        OV <<= 1 ;                     // Multiply the original data by 2
        OV &= 0xfefefefe ;             // Clear the MSBs
        OV ^= OF ;                     // Add in the overflow mask

        // Process bit 7
        T0 ^= OV;                      // Account for bit 7 of constant 93
        T2 ^= OV;                      // Account for bit 7 of constant e6

        // Now process the LFSR
        P0 = P1 ^ T0 ;
        P1 = P2 ^ T1 ;
        P2 = P3 ^ T2 ;
        P3 = P4 ^ T3 ;
        P4 = P5 ^ T4 ;
        P5 = P6 ^ T3 ;
        P6 = P7 ^ T2 ;
        P7 = P8 ^ T1 ;
        P8 = P9 ^ T0 ;
        P9 = OOV ;
    }

    if ( decoder == 0 )
    {
        // Now store computed check symbols back to memory
        loc = CWP[k+0] + i;
        *loc = P0 ;
        loc = CWP[k+1] + i;
        *loc = P1 ;
        loc = CWP[k+2] + i;
        *loc = P2 ;
        loc = CWP[k+3] + i;
        *loc = P3 ;
        loc = CWP[k+4] + i;
        *loc = P4 ;
        loc = CWP[k+5] + i;
        *loc = P5 ;
        loc = CWP[k+6] + i;
        *loc = P6 ;
        loc = CWP[k+7] + i;
        *loc = P7 ;
        loc = CWP[k+8] + i;
        *loc = P8 ;
        loc = CWP[k+9] + i;
        *loc = P9 ;
    }
    else
    {
        // Now check for parity equal to zero
        T0 = P0;
        T0 |= P1;
        T0 |= P2;
        T0 |= P3;
        T0 |= P4;
        T0 |= P5;
        T0 |= P6;
        T0 |= P7;
        T0 |= P8;
        T0 |= P9;
        if (T0 != 0)
        {
            unsigned char Sr[10], S[10], errLocs[10], errMags[10] ;
            unsigned char Lambda[10+1], B[10+1], T[10+1], Omega[10+1] ;
            
            if ( T0 & (0xff << 0) )
            {
                atomicAdd(&error_count, 1);
                Sr[0] = (P0>>0) & 0xff ;
                Sr[1] = (P1>>0) & 0xff ;
                Sr[2] = (P2>>0) & 0xff ;
                Sr[3] = (P3>>0) & 0xff ;
                Sr[4] = (P4>>0) & 0xff ;
                Sr[5] = (P5>>0) & 0xff ;
                Sr[6] = (P6>>0) & 0xff ;
                Sr[7] = (P7>>0) & 0xff ;
                Sr[8] = (P8>>0) & 0xff ;
                Sr[9] = (P9>>0) & 0xff ;
                rs_decode_palindromic_CUDA(Sr, S, 10, k, errLocs, errMags,
                                           0+(i<<2), Lambda, B, T, Omega) ;
            }
            if ( T0 & (0xff << 8) )
            {
                atomicAdd(&error_count, 1);
                Sr[0] = (P0>>8) & 0xff ;
                Sr[1] = (P1>>8) & 0xff ;
                Sr[2] = (P2>>8) & 0xff ;
                Sr[3] = (P3>>8) & 0xff ;
                Sr[4] = (P4>>8) & 0xff ;
                Sr[5] = (P5>>8) & 0xff ;
                Sr[6] = (P6>>8) & 0xff ;
                Sr[7] = (P7>>8) & 0xff ;
                Sr[8] = (P8>>8) & 0xff ;
                Sr[9] = (P9>>8) & 0xff ;
                rs_decode_palindromic_CUDA(Sr, S, 10, k, errLocs, errMags,
                                           1+(i<<2), Lambda, B, T, Omega) ;
            }
            if ( T0 & (0xff << 16) )
            {
                atomicAdd(&error_count, 1);
                Sr[0] = (P0>>16) & 0xff ;
                Sr[1] = (P1>>16) & 0xff ;
                Sr[2] = (P2>>16) & 0xff ;
                Sr[3] = (P3>>16) & 0xff ;
                Sr[4] = (P4>>16) & 0xff ;
                Sr[5] = (P5>>16) & 0xff ;
                Sr[6] = (P6>>16) & 0xff ;
                Sr[7] = (P7>>16) & 0xff ;
                Sr[8] = (P8>>16) & 0xff ;
                Sr[9] = (P9>>16) & 0xff ;
                rs_decode_palindromic_CUDA(Sr, S, 10, k, errLocs, errMags,
                                           2+(i<<2), Lambda, B, T, Omega) ;
            }
            if ( T0 & (0xff << 24) )
            {
                atomicAdd(&error_count, 1);
                Sr[0] = (P0>>24) & 0xff ;
                Sr[1] = (P1>>24) & 0xff ;
                Sr[2] = (P2>>24) & 0xff ;
                Sr[3] = (P3>>24) & 0xff ;
                Sr[4] = (P4>>24) & 0xff ;
                Sr[5] = (P5>>24) & 0xff ;
                Sr[6] = (P6>>24) & 0xff ;
                Sr[7] = (P7>>24) & 0xff ;
                Sr[8] = (P8>>24) & 0xff ;
                Sr[9] = (P9>>24) & 0xff ;
                rs_decode_palindromic_CUDA(Sr, S, 10, k, errLocs, errMags,
                                           3+(i<<2), Lambda, B, T, Omega) ;
            }
        }
    }
}

// ----------------------------------------------
// Parallel LFSR Sequencer for P = 11 Codewords
// ----------------------------------------------
__global__ void ParallelLFSRSequencer11_CUDA(int k, int decoder)
{
    // Find out place in the grid of threads
    int i = blockIdx.x * blockDim.x + threadIdx.x;

    // Pointer to develop original data address
    unsigned int* loc ;

    // Create the parity values, initialize to zero
    unsigned int P0 = 0 ;
    unsigned int P1 = 0 ;
    unsigned int P2 = 0 ;
    unsigned int P3 = 0 ;
    unsigned int P4 = 0 ;
    unsigned int P5 = 0 ;
    unsigned int P6 = 0 ;
    unsigned int P7 = 0 ;
    unsigned int P8 = 0 ;
    unsigned int P9 = 0 ;
    unsigned int P10 = 0 ;

    // Temporary values for self reciprocol polynomial
    unsigned int T0;
    unsigned int T1;
    unsigned int T2;
    unsigned int T3;
    unsigned int T4;

    // Overflow value for LFSR and carry mask
    unsigned int OV, MSK, OF, OOV ;

    // Adjust K if we are decoding
    k += decoder ? 11 : 0 ;

    // Loop through each symbol
    for ( int sym = 0 ; sym < k ; sym ++ )
    {
        loc = CWP[sym] + i;
        OV = *loc ^ P0 ;
        OOV = OV ;

        // Process bit 0
        T0  = OV;                      // Account for bit 0 of constant ef
        T3  = OV;                      // Account for bit 0 of constant f1
        T4  = OV;                      // Account for bit 0 of constant 29
        MSK = OV & 0x80808080 ;        // Create a mask of the MSBs
        OF = (MSK >> 7) * 0x1d ;       // Account for Overflow
        OV <<= 1 ;                     // Multiply the original data by 2
        OV &= 0xfefefefe ;             // Clear the MSBs
        OV ^= OF ;                     // Add in the overflow mask

        // Process bit 1
        T0 ^= OV;                      // Account for bit 1 of constant ef
        T1  = OV;                      // Account for bit 1 of constant 62
        T2  = OV;                      // Account for bit 1 of constant 1e
        MSK = OV & 0x80808080 ;        // Create a mask of the MSBs
        OF = (MSK >> 7) * 0x1d ;       // Account for Overflow
        OV <<= 1 ;                     // Multiply the original data by 2
        OV &= 0xfefefefe ;             // Clear the MSBs
        OV ^= OF ;                     // Add in the overflow mask

        // Process bit 2
        T0 ^= OV;                      // Account for bit 2 of constant ef
        T2 ^= OV;                      // Account for bit 2 of constant 1e
        MSK = OV & 0x80808080 ;        // Create a mask of the MSBs
        OF = (MSK >> 7) * 0x1d ;       // Account for Overflow
        OV <<= 1 ;                     // Multiply the original data by 2
        OV &= 0xfefefefe ;             // Clear the MSBs
        OV ^= OF ;                     // Add in the overflow mask

        // Process bit 3
        T0 ^= OV;                      // Account for bit 3 of constant ef
        T2 ^= OV;                      // Account for bit 3 of constant 1e
        T4 ^= OV;                      // Account for bit 3 of constant 29
        MSK = OV & 0x80808080 ;        // Create a mask of the MSBs
        OF = (MSK >> 7) * 0x1d ;       // Account for Overflow
        OV <<= 1 ;                     // Multiply the original data by 2
        OV &= 0xfefefefe ;             // Clear the MSBs
        OV ^= OF ;                     // Add in the overflow mask

        // Process bit 4
        T2 ^= OV;                      // Account for bit 4 of constant 1e
        T3 ^= OV;                      // Account for bit 4 of constant f1
        MSK = OV & 0x80808080 ;        // Create a mask of the MSBs
        OF = (MSK >> 7) * 0x1d ;       // Account for Overflow
        OV <<= 1 ;                     // Multiply the original data by 2
        OV &= 0xfefefefe ;             // Clear the MSBs
        OV ^= OF ;                     // Add in the overflow mask

        // Process bit 5
        T0 ^= OV;                      // Account for bit 5 of constant ef
        T1 ^= OV;                      // Account for bit 5 of constant 62
        T3 ^= OV;                      // Account for bit 5 of constant f1
        T4 ^= OV;                      // Account for bit 5 of constant 29
        MSK = OV & 0x80808080 ;        // Create a mask of the MSBs
        OF = (MSK >> 7) * 0x1d ;       // Account for Overflow
        OV <<= 1 ;                     // Multiply the original data by 2
        OV &= 0xfefefefe ;             // Clear the MSBs
        OV ^= OF ;                     // Add in the overflow mask

        // Process bit 6
        T0 ^= OV;                      // Account for bit 6 of constant ef
        T1 ^= OV;                      // Account for bit 6 of constant 62
        T3 ^= OV;                      // Account for bit 6 of constant f1
        MSK = OV & 0x80808080 ;        // Create a mask of the MSBs
        OF = (MSK >> 7) * 0x1d ;       // Account for Overflow
        OV <<= 1 ;                     // Multiply the original data by 2
        OV &= 0xfefefefe ;             // Clear the MSBs
        OV ^= OF ;                     // Add in the overflow mask

        // Process bit 7
        T0 ^= OV;                      // Account for bit 7 of constant ef
        T3 ^= OV;                      // Account for bit 7 of constant f1

        // Now process the LFSR
        P0 = P1 ^ T0 ;
        P1 = P2 ^ T1 ;
        P2 = P3 ^ T2 ;
        P3 = P4 ^ T3 ;
        P4 = P5 ^ T4 ;
        P5 = P6 ^ T4 ;
        P6 = P7 ^ T3 ;
        P7 = P8 ^ T2 ;
        P8 = P9 ^ T1 ;
        P9 = P10 ^ T0 ;
        P10 = OOV ;
    }

    if ( decoder == 0 )
    {
        // Now store computed check symbols back to memory
        loc = CWP[k+0] + i;
        *loc = P0 ;
        loc = CWP[k+1] + i;
        *loc = P1 ;
        loc = CWP[k+2] + i;
        *loc = P2 ;
        loc = CWP[k+3] + i;
        *loc = P3 ;
        loc = CWP[k+4] + i;
        *loc = P4 ;
        loc = CWP[k+5] + i;
        *loc = P5 ;
        loc = CWP[k+6] + i;
        *loc = P6 ;
        loc = CWP[k+7] + i;
        *loc = P7 ;
        loc = CWP[k+8] + i;
        *loc = P8 ;
        loc = CWP[k+9] + i;
        *loc = P9 ;
        loc = CWP[k+10] + i;
        *loc = P10 ;
    }
    else
    {
        // Now check for parity equal to zero
        T0 = P0;
        T0 |= P1;
        T0 |= P2;
        T0 |= P3;
        T0 |= P4;
        T0 |= P5;
        T0 |= P6;
        T0 |= P7;
        T0 |= P8;
        T0 |= P9;
        T0 |= P10;
        if (T0 != 0)
        {
            unsigned char Sr[11], S[11], errLocs[11], errMags[11] ;
            unsigned char Lambda[11+1], B[11+1], T[11+1], Omega[11+1] ;
            
            if ( T0 & (0xff << 0) )
            {
                atomicAdd(&error_count, 1);
                Sr[0] = (P0>>0) & 0xff ;
                Sr[1] = (P1>>0) & 0xff ;
                Sr[2] = (P2>>0) & 0xff ;
                Sr[3] = (P3>>0) & 0xff ;
                Sr[4] = (P4>>0) & 0xff ;
                Sr[5] = (P5>>0) & 0xff ;
                Sr[6] = (P6>>0) & 0xff ;
                Sr[7] = (P7>>0) & 0xff ;
                Sr[8] = (P8>>0) & 0xff ;
                Sr[9] = (P9>>0) & 0xff ;
                Sr[10] = (P10>>0) & 0xff ;
                rs_decode_palindromic_CUDA(Sr, S, 11, k, errLocs, errMags,
                                           0+(i<<2), Lambda, B, T, Omega) ;
            }
            if ( T0 & (0xff << 8) )
            {
                atomicAdd(&error_count, 1);
                Sr[0] = (P0>>8) & 0xff ;
                Sr[1] = (P1>>8) & 0xff ;
                Sr[2] = (P2>>8) & 0xff ;
                Sr[3] = (P3>>8) & 0xff ;
                Sr[4] = (P4>>8) & 0xff ;
                Sr[5] = (P5>>8) & 0xff ;
                Sr[6] = (P6>>8) & 0xff ;
                Sr[7] = (P7>>8) & 0xff ;
                Sr[8] = (P8>>8) & 0xff ;
                Sr[9] = (P9>>8) & 0xff ;
                Sr[10] = (P10>>8) & 0xff ;
                rs_decode_palindromic_CUDA(Sr, S, 11, k, errLocs, errMags,
                                           1+(i<<2), Lambda, B, T, Omega) ;
            }
            if ( T0 & (0xff << 16) )
            {
                atomicAdd(&error_count, 1);
                Sr[0] = (P0>>16) & 0xff ;
                Sr[1] = (P1>>16) & 0xff ;
                Sr[2] = (P2>>16) & 0xff ;
                Sr[3] = (P3>>16) & 0xff ;
                Sr[4] = (P4>>16) & 0xff ;
                Sr[5] = (P5>>16) & 0xff ;
                Sr[6] = (P6>>16) & 0xff ;
                Sr[7] = (P7>>16) & 0xff ;
                Sr[8] = (P8>>16) & 0xff ;
                Sr[9] = (P9>>16) & 0xff ;
                Sr[10] = (P10>>16) & 0xff ;
                rs_decode_palindromic_CUDA(Sr, S, 11, k, errLocs, errMags,
                                           2+(i<<2), Lambda, B, T, Omega) ;
            }
            if ( T0 & (0xff << 24) )
            {
                atomicAdd(&error_count, 1);
                Sr[0] = (P0>>24) & 0xff ;
                Sr[1] = (P1>>24) & 0xff ;
                Sr[2] = (P2>>24) & 0xff ;
                Sr[3] = (P3>>24) & 0xff ;
                Sr[4] = (P4>>24) & 0xff ;
                Sr[5] = (P5>>24) & 0xff ;
                Sr[6] = (P6>>24) & 0xff ;
                Sr[7] = (P7>>24) & 0xff ;
                Sr[8] = (P8>>24) & 0xff ;
                Sr[9] = (P9>>24) & 0xff ;
                Sr[10] = (P10>>24) & 0xff ;
                rs_decode_palindromic_CUDA(Sr, S, 11, k, errLocs, errMags,
                                           3+(i<<2), Lambda, B, T, Omega) ;
            }
        }
    }
}

// ----------------------------------------------
// Parallel LFSR Sequencer for P = 12 Codewords
// ----------------------------------------------
__global__ void ParallelLFSRSequencer12_CUDA(int k, int decoder)
{
    // Find out place in the grid of threads
    int i = blockIdx.x * blockDim.x + threadIdx.x;

    // Pointer to develop original data address
    unsigned int* loc ;

    // Create the parity values, initialize to zero
    unsigned int P0 = 0 ;
    unsigned int P1 = 0 ;
    unsigned int P2 = 0 ;
    unsigned int P3 = 0 ;
    unsigned int P4 = 0 ;
    unsigned int P5 = 0 ;
    unsigned int P6 = 0 ;
    unsigned int P7 = 0 ;
    unsigned int P8 = 0 ;
    unsigned int P9 = 0 ;
    unsigned int P10 = 0 ;
    unsigned int P11 = 0 ;

    // Temporary values for self reciprocol polynomial
    unsigned int T0;
    unsigned int T1;
    unsigned int T2;
    unsigned int T3;
    unsigned int T4;
    unsigned int T5;

    // Overflow value for LFSR and carry mask
    unsigned int OV, MSK, OF, OOV ;

    // Adjust K if we are decoding
    k += decoder ? 12 : 0 ;

    // Loop through each symbol
    for ( int sym = 0 ; sym < k ; sym ++ )
    {
        loc = CWP[sym] + i;
        OV = *loc ^ P0 ;
        OOV = OV ;

        // Process bit 0
        T1  = OV;                      // Account for bit 0 of constant 9d
        T4  = OV;                      // Account for bit 0 of constant 9d
        T5  = OV;                      // Account for bit 0 of constant fd
        MSK = OV & 0x80808080 ;        // Create a mask of the MSBs
        OF = (MSK >> 7) * 0x1d ;       // Account for Overflow
        OV <<= 1 ;                     // Multiply the original data by 2
        OV &= 0xfefefefe ;             // Clear the MSBs
        OV ^= OF ;                     // Add in the overflow mask

        // Process bit 1
        T0  = OV;                      // Account for bit 1 of constant 12
        T2  = OV;                      // Account for bit 1 of constant a2
        T3  = OV;                      // Account for bit 1 of constant 86
        MSK = OV & 0x80808080 ;        // Create a mask of the MSBs
        OF = (MSK >> 7) * 0x1d ;       // Account for Overflow
        OV <<= 1 ;                     // Multiply the original data by 2
        OV &= 0xfefefefe ;             // Clear the MSBs
        OV ^= OF ;                     // Add in the overflow mask

        // Process bit 2
        T1 ^= OV;                      // Account for bit 2 of constant 9d
        T3 ^= OV;                      // Account for bit 2 of constant 86
        T4 ^= OV;                      // Account for bit 2 of constant 9d
        T5 ^= OV;                      // Account for bit 2 of constant fd
        MSK = OV & 0x80808080 ;        // Create a mask of the MSBs
        OF = (MSK >> 7) * 0x1d ;       // Account for Overflow
        OV <<= 1 ;                     // Multiply the original data by 2
        OV &= 0xfefefefe ;             // Clear the MSBs
        OV ^= OF ;                     // Add in the overflow mask

        // Process bit 3
        T1 ^= OV;                      // Account for bit 3 of constant 9d
        T4 ^= OV;                      // Account for bit 3 of constant 9d
        T5 ^= OV;                      // Account for bit 3 of constant fd
        MSK = OV & 0x80808080 ;        // Create a mask of the MSBs
        OF = (MSK >> 7) * 0x1d ;       // Account for Overflow
        OV <<= 1 ;                     // Multiply the original data by 2
        OV &= 0xfefefefe ;             // Clear the MSBs
        OV ^= OF ;                     // Add in the overflow mask

        // Process bit 4
        T0 ^= OV;                      // Account for bit 4 of constant 12
        T1 ^= OV;                      // Account for bit 4 of constant 9d
        T4 ^= OV;                      // Account for bit 4 of constant 9d
        T5 ^= OV;                      // Account for bit 4 of constant fd
        MSK = OV & 0x80808080 ;        // Create a mask of the MSBs
        OF = (MSK >> 7) * 0x1d ;       // Account for Overflow
        OV <<= 1 ;                     // Multiply the original data by 2
        OV &= 0xfefefefe ;             // Clear the MSBs
        OV ^= OF ;                     // Add in the overflow mask

        // Process bit 5
        T2 ^= OV;                      // Account for bit 5 of constant a2
        T5 ^= OV;                      // Account for bit 5 of constant fd
        MSK = OV & 0x80808080 ;        // Create a mask of the MSBs
        OF = (MSK >> 7) * 0x1d ;       // Account for Overflow
        OV <<= 1 ;                     // Multiply the original data by 2
        OV &= 0xfefefefe ;             // Clear the MSBs
        OV ^= OF ;                     // Add in the overflow mask

        // Process bit 6
        T5 ^= OV;                      // Account for bit 6 of constant fd
        MSK = OV & 0x80808080 ;        // Create a mask of the MSBs
        OF = (MSK >> 7) * 0x1d ;       // Account for Overflow
        OV <<= 1 ;                     // Multiply the original data by 2
        OV &= 0xfefefefe ;             // Clear the MSBs
        OV ^= OF ;                     // Add in the overflow mask

        // Process bit 7
        T1 ^= OV;                      // Account for bit 7 of constant 9d
        T2 ^= OV;                      // Account for bit 7 of constant a2
        T3 ^= OV;                      // Account for bit 7 of constant 86
        T4 ^= OV;                      // Account for bit 7 of constant 9d
        T5 ^= OV;                      // Account for bit 7 of constant fd

        // Now process the LFSR
        P0 = P1 ^ T0 ;
        P1 = P2 ^ T1 ;
        P2 = P3 ^ T2 ;
        P3 = P4 ^ T3 ;
        P4 = P5 ^ T4 ;
        P5 = P6 ^ T5 ;
        P6 = P7 ^ T4 ;
        P7 = P8 ^ T3 ;
        P8 = P9 ^ T2 ;
        P9 = P10 ^ T1 ;
        P10 = P11 ^ T0 ;
        P11 = OOV ;
    }

    if ( decoder == 0 )
    {
        // Now store computed check symbols back to memory
        loc = CWP[k+0] + i;
        *loc = P0 ;
        loc = CWP[k+1] + i;
        *loc = P1 ;
        loc = CWP[k+2] + i;
        *loc = P2 ;
        loc = CWP[k+3] + i;
        *loc = P3 ;
        loc = CWP[k+4] + i;
        *loc = P4 ;
        loc = CWP[k+5] + i;
        *loc = P5 ;
        loc = CWP[k+6] + i;
        *loc = P6 ;
        loc = CWP[k+7] + i;
        *loc = P7 ;
        loc = CWP[k+8] + i;
        *loc = P8 ;
        loc = CWP[k+9] + i;
        *loc = P9 ;
        loc = CWP[k+10] + i;
        *loc = P10 ;
        loc = CWP[k+11] + i;
        *loc = P11 ;
    }
    else
    {
        // Now check for parity equal to zero
        T0 = P0;
        T0 |= P1;
        T0 |= P2;
        T0 |= P3;
        T0 |= P4;
        T0 |= P5;
        T0 |= P6;
        T0 |= P7;
        T0 |= P8;
        T0 |= P9;
        T0 |= P10;
        T0 |= P11;
        if (T0 != 0)
        {
            unsigned char Sr[12], S[12], errLocs[12], errMags[12] ;
            unsigned char Lambda[12+1], B[12+1], T[12+1], Omega[12+1] ;
            
            if ( T0 & (0xff << 0) )
            {
                atomicAdd(&error_count, 1);
                Sr[0] = (P0>>0) & 0xff ;
                Sr[1] = (P1>>0) & 0xff ;
                Sr[2] = (P2>>0) & 0xff ;
                Sr[3] = (P3>>0) & 0xff ;
                Sr[4] = (P4>>0) & 0xff ;
                Sr[5] = (P5>>0) & 0xff ;
                Sr[6] = (P6>>0) & 0xff ;
                Sr[7] = (P7>>0) & 0xff ;
                Sr[8] = (P8>>0) & 0xff ;
                Sr[9] = (P9>>0) & 0xff ;
                Sr[10] = (P10>>0) & 0xff ;
                Sr[11] = (P11>>0) & 0xff ;
                rs_decode_palindromic_CUDA(Sr, S, 12, k, errLocs, errMags,
                                           0+(i<<2), Lambda, B, T, Omega) ;
            }
            if ( T0 & (0xff << 8) )
            {
                atomicAdd(&error_count, 1);
                Sr[0] = (P0>>8) & 0xff ;
                Sr[1] = (P1>>8) & 0xff ;
                Sr[2] = (P2>>8) & 0xff ;
                Sr[3] = (P3>>8) & 0xff ;
                Sr[4] = (P4>>8) & 0xff ;
                Sr[5] = (P5>>8) & 0xff ;
                Sr[6] = (P6>>8) & 0xff ;
                Sr[7] = (P7>>8) & 0xff ;
                Sr[8] = (P8>>8) & 0xff ;
                Sr[9] = (P9>>8) & 0xff ;
                Sr[10] = (P10>>8) & 0xff ;
                Sr[11] = (P11>>8) & 0xff ;
                rs_decode_palindromic_CUDA(Sr, S, 12, k, errLocs, errMags,
                                           1+(i<<2), Lambda, B, T, Omega) ;
            }
            if ( T0 & (0xff << 16) )
            {
                atomicAdd(&error_count, 1);
                Sr[0] = (P0>>16) & 0xff ;
                Sr[1] = (P1>>16) & 0xff ;
                Sr[2] = (P2>>16) & 0xff ;
                Sr[3] = (P3>>16) & 0xff ;
                Sr[4] = (P4>>16) & 0xff ;
                Sr[5] = (P5>>16) & 0xff ;
                Sr[6] = (P6>>16) & 0xff ;
                Sr[7] = (P7>>16) & 0xff ;
                Sr[8] = (P8>>16) & 0xff ;
                Sr[9] = (P9>>16) & 0xff ;
                Sr[10] = (P10>>16) & 0xff ;
                Sr[11] = (P11>>16) & 0xff ;
                rs_decode_palindromic_CUDA(Sr, S, 12, k, errLocs, errMags,
                                           2+(i<<2), Lambda, B, T, Omega) ;
            }
            if ( T0 & (0xff << 24) )
            {
                atomicAdd(&error_count, 1);
                Sr[0] = (P0>>24) & 0xff ;
                Sr[1] = (P1>>24) & 0xff ;
                Sr[2] = (P2>>24) & 0xff ;
                Sr[3] = (P3>>24) & 0xff ;
                Sr[4] = (P4>>24) & 0xff ;
                Sr[5] = (P5>>24) & 0xff ;
                Sr[6] = (P6>>24) & 0xff ;
                Sr[7] = (P7>>24) & 0xff ;
                Sr[8] = (P8>>24) & 0xff ;
                Sr[9] = (P9>>24) & 0xff ;
                Sr[10] = (P10>>24) & 0xff ;
                Sr[11] = (P11>>24) & 0xff ;
                rs_decode_palindromic_CUDA(Sr, S, 12, k, errLocs, errMags,
                                           3+(i<<2), Lambda, B, T, Omega) ;
            }
        }
    }
}

// ----------------------------------------------
// Parallel LFSR Sequencer for P = 13 Codewords
// ----------------------------------------------
__global__ void ParallelLFSRSequencer13_CUDA(int k, int decoder)
{
    // Find out place in the grid of threads
    int i = blockIdx.x * blockDim.x + threadIdx.x;

    // Pointer to develop original data address
    unsigned int* loc ;

    // Create the parity values, initialize to zero
    unsigned int P0 = 0 ;
    unsigned int P1 = 0 ;
    unsigned int P2 = 0 ;
    unsigned int P3 = 0 ;
    unsigned int P4 = 0 ;
    unsigned int P5 = 0 ;
    unsigned int P6 = 0 ;
    unsigned int P7 = 0 ;
    unsigned int P8 = 0 ;
    unsigned int P9 = 0 ;
    unsigned int P10 = 0 ;
    unsigned int P11 = 0 ;
    unsigned int P12 = 0 ;

    // Temporary values for self reciprocol polynomial
    unsigned int T0;
    unsigned int T1;
    unsigned int T2;
    unsigned int T3;
    unsigned int T4;
    unsigned int T5;

    // Overflow value for LFSR and carry mask
    unsigned int OV, MSK, OF, OOV ;

    // Adjust K if we are decoding
    k += decoder ? 13 : 0 ;

    // Loop through each symbol
    for ( int sym = 0 ; sym < k ; sym ++ )
    {
        loc = CWP[sym] + i;
        OV = *loc ^ P0 ;
        OOV = OV ;

        // Process bit 0
        T0  = OV;                      // Account for bit 0 of constant 99
        T2  = OV;                      // Account for bit 0 of constant b7
        T4  = OV;                      // Account for bit 0 of constant 5d
        T5  = OV;                      // Account for bit 0 of constant 5f
        MSK = OV & 0x80808080 ;        // Create a mask of the MSBs
        OF = (MSK >> 7) * 0x1d ;       // Account for Overflow
        OV <<= 1 ;                     // Multiply the original data by 2
        OV &= 0xfefefefe ;             // Clear the MSBs
        OV ^= OF ;                     // Add in the overflow mask

        // Process bit 1
        T2 ^= OV;                      // Account for bit 1 of constant b7
        T3  = OV;                      // Account for bit 1 of constant 1e
        T5 ^= OV;                      // Account for bit 1 of constant 5f
        MSK = OV & 0x80808080 ;        // Create a mask of the MSBs
        OF = (MSK >> 7) * 0x1d ;       // Account for Overflow
        OV <<= 1 ;                     // Multiply the original data by 2
        OV &= 0xfefefefe ;             // Clear the MSBs
        OV ^= OF ;                     // Add in the overflow mask

        // Process bit 2
        T1  = OV;                      // Account for bit 2 of constant 84
        T2 ^= OV;                      // Account for bit 2 of constant b7
        T3 ^= OV;                      // Account for bit 2 of constant 1e
        T4 ^= OV;                      // Account for bit 2 of constant 5d
        T5 ^= OV;                      // Account for bit 2 of constant 5f
        MSK = OV & 0x80808080 ;        // Create a mask of the MSBs
        OF = (MSK >> 7) * 0x1d ;       // Account for Overflow
        OV <<= 1 ;                     // Multiply the original data by 2
        OV &= 0xfefefefe ;             // Clear the MSBs
        OV ^= OF ;                     // Add in the overflow mask

        // Process bit 3
        T0 ^= OV;                      // Account for bit 3 of constant 99
        T3 ^= OV;                      // Account for bit 3 of constant 1e
        T4 ^= OV;                      // Account for bit 3 of constant 5d
        T5 ^= OV;                      // Account for bit 3 of constant 5f
        MSK = OV & 0x80808080 ;        // Create a mask of the MSBs
        OF = (MSK >> 7) * 0x1d ;       // Account for Overflow
        OV <<= 1 ;                     // Multiply the original data by 2
        OV &= 0xfefefefe ;             // Clear the MSBs
        OV ^= OF ;                     // Add in the overflow mask

        // Process bit 4
        T0 ^= OV;                      // Account for bit 4 of constant 99
        T2 ^= OV;                      // Account for bit 4 of constant b7
        T3 ^= OV;                      // Account for bit 4 of constant 1e
        T4 ^= OV;                      // Account for bit 4 of constant 5d
        T5 ^= OV;                      // Account for bit 4 of constant 5f
        MSK = OV & 0x80808080 ;        // Create a mask of the MSBs
        OF = (MSK >> 7) * 0x1d ;       // Account for Overflow
        OV <<= 1 ;                     // Multiply the original data by 2
        OV &= 0xfefefefe ;             // Clear the MSBs
        OV ^= OF ;                     // Add in the overflow mask

        // Process bit 5
        T2 ^= OV;                      // Account for bit 5 of constant b7
        MSK = OV & 0x80808080 ;        // Create a mask of the MSBs
        OF = (MSK >> 7) * 0x1d ;       // Account for Overflow
        OV <<= 1 ;                     // Multiply the original data by 2
        OV &= 0xfefefefe ;             // Clear the MSBs
        OV ^= OF ;                     // Add in the overflow mask

        // Process bit 6
        T4 ^= OV;                      // Account for bit 6 of constant 5d
        T5 ^= OV;                      // Account for bit 6 of constant 5f
        MSK = OV & 0x80808080 ;        // Create a mask of the MSBs
        OF = (MSK >> 7) * 0x1d ;       // Account for Overflow
        OV <<= 1 ;                     // Multiply the original data by 2
        OV &= 0xfefefefe ;             // Clear the MSBs
        OV ^= OF ;                     // Add in the overflow mask

        // Process bit 7
        T0 ^= OV;                      // Account for bit 7 of constant 99
        T1 ^= OV;                      // Account for bit 7 of constant 84
        T2 ^= OV;                      // Account for bit 7 of constant b7

        // Now process the LFSR
        P0 = P1 ^ T0 ;
        P1 = P2 ^ T1 ;
        P2 = P3 ^ T2 ;
        P3 = P4 ^ T3 ;
        P4 = P5 ^ T4 ;
        P5 = P6 ^ T5 ;
        P6 = P7 ^ T5 ;
        P7 = P8 ^ T4 ;
        P8 = P9 ^ T3 ;
        P9 = P10 ^ T2 ;
        P10 = P11 ^ T1 ;
        P11 = P12 ^ T0 ;
        P12 = OOV ;
    }

    if ( decoder == 0 )
    {
        // Now store computed check symbols back to memory
        loc = CWP[k+0] + i;
        *loc = P0 ;
        loc = CWP[k+1] + i;
        *loc = P1 ;
        loc = CWP[k+2] + i;
        *loc = P2 ;
        loc = CWP[k+3] + i;
        *loc = P3 ;
        loc = CWP[k+4] + i;
        *loc = P4 ;
        loc = CWP[k+5] + i;
        *loc = P5 ;
        loc = CWP[k+6] + i;
        *loc = P6 ;
        loc = CWP[k+7] + i;
        *loc = P7 ;
        loc = CWP[k+8] + i;
        *loc = P8 ;
        loc = CWP[k+9] + i;
        *loc = P9 ;
        loc = CWP[k+10] + i;
        *loc = P10 ;
        loc = CWP[k+11] + i;
        *loc = P11 ;
        loc = CWP[k+12] + i;
        *loc = P12 ;
    }
    else
    {
        // Now check for parity equal to zero
        T0 = P0;
        T0 |= P1;
        T0 |= P2;
        T0 |= P3;
        T0 |= P4;
        T0 |= P5;
        T0 |= P6;
        T0 |= P7;
        T0 |= P8;
        T0 |= P9;
        T0 |= P10;
        T0 |= P11;
        T0 |= P12;
        if (T0 != 0)
        {
            unsigned char Sr[13], S[13], errLocs[13], errMags[13] ;
            unsigned char Lambda[13+1], B[13+1], T[13+1], Omega[13+1] ;
            
            if ( T0 & (0xff << 0) )
            {
                atomicAdd(&error_count, 1);
                Sr[0] = (P0>>0) & 0xff ;
                Sr[1] = (P1>>0) & 0xff ;
                Sr[2] = (P2>>0) & 0xff ;
                Sr[3] = (P3>>0) & 0xff ;
                Sr[4] = (P4>>0) & 0xff ;
                Sr[5] = (P5>>0) & 0xff ;
                Sr[6] = (P6>>0) & 0xff ;
                Sr[7] = (P7>>0) & 0xff ;
                Sr[8] = (P8>>0) & 0xff ;
                Sr[9] = (P9>>0) & 0xff ;
                Sr[10] = (P10>>0) & 0xff ;
                Sr[11] = (P11>>0) & 0xff ;
                Sr[12] = (P12>>0) & 0xff ;
                rs_decode_palindromic_CUDA(Sr, S, 13, k, errLocs, errMags,
                                           0+(i<<2), Lambda, B, T, Omega) ;
            }
            if ( T0 & (0xff << 8) )
            {
                atomicAdd(&error_count, 1);
                Sr[0] = (P0>>8) & 0xff ;
                Sr[1] = (P1>>8) & 0xff ;
                Sr[2] = (P2>>8) & 0xff ;
                Sr[3] = (P3>>8) & 0xff ;
                Sr[4] = (P4>>8) & 0xff ;
                Sr[5] = (P5>>8) & 0xff ;
                Sr[6] = (P6>>8) & 0xff ;
                Sr[7] = (P7>>8) & 0xff ;
                Sr[8] = (P8>>8) & 0xff ;
                Sr[9] = (P9>>8) & 0xff ;
                Sr[10] = (P10>>8) & 0xff ;
                Sr[11] = (P11>>8) & 0xff ;
                Sr[12] = (P12>>8) & 0xff ;
                rs_decode_palindromic_CUDA(Sr, S, 13, k, errLocs, errMags,
                                           1+(i<<2), Lambda, B, T, Omega) ;
            }
            if ( T0 & (0xff << 16) )
            {
                atomicAdd(&error_count, 1);
                Sr[0] = (P0>>16) & 0xff ;
                Sr[1] = (P1>>16) & 0xff ;
                Sr[2] = (P2>>16) & 0xff ;
                Sr[3] = (P3>>16) & 0xff ;
                Sr[4] = (P4>>16) & 0xff ;
                Sr[5] = (P5>>16) & 0xff ;
                Sr[6] = (P6>>16) & 0xff ;
                Sr[7] = (P7>>16) & 0xff ;
                Sr[8] = (P8>>16) & 0xff ;
                Sr[9] = (P9>>16) & 0xff ;
                Sr[10] = (P10>>16) & 0xff ;
                Sr[11] = (P11>>16) & 0xff ;
                Sr[12] = (P12>>16) & 0xff ;
                rs_decode_palindromic_CUDA(Sr, S, 13, k, errLocs, errMags,
                                           2+(i<<2), Lambda, B, T, Omega) ;
            }
            if ( T0 & (0xff << 24) )
            {
                atomicAdd(&error_count, 1);
                Sr[0] = (P0>>24) & 0xff ;
                Sr[1] = (P1>>24) & 0xff ;
                Sr[2] = (P2>>24) & 0xff ;
                Sr[3] = (P3>>24) & 0xff ;
                Sr[4] = (P4>>24) & 0xff ;
                Sr[5] = (P5>>24) & 0xff ;
                Sr[6] = (P6>>24) & 0xff ;
                Sr[7] = (P7>>24) & 0xff ;
                Sr[8] = (P8>>24) & 0xff ;
                Sr[9] = (P9>>24) & 0xff ;
                Sr[10] = (P10>>24) & 0xff ;
                Sr[11] = (P11>>24) & 0xff ;
                Sr[12] = (P12>>24) & 0xff ;
                rs_decode_palindromic_CUDA(Sr, S, 13, k, errLocs, errMags,
                                           3+(i<<2), Lambda, B, T, Omega) ;
            }
        }
    }
}

// ----------------------------------------------
// Parallel LFSR Sequencer for P = 14 Codewords
// ----------------------------------------------
__global__ void ParallelLFSRSequencer14_CUDA(int k, int decoder)
{
    // Find out place in the grid of threads
    int i = blockIdx.x * blockDim.x + threadIdx.x;

    // Pointer to develop original data address
    unsigned int* loc ;

    // Create the parity values, initialize to zero
    unsigned int P0 = 0 ;
    unsigned int P1 = 0 ;
    unsigned int P2 = 0 ;
    unsigned int P3 = 0 ;
    unsigned int P4 = 0 ;
    unsigned int P5 = 0 ;
    unsigned int P6 = 0 ;
    unsigned int P7 = 0 ;
    unsigned int P8 = 0 ;
    unsigned int P9 = 0 ;
    unsigned int P10 = 0 ;
    unsigned int P11 = 0 ;
    unsigned int P12 = 0 ;
    unsigned int P13 = 0 ;

    // Temporary values for self reciprocol polynomial
    unsigned int T0;
    unsigned int T1;
    unsigned int T2;
    unsigned int T3;
    unsigned int T4;
    unsigned int T5;
    unsigned int T6;

    // Overflow value for LFSR and carry mask
    unsigned int OV, MSK, OF, OOV ;

    // Adjust K if we are decoding
    k += decoder ? 14 : 0 ;

    // Loop through each symbol
    for ( int sym = 0 ; sym < k ; sym ++ )
    {
        loc = CWP[sym] + i;
        OV = *loc ^ P0 ;
        OOV = OV ;

        // Process bit 0
        T1  = OV;                      // Account for bit 0 of constant cb
        T2  = OV;                      // Account for bit 0 of constant 1d
        T6  = OV;                      // Account for bit 0 of constant c1
        MSK = OV & 0x80808080 ;        // Create a mask of the MSBs
        OF = (MSK >> 7) * 0x1d ;       // Account for Overflow
        OV <<= 1 ;                     // Multiply the original data by 2
        OV &= 0xfefefefe ;             // Clear the MSBs
        OV ^= OF ;                     // Add in the overflow mask

        // Process bit 1
        T0  = OV;                      // Account for bit 1 of constant be
        T1 ^= OV;                      // Account for bit 1 of constant cb
        T3  = OV;                      // Account for bit 1 of constant ea
        T5  = OV;                      // Account for bit 1 of constant d6
        MSK = OV & 0x80808080 ;        // Create a mask of the MSBs
        OF = (MSK >> 7) * 0x1d ;       // Account for Overflow
        OV <<= 1 ;                     // Multiply the original data by 2
        OV &= 0xfefefefe ;             // Clear the MSBs
        OV ^= OF ;                     // Add in the overflow mask

        // Process bit 2
        T0 ^= OV;                      // Account for bit 2 of constant be
        T2 ^= OV;                      // Account for bit 2 of constant 1d
        T5 ^= OV;                      // Account for bit 2 of constant d6
        MSK = OV & 0x80808080 ;        // Create a mask of the MSBs
        OF = (MSK >> 7) * 0x1d ;       // Account for Overflow
        OV <<= 1 ;                     // Multiply the original data by 2
        OV &= 0xfefefefe ;             // Clear the MSBs
        OV ^= OF ;                     // Add in the overflow mask

        // Process bit 3
        T0 ^= OV;                      // Account for bit 3 of constant be
        T1 ^= OV;                      // Account for bit 3 of constant cb
        T2 ^= OV;                      // Account for bit 3 of constant 1d
        T3 ^= OV;                      // Account for bit 3 of constant ea
        MSK = OV & 0x80808080 ;        // Create a mask of the MSBs
        OF = (MSK >> 7) * 0x1d ;       // Account for Overflow
        OV <<= 1 ;                     // Multiply the original data by 2
        OV &= 0xfefefefe ;             // Clear the MSBs
        OV ^= OF ;                     // Add in the overflow mask

        // Process bit 4
        T0 ^= OV;                      // Account for bit 4 of constant be
        T2 ^= OV;                      // Account for bit 4 of constant 1d
        T5 ^= OV;                      // Account for bit 4 of constant d6
        MSK = OV & 0x80808080 ;        // Create a mask of the MSBs
        OF = (MSK >> 7) * 0x1d ;       // Account for Overflow
        OV <<= 1 ;                     // Multiply the original data by 2
        OV &= 0xfefefefe ;             // Clear the MSBs
        OV ^= OF ;                     // Add in the overflow mask

        // Process bit 5
        T0 ^= OV;                      // Account for bit 5 of constant be
        T3 ^= OV;                      // Account for bit 5 of constant ea
        T4  = OV;                      // Account for bit 5 of constant 60
        MSK = OV & 0x80808080 ;        // Create a mask of the MSBs
        OF = (MSK >> 7) * 0x1d ;       // Account for Overflow
        OV <<= 1 ;                     // Multiply the original data by 2
        OV &= 0xfefefefe ;             // Clear the MSBs
        OV ^= OF ;                     // Add in the overflow mask

        // Process bit 6
        T1 ^= OV;                      // Account for bit 6 of constant cb
        T3 ^= OV;                      // Account for bit 6 of constant ea
        T4 ^= OV;                      // Account for bit 6 of constant 60
        T5 ^= OV;                      // Account for bit 6 of constant d6
        T6 ^= OV;                      // Account for bit 6 of constant c1
        MSK = OV & 0x80808080 ;        // Create a mask of the MSBs
        OF = (MSK >> 7) * 0x1d ;       // Account for Overflow
        OV <<= 1 ;                     // Multiply the original data by 2
        OV &= 0xfefefefe ;             // Clear the MSBs
        OV ^= OF ;                     // Add in the overflow mask

        // Process bit 7
        T0 ^= OV;                      // Account for bit 7 of constant be
        T1 ^= OV;                      // Account for bit 7 of constant cb
        T3 ^= OV;                      // Account for bit 7 of constant ea
        T5 ^= OV;                      // Account for bit 7 of constant d6
        T6 ^= OV;                      // Account for bit 7 of constant c1

        // Now process the LFSR
        P0 = P1 ^ T0 ;
        P1 = P2 ^ T1 ;
        P2 = P3 ^ T2 ;
        P3 = P4 ^ T3 ;
        P4 = P5 ^ T4 ;
        P5 = P6 ^ T5 ;
        P6 = P7 ^ T6 ;
        P7 = P8 ^ T5 ;
        P8 = P9 ^ T4 ;
        P9 = P10 ^ T3 ;
        P10 = P11 ^ T2 ;
        P11 = P12 ^ T1 ;
        P12 = P13 ^ T0 ;
        P13 = OOV ;
    }

    if ( decoder == 0 )
    {
        // Now store computed check symbols back to memory
        loc = CWP[k+0] + i;
        *loc = P0 ;
        loc = CWP[k+1] + i;
        *loc = P1 ;
        loc = CWP[k+2] + i;
        *loc = P2 ;
        loc = CWP[k+3] + i;
        *loc = P3 ;
        loc = CWP[k+4] + i;
        *loc = P4 ;
        loc = CWP[k+5] + i;
        *loc = P5 ;
        loc = CWP[k+6] + i;
        *loc = P6 ;
        loc = CWP[k+7] + i;
        *loc = P7 ;
        loc = CWP[k+8] + i;
        *loc = P8 ;
        loc = CWP[k+9] + i;
        *loc = P9 ;
        loc = CWP[k+10] + i;
        *loc = P10 ;
        loc = CWP[k+11] + i;
        *loc = P11 ;
        loc = CWP[k+12] + i;
        *loc = P12 ;
        loc = CWP[k+13] + i;
        *loc = P13 ;
    }
    else
    {
        // Now check for parity equal to zero
        T0 = P0;
        T0 |= P1;
        T0 |= P2;
        T0 |= P3;
        T0 |= P4;
        T0 |= P5;
        T0 |= P6;
        T0 |= P7;
        T0 |= P8;
        T0 |= P9;
        T0 |= P10;
        T0 |= P11;
        T0 |= P12;
        T0 |= P13;
        if (T0 != 0)
        {
            unsigned char Sr[14], S[14], errLocs[14], errMags[14] ;
            unsigned char Lambda[14+1], B[14+1], T[14+1], Omega[14+1] ;
            
            if ( T0 & (0xff << 0) )
            {
                atomicAdd(&error_count, 1);
                Sr[0] = (P0>>0) & 0xff ;
                Sr[1] = (P1>>0) & 0xff ;
                Sr[2] = (P2>>0) & 0xff ;
                Sr[3] = (P3>>0) & 0xff ;
                Sr[4] = (P4>>0) & 0xff ;
                Sr[5] = (P5>>0) & 0xff ;
                Sr[6] = (P6>>0) & 0xff ;
                Sr[7] = (P7>>0) & 0xff ;
                Sr[8] = (P8>>0) & 0xff ;
                Sr[9] = (P9>>0) & 0xff ;
                Sr[10] = (P10>>0) & 0xff ;
                Sr[11] = (P11>>0) & 0xff ;
                Sr[12] = (P12>>0) & 0xff ;
                Sr[13] = (P13>>0) & 0xff ;
                rs_decode_palindromic_CUDA(Sr, S, 14, k, errLocs, errMags,
                                           0+(i<<2), Lambda, B, T, Omega) ;
            }
            if ( T0 & (0xff << 8) )
            {
                atomicAdd(&error_count, 1);
                Sr[0] = (P0>>8) & 0xff ;
                Sr[1] = (P1>>8) & 0xff ;
                Sr[2] = (P2>>8) & 0xff ;
                Sr[3] = (P3>>8) & 0xff ;
                Sr[4] = (P4>>8) & 0xff ;
                Sr[5] = (P5>>8) & 0xff ;
                Sr[6] = (P6>>8) & 0xff ;
                Sr[7] = (P7>>8) & 0xff ;
                Sr[8] = (P8>>8) & 0xff ;
                Sr[9] = (P9>>8) & 0xff ;
                Sr[10] = (P10>>8) & 0xff ;
                Sr[11] = (P11>>8) & 0xff ;
                Sr[12] = (P12>>8) & 0xff ;
                Sr[13] = (P13>>8) & 0xff ;
                rs_decode_palindromic_CUDA(Sr, S, 14, k, errLocs, errMags,
                                           1+(i<<2), Lambda, B, T, Omega) ;
            }
            if ( T0 & (0xff << 16) )
            {
                atomicAdd(&error_count, 1);
                Sr[0] = (P0>>16) & 0xff ;
                Sr[1] = (P1>>16) & 0xff ;
                Sr[2] = (P2>>16) & 0xff ;
                Sr[3] = (P3>>16) & 0xff ;
                Sr[4] = (P4>>16) & 0xff ;
                Sr[5] = (P5>>16) & 0xff ;
                Sr[6] = (P6>>16) & 0xff ;
                Sr[7] = (P7>>16) & 0xff ;
                Sr[8] = (P8>>16) & 0xff ;
                Sr[9] = (P9>>16) & 0xff ;
                Sr[10] = (P10>>16) & 0xff ;
                Sr[11] = (P11>>16) & 0xff ;
                Sr[12] = (P12>>16) & 0xff ;
                Sr[13] = (P13>>16) & 0xff ;
                rs_decode_palindromic_CUDA(Sr, S, 14, k, errLocs, errMags,
                                           2+(i<<2), Lambda, B, T, Omega) ;
            }
            if ( T0 & (0xff << 24) )
            {
                atomicAdd(&error_count, 1);
                Sr[0] = (P0>>24) & 0xff ;
                Sr[1] = (P1>>24) & 0xff ;
                Sr[2] = (P2>>24) & 0xff ;
                Sr[3] = (P3>>24) & 0xff ;
                Sr[4] = (P4>>24) & 0xff ;
                Sr[5] = (P5>>24) & 0xff ;
                Sr[6] = (P6>>24) & 0xff ;
                Sr[7] = (P7>>24) & 0xff ;
                Sr[8] = (P8>>24) & 0xff ;
                Sr[9] = (P9>>24) & 0xff ;
                Sr[10] = (P10>>24) & 0xff ;
                Sr[11] = (P11>>24) & 0xff ;
                Sr[12] = (P12>>24) & 0xff ;
                Sr[13] = (P13>>24) & 0xff ;
                rs_decode_palindromic_CUDA(Sr, S, 14, k, errLocs, errMags,
                                           3+(i<<2), Lambda, B, T, Omega) ;
            }
        }
    }
}

// ----------------------------------------------
// Parallel LFSR Sequencer for P = 15 Codewords
// ----------------------------------------------
__global__ void ParallelLFSRSequencer15_CUDA(int k, int decoder)
{
    // Find out place in the grid of threads
    int i = blockIdx.x * blockDim.x + threadIdx.x;

    // Pointer to develop original data address
    unsigned int* loc ;

    // Create the parity values, initialize to zero
    unsigned int P0 = 0 ;
    unsigned int P1 = 0 ;
    unsigned int P2 = 0 ;
    unsigned int P3 = 0 ;
    unsigned int P4 = 0 ;
    unsigned int P5 = 0 ;
    unsigned int P6 = 0 ;
    unsigned int P7 = 0 ;
    unsigned int P8 = 0 ;
    unsigned int P9 = 0 ;
    unsigned int P10 = 0 ;
    unsigned int P11 = 0 ;
    unsigned int P12 = 0 ;
    unsigned int P13 = 0 ;
    unsigned int P14 = 0 ;

    // Temporary values for self reciprocol polynomial
    unsigned int T0;
    unsigned int T1;
    unsigned int T2;
    unsigned int T3;
    unsigned int T4;
    unsigned int T5;
    unsigned int T6;

    // Overflow value for LFSR and carry mask
    unsigned int OV, MSK, OF, OOV ;

    // Adjust K if we are decoding
    k += decoder ? 15 : 0 ;

    // Loop through each symbol
    for ( int sym = 0 ; sym < k ; sym ++ )
    {
        loc = CWP[sym] + i;
        OV = *loc ^ P0 ;
        OOV = OV ;

        // Process bit 0
        T1  = OV;                      // Account for bit 0 of constant e5
        T3  = OV;                      // Account for bit 0 of constant a1
        MSK = OV & 0x80808080 ;        // Create a mask of the MSBs
        OF = (MSK >> 7) * 0x1d ;       // Account for Overflow
        OV <<= 1 ;                     // Multiply the original data by 2
        OV &= 0xfefefefe ;             // Clear the MSBs
        OV ^= OF ;                     // Add in the overflow mask

        // Process bit 1
        T0  = OV;                      // Account for bit 1 of constant 2
        T2  = OV;                      // Account for bit 1 of constant 6a
        T4  = OV;                      // Account for bit 1 of constant 7e
        MSK = OV & 0x80808080 ;        // Create a mask of the MSBs
        OF = (MSK >> 7) * 0x1d ;       // Account for Overflow
        OV <<= 1 ;                     // Multiply the original data by 2
        OV &= 0xfefefefe ;             // Clear the MSBs
        OV ^= OF ;                     // Add in the overflow mask

        // Process bit 2
        T1 ^= OV;                      // Account for bit 2 of constant e5
        T4 ^= OV;                      // Account for bit 2 of constant 7e
        T5  = OV;                      // Account for bit 2 of constant 6c
        T6  = OV;                      // Account for bit 2 of constant 4
        MSK = OV & 0x80808080 ;        // Create a mask of the MSBs
        OF = (MSK >> 7) * 0x1d ;       // Account for Overflow
        OV <<= 1 ;                     // Multiply the original data by 2
        OV &= 0xfefefefe ;             // Clear the MSBs
        OV ^= OF ;                     // Add in the overflow mask

        // Process bit 3
        T2 ^= OV;                      // Account for bit 3 of constant 6a
        T4 ^= OV;                      // Account for bit 3 of constant 7e
        T5 ^= OV;                      // Account for bit 3 of constant 6c
        MSK = OV & 0x80808080 ;        // Create a mask of the MSBs
        OF = (MSK >> 7) * 0x1d ;       // Account for Overflow
        OV <<= 1 ;                     // Multiply the original data by 2
        OV &= 0xfefefefe ;             // Clear the MSBs
        OV ^= OF ;                     // Add in the overflow mask

        // Process bit 4
        T4 ^= OV;                      // Account for bit 4 of constant 7e
        MSK = OV & 0x80808080 ;        // Create a mask of the MSBs
        OF = (MSK >> 7) * 0x1d ;       // Account for Overflow
        OV <<= 1 ;                     // Multiply the original data by 2
        OV &= 0xfefefefe ;             // Clear the MSBs
        OV ^= OF ;                     // Add in the overflow mask

        // Process bit 5
        T1 ^= OV;                      // Account for bit 5 of constant e5
        T2 ^= OV;                      // Account for bit 5 of constant 6a
        T3 ^= OV;                      // Account for bit 5 of constant a1
        T4 ^= OV;                      // Account for bit 5 of constant 7e
        T5 ^= OV;                      // Account for bit 5 of constant 6c
        MSK = OV & 0x80808080 ;        // Create a mask of the MSBs
        OF = (MSK >> 7) * 0x1d ;       // Account for Overflow
        OV <<= 1 ;                     // Multiply the original data by 2
        OV &= 0xfefefefe ;             // Clear the MSBs
        OV ^= OF ;                     // Add in the overflow mask

        // Process bit 6
        T1 ^= OV;                      // Account for bit 6 of constant e5
        T2 ^= OV;                      // Account for bit 6 of constant 6a
        T4 ^= OV;                      // Account for bit 6 of constant 7e
        T5 ^= OV;                      // Account for bit 6 of constant 6c
        MSK = OV & 0x80808080 ;        // Create a mask of the MSBs
        OF = (MSK >> 7) * 0x1d ;       // Account for Overflow
        OV <<= 1 ;                     // Multiply the original data by 2
        OV &= 0xfefefefe ;             // Clear the MSBs
        OV ^= OF ;                     // Add in the overflow mask

        // Process bit 7
        T1 ^= OV;                      // Account for bit 7 of constant e5
        T3 ^= OV;                      // Account for bit 7 of constant a1

        // Now process the LFSR
        P0 = P1 ^ T0 ;
        P1 = P2 ^ T1 ;
        P2 = P3 ^ T2 ;
        P3 = P4 ^ T3 ;
        P4 = P5 ^ T4 ;
        P5 = P6 ^ T5 ;
        P6 = P7 ^ T6 ;
        P7 = P8 ^ T6 ;
        P8 = P9 ^ T5 ;
        P9 = P10 ^ T4 ;
        P10 = P11 ^ T3 ;
        P11 = P12 ^ T2 ;
        P12 = P13 ^ T1 ;
        P13 = P14 ^ T0 ;
        P14 = OOV ;
    }

    if ( decoder == 0 )
    {
        // Now store computed check symbols back to memory
        loc = CWP[k+0] + i;
        *loc = P0 ;
        loc = CWP[k+1] + i;
        *loc = P1 ;
        loc = CWP[k+2] + i;
        *loc = P2 ;
        loc = CWP[k+3] + i;
        *loc = P3 ;
        loc = CWP[k+4] + i;
        *loc = P4 ;
        loc = CWP[k+5] + i;
        *loc = P5 ;
        loc = CWP[k+6] + i;
        *loc = P6 ;
        loc = CWP[k+7] + i;
        *loc = P7 ;
        loc = CWP[k+8] + i;
        *loc = P8 ;
        loc = CWP[k+9] + i;
        *loc = P9 ;
        loc = CWP[k+10] + i;
        *loc = P10 ;
        loc = CWP[k+11] + i;
        *loc = P11 ;
        loc = CWP[k+12] + i;
        *loc = P12 ;
        loc = CWP[k+13] + i;
        *loc = P13 ;
        loc = CWP[k+14] + i;
        *loc = P14 ;
    }
    else
    {
        // Now check for parity equal to zero
        T0 = P0;
        T0 |= P1;
        T0 |= P2;
        T0 |= P3;
        T0 |= P4;
        T0 |= P5;
        T0 |= P6;
        T0 |= P7;
        T0 |= P8;
        T0 |= P9;
        T0 |= P10;
        T0 |= P11;
        T0 |= P12;
        T0 |= P13;
        T0 |= P14;
        if (T0 != 0)
        {
            unsigned char Sr[15], S[15], errLocs[15], errMags[15] ;
            unsigned char Lambda[15+1], B[15+1], T[15+1], Omega[15+1] ;
            
            if ( T0 & (0xff << 0) )
            {
                atomicAdd(&error_count, 1);
                Sr[0] = (P0>>0) & 0xff ;
                Sr[1] = (P1>>0) & 0xff ;
                Sr[2] = (P2>>0) & 0xff ;
                Sr[3] = (P3>>0) & 0xff ;
                Sr[4] = (P4>>0) & 0xff ;
                Sr[5] = (P5>>0) & 0xff ;
                Sr[6] = (P6>>0) & 0xff ;
                Sr[7] = (P7>>0) & 0xff ;
                Sr[8] = (P8>>0) & 0xff ;
                Sr[9] = (P9>>0) & 0xff ;
                Sr[10] = (P10>>0) & 0xff ;
                Sr[11] = (P11>>0) & 0xff ;
                Sr[12] = (P12>>0) & 0xff ;
                Sr[13] = (P13>>0) & 0xff ;
                Sr[14] = (P14>>0) & 0xff ;
                rs_decode_palindromic_CUDA(Sr, S, 15, k, errLocs, errMags,
                                           0+(i<<2), Lambda, B, T, Omega) ;
            }
            if ( T0 & (0xff << 8) )
            {
                atomicAdd(&error_count, 1);
                Sr[0] = (P0>>8) & 0xff ;
                Sr[1] = (P1>>8) & 0xff ;
                Sr[2] = (P2>>8) & 0xff ;
                Sr[3] = (P3>>8) & 0xff ;
                Sr[4] = (P4>>8) & 0xff ;
                Sr[5] = (P5>>8) & 0xff ;
                Sr[6] = (P6>>8) & 0xff ;
                Sr[7] = (P7>>8) & 0xff ;
                Sr[8] = (P8>>8) & 0xff ;
                Sr[9] = (P9>>8) & 0xff ;
                Sr[10] = (P10>>8) & 0xff ;
                Sr[11] = (P11>>8) & 0xff ;
                Sr[12] = (P12>>8) & 0xff ;
                Sr[13] = (P13>>8) & 0xff ;
                Sr[14] = (P14>>8) & 0xff ;
                rs_decode_palindromic_CUDA(Sr, S, 15, k, errLocs, errMags,
                                           1+(i<<2), Lambda, B, T, Omega) ;
            }
            if ( T0 & (0xff << 16) )
            {
                atomicAdd(&error_count, 1);
                Sr[0] = (P0>>16) & 0xff ;
                Sr[1] = (P1>>16) & 0xff ;
                Sr[2] = (P2>>16) & 0xff ;
                Sr[3] = (P3>>16) & 0xff ;
                Sr[4] = (P4>>16) & 0xff ;
                Sr[5] = (P5>>16) & 0xff ;
                Sr[6] = (P6>>16) & 0xff ;
                Sr[7] = (P7>>16) & 0xff ;
                Sr[8] = (P8>>16) & 0xff ;
                Sr[9] = (P9>>16) & 0xff ;
                Sr[10] = (P10>>16) & 0xff ;
                Sr[11] = (P11>>16) & 0xff ;
                Sr[12] = (P12>>16) & 0xff ;
                Sr[13] = (P13>>16) & 0xff ;
                Sr[14] = (P14>>16) & 0xff ;
                rs_decode_palindromic_CUDA(Sr, S, 15, k, errLocs, errMags,
                                           2+(i<<2), Lambda, B, T, Omega) ;
            }
            if ( T0 & (0xff << 24) )
            {
                atomicAdd(&error_count, 1);
                Sr[0] = (P0>>24) & 0xff ;
                Sr[1] = (P1>>24) & 0xff ;
                Sr[2] = (P2>>24) & 0xff ;
                Sr[3] = (P3>>24) & 0xff ;
                Sr[4] = (P4>>24) & 0xff ;
                Sr[5] = (P5>>24) & 0xff ;
                Sr[6] = (P6>>24) & 0xff ;
                Sr[7] = (P7>>24) & 0xff ;
                Sr[8] = (P8>>24) & 0xff ;
                Sr[9] = (P9>>24) & 0xff ;
                Sr[10] = (P10>>24) & 0xff ;
                Sr[11] = (P11>>24) & 0xff ;
                Sr[12] = (P12>>24) & 0xff ;
                Sr[13] = (P13>>24) & 0xff ;
                Sr[14] = (P14>>24) & 0xff ;
                rs_decode_palindromic_CUDA(Sr, S, 15, k, errLocs, errMags,
                                           3+(i<<2), Lambda, B, T, Omega) ;
            }
        }
    }
}

// ----------------------------------------------
// Parallel LFSR Sequencer for P = 16 Codewords
// ----------------------------------------------
__global__ void ParallelLFSRSequencer16_CUDA(int k, int decoder)
{
    // Find out place in the grid of threads
    int i = blockIdx.x * blockDim.x + threadIdx.x;

    // Pointer to develop original data address
    unsigned int* loc ;

    // Create the parity values, initialize to zero
    unsigned int P0 = 0 ;
    unsigned int P1 = 0 ;
    unsigned int P2 = 0 ;
    unsigned int P3 = 0 ;
    unsigned int P4 = 0 ;
    unsigned int P5 = 0 ;
    unsigned int P6 = 0 ;
    unsigned int P7 = 0 ;
    unsigned int P8 = 0 ;
    unsigned int P9 = 0 ;
    unsigned int P10 = 0 ;
    unsigned int P11 = 0 ;
    unsigned int P12 = 0 ;
    unsigned int P13 = 0 ;
    unsigned int P14 = 0 ;
    unsigned int P15 = 0 ;

    // Temporary values for self reciprocol polynomial
    unsigned int T0;
    unsigned int T1;
    unsigned int T2;
    unsigned int T3;
    unsigned int T4;
    unsigned int T5;
    unsigned int T6;
    unsigned int T7;

    // Overflow value for LFSR and carry mask
    unsigned int OV, MSK, OF, OOV ;

    // Adjust K if we are decoding
    k += decoder ? 16 : 0 ;

    // Loop through each symbol
    for ( int sym = 0 ; sym < k ; sym ++ )
    {
        loc = CWP[sym] + i;
        OV = *loc ^ P0 ;
        OOV = OV ;

        // Process bit 0
        T1  = OV;                      // Account for bit 0 of constant e1
        T2  = OV;                      // Account for bit 0 of constant 79
        T7  = OV;                      // Account for bit 0 of constant 7
        MSK = OV & 0x80808080 ;        // Create a mask of the MSBs
        OF = (MSK >> 7) * 0x1d ;       // Account for Overflow
        OV <<= 1 ;                     // Multiply the original data by 2
        OV &= 0xfefefefe ;             // Clear the MSBs
        OV ^= OF ;                     // Add in the overflow mask

        // Process bit 1
        T4  = OV;                      // Account for bit 1 of constant 82
        T6  = OV;                      // Account for bit 1 of constant aa
        T7 ^= OV;                      // Account for bit 1 of constant 7
        MSK = OV & 0x80808080 ;        // Create a mask of the MSBs
        OF = (MSK >> 7) * 0x1d ;       // Account for Overflow
        OV <<= 1 ;                     // Multiply the original data by 2
        OV &= 0xfefefefe ;             // Clear the MSBs
        OV ^= OF ;                     // Add in the overflow mask

        // Process bit 2
        T0  = OV;                      // Account for bit 2 of constant 2c
        T7 ^= OV;                      // Account for bit 2 of constant 7
        MSK = OV & 0x80808080 ;        // Create a mask of the MSBs
        OF = (MSK >> 7) * 0x1d ;       // Account for Overflow
        OV <<= 1 ;                     // Multiply the original data by 2
        OV &= 0xfefefefe ;             // Clear the MSBs
        OV ^= OF ;                     // Add in the overflow mask

        // Process bit 3
        T0 ^= OV;                      // Account for bit 3 of constant 2c
        T2 ^= OV;                      // Account for bit 3 of constant 79
        T5  = OV;                      // Account for bit 3 of constant f8
        T6 ^= OV;                      // Account for bit 3 of constant aa
        MSK = OV & 0x80808080 ;        // Create a mask of the MSBs
        OF = (MSK >> 7) * 0x1d ;       // Account for Overflow
        OV <<= 1 ;                     // Multiply the original data by 2
        OV &= 0xfefefefe ;             // Clear the MSBs
        OV ^= OF ;                     // Add in the overflow mask

        // Process bit 4
        T2 ^= OV;                      // Account for bit 4 of constant 79
        T3  = OV;                      // Account for bit 4 of constant f0
        T5 ^= OV;                      // Account for bit 4 of constant f8
        MSK = OV & 0x80808080 ;        // Create a mask of the MSBs
        OF = (MSK >> 7) * 0x1d ;       // Account for Overflow
        OV <<= 1 ;                     // Multiply the original data by 2
        OV &= 0xfefefefe ;             // Clear the MSBs
        OV ^= OF ;                     // Add in the overflow mask

        // Process bit 5
        T0 ^= OV;                      // Account for bit 5 of constant 2c
        T1 ^= OV;                      // Account for bit 5 of constant e1
        T2 ^= OV;                      // Account for bit 5 of constant 79
        T3 ^= OV;                      // Account for bit 5 of constant f0
        T5 ^= OV;                      // Account for bit 5 of constant f8
        T6 ^= OV;                      // Account for bit 5 of constant aa
        MSK = OV & 0x80808080 ;        // Create a mask of the MSBs
        OF = (MSK >> 7) * 0x1d ;       // Account for Overflow
        OV <<= 1 ;                     // Multiply the original data by 2
        OV &= 0xfefefefe ;             // Clear the MSBs
        OV ^= OF ;                     // Add in the overflow mask

        // Process bit 6
        T1 ^= OV;                      // Account for bit 6 of constant e1
        T2 ^= OV;                      // Account for bit 6 of constant 79
        T3 ^= OV;                      // Account for bit 6 of constant f0
        T5 ^= OV;                      // Account for bit 6 of constant f8
        MSK = OV & 0x80808080 ;        // Create a mask of the MSBs
        OF = (MSK >> 7) * 0x1d ;       // Account for Overflow
        OV <<= 1 ;                     // Multiply the original data by 2
        OV &= 0xfefefefe ;             // Clear the MSBs
        OV ^= OF ;                     // Add in the overflow mask

        // Process bit 7
        T1 ^= OV;                      // Account for bit 7 of constant e1
        T3 ^= OV;                      // Account for bit 7 of constant f0
        T4 ^= OV;                      // Account for bit 7 of constant 82
        T5 ^= OV;                      // Account for bit 7 of constant f8
        T6 ^= OV;                      // Account for bit 7 of constant aa

        // Now process the LFSR
        P0 = P1 ^ T0 ;
        P1 = P2 ^ T1 ;
        P2 = P3 ^ T2 ;
        P3 = P4 ^ T3 ;
        P4 = P5 ^ T4 ;
        P5 = P6 ^ T5 ;
        P6 = P7 ^ T6 ;
        P7 = P8 ^ T7 ;
        P8 = P9 ^ T6 ;
        P9 = P10 ^ T5 ;
        P10 = P11 ^ T4 ;
        P11 = P12 ^ T3 ;
        P12 = P13 ^ T2 ;
        P13 = P14 ^ T1 ;
        P14 = P15 ^ T0 ;
        P15 = OOV ;
    }

    if ( decoder == 0 )
    {
        // Now store computed check symbols back to memory
        loc = CWP[k+0] + i;
        *loc = P0 ;
        loc = CWP[k+1] + i;
        *loc = P1 ;
        loc = CWP[k+2] + i;
        *loc = P2 ;
        loc = CWP[k+3] + i;
        *loc = P3 ;
        loc = CWP[k+4] + i;
        *loc = P4 ;
        loc = CWP[k+5] + i;
        *loc = P5 ;
        loc = CWP[k+6] + i;
        *loc = P6 ;
        loc = CWP[k+7] + i;
        *loc = P7 ;
        loc = CWP[k+8] + i;
        *loc = P8 ;
        loc = CWP[k+9] + i;
        *loc = P9 ;
        loc = CWP[k+10] + i;
        *loc = P10 ;
        loc = CWP[k+11] + i;
        *loc = P11 ;
        loc = CWP[k+12] + i;
        *loc = P12 ;
        loc = CWP[k+13] + i;
        *loc = P13 ;
        loc = CWP[k+14] + i;
        *loc = P14 ;
        loc = CWP[k+15] + i;
        *loc = P15 ;
    }
    else
    {
        // Now check for parity equal to zero
        T0 = P0;
        T0 |= P1;
        T0 |= P2;
        T0 |= P3;
        T0 |= P4;
        T0 |= P5;
        T0 |= P6;
        T0 |= P7;
        T0 |= P8;
        T0 |= P9;
        T0 |= P10;
        T0 |= P11;
        T0 |= P12;
        T0 |= P13;
        T0 |= P14;
        T0 |= P15;
        if (T0 != 0)
        {
            unsigned char Sr[16], S[16], errLocs[16], errMags[16] ;
            unsigned char Lambda[16+1], B[16+1], T[16+1], Omega[16+1] ;
            
            if ( T0 & (0xff << 0) )
            {
                atomicAdd(&error_count, 1);
                Sr[0] = (P0>>0) & 0xff ;
                Sr[1] = (P1>>0) & 0xff ;
                Sr[2] = (P2>>0) & 0xff ;
                Sr[3] = (P3>>0) & 0xff ;
                Sr[4] = (P4>>0) & 0xff ;
                Sr[5] = (P5>>0) & 0xff ;
                Sr[6] = (P6>>0) & 0xff ;
                Sr[7] = (P7>>0) & 0xff ;
                Sr[8] = (P8>>0) & 0xff ;
                Sr[9] = (P9>>0) & 0xff ;
                Sr[10] = (P10>>0) & 0xff ;
                Sr[11] = (P11>>0) & 0xff ;
                Sr[12] = (P12>>0) & 0xff ;
                Sr[13] = (P13>>0) & 0xff ;
                Sr[14] = (P14>>0) & 0xff ;
                Sr[15] = (P15>>0) & 0xff ;
                rs_decode_palindromic_CUDA(Sr, S, 16, k, errLocs, errMags,
                                           0+(i<<2), Lambda, B, T, Omega) ;
            }
            if ( T0 & (0xff << 8) )
            {
                atomicAdd(&error_count, 1);
                Sr[0] = (P0>>8) & 0xff ;
                Sr[1] = (P1>>8) & 0xff ;
                Sr[2] = (P2>>8) & 0xff ;
                Sr[3] = (P3>>8) & 0xff ;
                Sr[4] = (P4>>8) & 0xff ;
                Sr[5] = (P5>>8) & 0xff ;
                Sr[6] = (P6>>8) & 0xff ;
                Sr[7] = (P7>>8) & 0xff ;
                Sr[8] = (P8>>8) & 0xff ;
                Sr[9] = (P9>>8) & 0xff ;
                Sr[10] = (P10>>8) & 0xff ;
                Sr[11] = (P11>>8) & 0xff ;
                Sr[12] = (P12>>8) & 0xff ;
                Sr[13] = (P13>>8) & 0xff ;
                Sr[14] = (P14>>8) & 0xff ;
                Sr[15] = (P15>>8) & 0xff ;
                rs_decode_palindromic_CUDA(Sr, S, 16, k, errLocs, errMags,
                                           1+(i<<2), Lambda, B, T, Omega) ;
            }
            if ( T0 & (0xff << 16) )
            {
                atomicAdd(&error_count, 1);
                Sr[0] = (P0>>16) & 0xff ;
                Sr[1] = (P1>>16) & 0xff ;
                Sr[2] = (P2>>16) & 0xff ;
                Sr[3] = (P3>>16) & 0xff ;
                Sr[4] = (P4>>16) & 0xff ;
                Sr[5] = (P5>>16) & 0xff ;
                Sr[6] = (P6>>16) & 0xff ;
                Sr[7] = (P7>>16) & 0xff ;
                Sr[8] = (P8>>16) & 0xff ;
                Sr[9] = (P9>>16) & 0xff ;
                Sr[10] = (P10>>16) & 0xff ;
                Sr[11] = (P11>>16) & 0xff ;
                Sr[12] = (P12>>16) & 0xff ;
                Sr[13] = (P13>>16) & 0xff ;
                Sr[14] = (P14>>16) & 0xff ;
                Sr[15] = (P15>>16) & 0xff ;
                rs_decode_palindromic_CUDA(Sr, S, 16, k, errLocs, errMags,
                                           2+(i<<2), Lambda, B, T, Omega) ;
            }
            if ( T0 & (0xff << 24) )
            {
                atomicAdd(&error_count, 1);
                Sr[0] = (P0>>24) & 0xff ;
                Sr[1] = (P1>>24) & 0xff ;
                Sr[2] = (P2>>24) & 0xff ;
                Sr[3] = (P3>>24) & 0xff ;
                Sr[4] = (P4>>24) & 0xff ;
                Sr[5] = (P5>>24) & 0xff ;
                Sr[6] = (P6>>24) & 0xff ;
                Sr[7] = (P7>>24) & 0xff ;
                Sr[8] = (P8>>24) & 0xff ;
                Sr[9] = (P9>>24) & 0xff ;
                Sr[10] = (P10>>24) & 0xff ;
                Sr[11] = (P11>>24) & 0xff ;
                Sr[12] = (P12>>24) & 0xff ;
                Sr[13] = (P13>>24) & 0xff ;
                Sr[14] = (P14>>24) & 0xff ;
                Sr[15] = (P15>>24) & 0xff ;
                rs_decode_palindromic_CUDA(Sr, S, 16, k, errLocs, errMags,
                                           3+(i<<2), Lambda, B, T, Omega) ;
            }
        }
    }
}

// ----------------------------------------------
// Parallel LFSR Sequencer for P = 17 Codewords
// ----------------------------------------------
__global__ void ParallelLFSRSequencer17_CUDA(int k, int decoder)
{
    // Find out place in the grid of threads
    int i = blockIdx.x * blockDim.x + threadIdx.x;

    // Pointer to develop original data address
    unsigned int* loc ;

    // Create the parity values, initialize to zero
    unsigned int P0 = 0 ;
    unsigned int P1 = 0 ;
    unsigned int P2 = 0 ;
    unsigned int P3 = 0 ;
    unsigned int P4 = 0 ;
    unsigned int P5 = 0 ;
    unsigned int P6 = 0 ;
    unsigned int P7 = 0 ;
    unsigned int P8 = 0 ;
    unsigned int P9 = 0 ;
    unsigned int P10 = 0 ;
    unsigned int P11 = 0 ;
    unsigned int P12 = 0 ;
    unsigned int P13 = 0 ;
    unsigned int P14 = 0 ;
    unsigned int P15 = 0 ;
    unsigned int P16 = 0 ;

    // Temporary values for self reciprocol polynomial
    unsigned int T0;
    unsigned int T1;
    unsigned int T2;
    unsigned int T3;
    unsigned int T4;
    unsigned int T5;
    unsigned int T6;
    unsigned int T7;

    // Overflow value for LFSR and carry mask
    unsigned int OV, MSK, OF, OOV ;

    // Adjust K if we are decoding
    k += decoder ? 17 : 0 ;

    // Loop through each symbol
    for ( int sym = 0 ; sym < k ; sym ++ )
    {
        loc = CWP[sym] + i;
        OV = *loc ^ P0 ;
        OOV = OV ;

        // Process bit 0
        T1  = OV;                      // Account for bit 0 of constant c5
        T3  = OV;                      // Account for bit 0 of constant 9f
        T5  = OV;                      // Account for bit 0 of constant 41
        MSK = OV & 0x80808080 ;        // Create a mask of the MSBs
        OF = (MSK >> 7) * 0x1d ;       // Account for Overflow
        OV <<= 1 ;                     // Multiply the original data by 2
        OV &= 0xfefefefe ;             // Clear the MSBs
        OV ^= OF ;                     // Add in the overflow mask

        // Process bit 1
        T2  = OV;                      // Account for bit 1 of constant 62
        T3 ^= OV;                      // Account for bit 1 of constant 9f
        T6  = OV;                      // Account for bit 1 of constant c2
        T7  = OV;                      // Account for bit 1 of constant 2a
        MSK = OV & 0x80808080 ;        // Create a mask of the MSBs
        OF = (MSK >> 7) * 0x1d ;       // Account for Overflow
        OV <<= 1 ;                     // Multiply the original data by 2
        OV &= 0xfefefefe ;             // Clear the MSBs
        OV ^= OF ;                     // Add in the overflow mask

        // Process bit 2
        T0  = OV;                      // Account for bit 2 of constant 9c
        T1 ^= OV;                      // Account for bit 2 of constant c5
        T3 ^= OV;                      // Account for bit 2 of constant 9f
        MSK = OV & 0x80808080 ;        // Create a mask of the MSBs
        OF = (MSK >> 7) * 0x1d ;       // Account for Overflow
        OV <<= 1 ;                     // Multiply the original data by 2
        OV &= 0xfefefefe ;             // Clear the MSBs
        OV ^= OF ;                     // Add in the overflow mask

        // Process bit 3
        T0 ^= OV;                      // Account for bit 3 of constant 9c
        T3 ^= OV;                      // Account for bit 3 of constant 9f
        T4  = OV;                      // Account for bit 3 of constant 8
        T7 ^= OV;                      // Account for bit 3 of constant 2a
        MSK = OV & 0x80808080 ;        // Create a mask of the MSBs
        OF = (MSK >> 7) * 0x1d ;       // Account for Overflow
        OV <<= 1 ;                     // Multiply the original data by 2
        OV &= 0xfefefefe ;             // Clear the MSBs
        OV ^= OF ;                     // Add in the overflow mask

        // Process bit 4
        T0 ^= OV;                      // Account for bit 4 of constant 9c
        T3 ^= OV;                      // Account for bit 4 of constant 9f
        MSK = OV & 0x80808080 ;        // Create a mask of the MSBs
        OF = (MSK >> 7) * 0x1d ;       // Account for Overflow
        OV <<= 1 ;                     // Multiply the original data by 2
        OV &= 0xfefefefe ;             // Clear the MSBs
        OV ^= OF ;                     // Add in the overflow mask

        // Process bit 5
        T2 ^= OV;                      // Account for bit 5 of constant 62
        T7 ^= OV;                      // Account for bit 5 of constant 2a
        MSK = OV & 0x80808080 ;        // Create a mask of the MSBs
        OF = (MSK >> 7) * 0x1d ;       // Account for Overflow
        OV <<= 1 ;                     // Multiply the original data by 2
        OV &= 0xfefefefe ;             // Clear the MSBs
        OV ^= OF ;                     // Add in the overflow mask

        // Process bit 6
        T1 ^= OV;                      // Account for bit 6 of constant c5
        T2 ^= OV;                      // Account for bit 6 of constant 62
        T5 ^= OV;                      // Account for bit 6 of constant 41
        T6 ^= OV;                      // Account for bit 6 of constant c2
        MSK = OV & 0x80808080 ;        // Create a mask of the MSBs
        OF = (MSK >> 7) * 0x1d ;       // Account for Overflow
        OV <<= 1 ;                     // Multiply the original data by 2
        OV &= 0xfefefefe ;             // Clear the MSBs
        OV ^= OF ;                     // Add in the overflow mask

        // Process bit 7
        T0 ^= OV;                      // Account for bit 7 of constant 9c
        T1 ^= OV;                      // Account for bit 7 of constant c5
        T3 ^= OV;                      // Account for bit 7 of constant 9f
        T6 ^= OV;                      // Account for bit 7 of constant c2

        // Now process the LFSR
        P0 = P1 ^ T0 ;
        P1 = P2 ^ T1 ;
        P2 = P3 ^ T2 ;
        P3 = P4 ^ T3 ;
        P4 = P5 ^ T4 ;
        P5 = P6 ^ T5 ;
        P6 = P7 ^ T6 ;
        P7 = P8 ^ T7 ;
        P8 = P9 ^ T7 ;
        P9 = P10 ^ T6 ;
        P10 = P11 ^ T5 ;
        P11 = P12 ^ T4 ;
        P12 = P13 ^ T3 ;
        P13 = P14 ^ T2 ;
        P14 = P15 ^ T1 ;
        P15 = P16 ^ T0 ;
        P16 = OOV ;
    }

    if ( decoder == 0 )
    {
        // Now store computed check symbols back to memory
        loc = CWP[k+0] + i;
        *loc = P0 ;
        loc = CWP[k+1] + i;
        *loc = P1 ;
        loc = CWP[k+2] + i;
        *loc = P2 ;
        loc = CWP[k+3] + i;
        *loc = P3 ;
        loc = CWP[k+4] + i;
        *loc = P4 ;
        loc = CWP[k+5] + i;
        *loc = P5 ;
        loc = CWP[k+6] + i;
        *loc = P6 ;
        loc = CWP[k+7] + i;
        *loc = P7 ;
        loc = CWP[k+8] + i;
        *loc = P8 ;
        loc = CWP[k+9] + i;
        *loc = P9 ;
        loc = CWP[k+10] + i;
        *loc = P10 ;
        loc = CWP[k+11] + i;
        *loc = P11 ;
        loc = CWP[k+12] + i;
        *loc = P12 ;
        loc = CWP[k+13] + i;
        *loc = P13 ;
        loc = CWP[k+14] + i;
        *loc = P14 ;
        loc = CWP[k+15] + i;
        *loc = P15 ;
        loc = CWP[k+16] + i;
        *loc = P16 ;
    }
    else
    {
        // Now check for parity equal to zero
        T0 = P0;
        T0 |= P1;
        T0 |= P2;
        T0 |= P3;
        T0 |= P4;
        T0 |= P5;
        T0 |= P6;
        T0 |= P7;
        T0 |= P8;
        T0 |= P9;
        T0 |= P10;
        T0 |= P11;
        T0 |= P12;
        T0 |= P13;
        T0 |= P14;
        T0 |= P15;
        T0 |= P16;
        if (T0 != 0)
        {
            unsigned char Sr[17], S[17], errLocs[17], errMags[17] ;
            unsigned char Lambda[17+1], B[17+1], T[17+1], Omega[17+1] ;
            
            if ( T0 & (0xff << 0) )
            {
                atomicAdd(&error_count, 1);
                Sr[0] = (P0>>0) & 0xff ;
                Sr[1] = (P1>>0) & 0xff ;
                Sr[2] = (P2>>0) & 0xff ;
                Sr[3] = (P3>>0) & 0xff ;
                Sr[4] = (P4>>0) & 0xff ;
                Sr[5] = (P5>>0) & 0xff ;
                Sr[6] = (P6>>0) & 0xff ;
                Sr[7] = (P7>>0) & 0xff ;
                Sr[8] = (P8>>0) & 0xff ;
                Sr[9] = (P9>>0) & 0xff ;
                Sr[10] = (P10>>0) & 0xff ;
                Sr[11] = (P11>>0) & 0xff ;
                Sr[12] = (P12>>0) & 0xff ;
                Sr[13] = (P13>>0) & 0xff ;
                Sr[14] = (P14>>0) & 0xff ;
                Sr[15] = (P15>>0) & 0xff ;
                Sr[16] = (P16>>0) & 0xff ;
                rs_decode_palindromic_CUDA(Sr, S, 17, k, errLocs, errMags,
                                           0+(i<<2), Lambda, B, T, Omega) ;
            }
            if ( T0 & (0xff << 8) )
            {
                atomicAdd(&error_count, 1);
                Sr[0] = (P0>>8) & 0xff ;
                Sr[1] = (P1>>8) & 0xff ;
                Sr[2] = (P2>>8) & 0xff ;
                Sr[3] = (P3>>8) & 0xff ;
                Sr[4] = (P4>>8) & 0xff ;
                Sr[5] = (P5>>8) & 0xff ;
                Sr[6] = (P6>>8) & 0xff ;
                Sr[7] = (P7>>8) & 0xff ;
                Sr[8] = (P8>>8) & 0xff ;
                Sr[9] = (P9>>8) & 0xff ;
                Sr[10] = (P10>>8) & 0xff ;
                Sr[11] = (P11>>8) & 0xff ;
                Sr[12] = (P12>>8) & 0xff ;
                Sr[13] = (P13>>8) & 0xff ;
                Sr[14] = (P14>>8) & 0xff ;
                Sr[15] = (P15>>8) & 0xff ;
                Sr[16] = (P16>>8) & 0xff ;
                rs_decode_palindromic_CUDA(Sr, S, 17, k, errLocs, errMags,
                                           1+(i<<2), Lambda, B, T, Omega) ;
            }
            if ( T0 & (0xff << 16) )
            {
                atomicAdd(&error_count, 1);
                Sr[0] = (P0>>16) & 0xff ;
                Sr[1] = (P1>>16) & 0xff ;
                Sr[2] = (P2>>16) & 0xff ;
                Sr[3] = (P3>>16) & 0xff ;
                Sr[4] = (P4>>16) & 0xff ;
                Sr[5] = (P5>>16) & 0xff ;
                Sr[6] = (P6>>16) & 0xff ;
                Sr[7] = (P7>>16) & 0xff ;
                Sr[8] = (P8>>16) & 0xff ;
                Sr[9] = (P9>>16) & 0xff ;
                Sr[10] = (P10>>16) & 0xff ;
                Sr[11] = (P11>>16) & 0xff ;
                Sr[12] = (P12>>16) & 0xff ;
                Sr[13] = (P13>>16) & 0xff ;
                Sr[14] = (P14>>16) & 0xff ;
                Sr[15] = (P15>>16) & 0xff ;
                Sr[16] = (P16>>16) & 0xff ;
                rs_decode_palindromic_CUDA(Sr, S, 17, k, errLocs, errMags,
                                           2+(i<<2), Lambda, B, T, Omega) ;
            }
            if ( T0 & (0xff << 24) )
            {
                atomicAdd(&error_count, 1);
                Sr[0] = (P0>>24) & 0xff ;
                Sr[1] = (P1>>24) & 0xff ;
                Sr[2] = (P2>>24) & 0xff ;
                Sr[3] = (P3>>24) & 0xff ;
                Sr[4] = (P4>>24) & 0xff ;
                Sr[5] = (P5>>24) & 0xff ;
                Sr[6] = (P6>>24) & 0xff ;
                Sr[7] = (P7>>24) & 0xff ;
                Sr[8] = (P8>>24) & 0xff ;
                Sr[9] = (P9>>24) & 0xff ;
                Sr[10] = (P10>>24) & 0xff ;
                Sr[11] = (P11>>24) & 0xff ;
                Sr[12] = (P12>>24) & 0xff ;
                Sr[13] = (P13>>24) & 0xff ;
                Sr[14] = (P14>>24) & 0xff ;
                Sr[15] = (P15>>24) & 0xff ;
                Sr[16] = (P16>>24) & 0xff ;
                rs_decode_palindromic_CUDA(Sr, S, 17, k, errLocs, errMags,
                                           3+(i<<2), Lambda, B, T, Omega) ;
            }
        }
    }
}

// ----------------------------------------------
// Parallel LFSR Sequencer for P = 18 Codewords
// ----------------------------------------------
__global__ void ParallelLFSRSequencer18_CUDA(int k, int decoder)
{
    // Find out place in the grid of threads
    int i = blockIdx.x * blockDim.x + threadIdx.x;

    // Pointer to develop original data address
    unsigned int* loc ;

    // Create the parity values, initialize to zero
    unsigned int P0 = 0 ;
    unsigned int P1 = 0 ;
    unsigned int P2 = 0 ;
    unsigned int P3 = 0 ;
    unsigned int P4 = 0 ;
    unsigned int P5 = 0 ;
    unsigned int P6 = 0 ;
    unsigned int P7 = 0 ;
    unsigned int P8 = 0 ;
    unsigned int P9 = 0 ;
    unsigned int P10 = 0 ;
    unsigned int P11 = 0 ;
    unsigned int P12 = 0 ;
    unsigned int P13 = 0 ;
    unsigned int P14 = 0 ;
    unsigned int P15 = 0 ;
    unsigned int P16 = 0 ;
    unsigned int P17 = 0 ;

    // Temporary values for self reciprocol polynomial
    unsigned int T0;
    unsigned int T1;
    unsigned int T2;
    unsigned int T3;
    unsigned int T4;
    unsigned int T5;
    unsigned int T6;
    unsigned int T7;
    unsigned int T8;

    // Overflow value for LFSR and carry mask
    unsigned int OV, MSK, OF, OOV ;

    // Adjust K if we are decoding
    k += decoder ? 18 : 0 ;

    // Loop through each symbol
    for ( int sym = 0 ; sym < k ; sym ++ )
    {
        loc = CWP[sym] + i;
        OV = *loc ^ P0 ;
        OOV = OV ;

        // Process bit 0
        T1  = OV;                      // Account for bit 0 of constant 9b
        T3  = OV;                      // Account for bit 0 of constant b9
        T4  = OV;                      // Account for bit 0 of constant 13
        T5  = OV;                      // Account for bit 0 of constant 9f
        MSK = OV & 0x80808080 ;        // Create a mask of the MSBs
        OF = (MSK >> 7) * 0x1d ;       // Account for Overflow
        OV <<= 1 ;                     // Multiply the original data by 2
        OV &= 0xfefefefe ;             // Clear the MSBs
        OV ^= OF ;                     // Add in the overflow mask

        // Process bit 1
        T1 ^= OV;                      // Account for bit 1 of constant 9b
        T4 ^= OV;                      // Account for bit 1 of constant 13
        T5 ^= OV;                      // Account for bit 1 of constant 9f
        T6  = OV;                      // Account for bit 1 of constant 6e
        T8  = OV;                      // Account for bit 1 of constant 2e
        MSK = OV & 0x80808080 ;        // Create a mask of the MSBs
        OF = (MSK >> 7) * 0x1d ;       // Account for Overflow
        OV <<= 1 ;                     // Multiply the original data by 2
        OV &= 0xfefefefe ;             // Clear the MSBs
        OV ^= OF ;                     // Add in the overflow mask

        // Process bit 2
        T5 ^= OV;                      // Account for bit 2 of constant 9f
        T6 ^= OV;                      // Account for bit 2 of constant 6e
        T7  = OV;                      // Account for bit 2 of constant 44
        T8 ^= OV;                      // Account for bit 2 of constant 2e
        MSK = OV & 0x80808080 ;        // Create a mask of the MSBs
        OF = (MSK >> 7) * 0x1d ;       // Account for Overflow
        OV <<= 1 ;                     // Multiply the original data by 2
        OV &= 0xfefefefe ;             // Clear the MSBs
        OV ^= OF ;                     // Add in the overflow mask

        // Process bit 3
        T1 ^= OV;                      // Account for bit 3 of constant 9b
        T3 ^= OV;                      // Account for bit 3 of constant b9
        T5 ^= OV;                      // Account for bit 3 of constant 9f
        T6 ^= OV;                      // Account for bit 3 of constant 6e
        T8 ^= OV;                      // Account for bit 3 of constant 2e
        MSK = OV & 0x80808080 ;        // Create a mask of the MSBs
        OF = (MSK >> 7) * 0x1d ;       // Account for Overflow
        OV <<= 1 ;                     // Multiply the original data by 2
        OV &= 0xfefefefe ;             // Clear the MSBs
        OV ^= OF ;                     // Add in the overflow mask

        // Process bit 4
        T0  = OV;                      // Account for bit 4 of constant f0
        T1 ^= OV;                      // Account for bit 4 of constant 9b
        T3 ^= OV;                      // Account for bit 4 of constant b9
        T4 ^= OV;                      // Account for bit 4 of constant 13
        T5 ^= OV;                      // Account for bit 4 of constant 9f
        MSK = OV & 0x80808080 ;        // Create a mask of the MSBs
        OF = (MSK >> 7) * 0x1d ;       // Account for Overflow
        OV <<= 1 ;                     // Multiply the original data by 2
        OV &= 0xfefefefe ;             // Clear the MSBs
        OV ^= OF ;                     // Add in the overflow mask

        // Process bit 5
        T0 ^= OV;                      // Account for bit 5 of constant f0
        T2  = OV;                      // Account for bit 5 of constant 20
        T3 ^= OV;                      // Account for bit 5 of constant b9
        T6 ^= OV;                      // Account for bit 5 of constant 6e
        T8 ^= OV;                      // Account for bit 5 of constant 2e
        MSK = OV & 0x80808080 ;        // Create a mask of the MSBs
        OF = (MSK >> 7) * 0x1d ;       // Account for Overflow
        OV <<= 1 ;                     // Multiply the original data by 2
        OV &= 0xfefefefe ;             // Clear the MSBs
        OV ^= OF ;                     // Add in the overflow mask

        // Process bit 6
        T0 ^= OV;                      // Account for bit 6 of constant f0
        T6 ^= OV;                      // Account for bit 6 of constant 6e
        T7 ^= OV;                      // Account for bit 6 of constant 44
        MSK = OV & 0x80808080 ;        // Create a mask of the MSBs
        OF = (MSK >> 7) * 0x1d ;       // Account for Overflow
        OV <<= 1 ;                     // Multiply the original data by 2
        OV &= 0xfefefefe ;             // Clear the MSBs
        OV ^= OF ;                     // Add in the overflow mask

        // Process bit 7
        T0 ^= OV;                      // Account for bit 7 of constant f0
        T1 ^= OV;                      // Account for bit 7 of constant 9b
        T3 ^= OV;                      // Account for bit 7 of constant b9
        T5 ^= OV;                      // Account for bit 7 of constant 9f

        // Now process the LFSR
        P0 = P1 ^ T0 ;
        P1 = P2 ^ T1 ;
        P2 = P3 ^ T2 ;
        P3 = P4 ^ T3 ;
        P4 = P5 ^ T4 ;
        P5 = P6 ^ T5 ;
        P6 = P7 ^ T6 ;
        P7 = P8 ^ T7 ;
        P8 = P9 ^ T8 ;
        P9 = P10 ^ T7 ;
        P10 = P11 ^ T6 ;
        P11 = P12 ^ T5 ;
        P12 = P13 ^ T4 ;
        P13 = P14 ^ T3 ;
        P14 = P15 ^ T2 ;
        P15 = P16 ^ T1 ;
        P16 = P17 ^ T0 ;
        P17 = OOV ;
    }

    if ( decoder == 0 )
    {
        // Now store computed check symbols back to memory
        loc = CWP[k+0] + i;
        *loc = P0 ;
        loc = CWP[k+1] + i;
        *loc = P1 ;
        loc = CWP[k+2] + i;
        *loc = P2 ;
        loc = CWP[k+3] + i;
        *loc = P3 ;
        loc = CWP[k+4] + i;
        *loc = P4 ;
        loc = CWP[k+5] + i;
        *loc = P5 ;
        loc = CWP[k+6] + i;
        *loc = P6 ;
        loc = CWP[k+7] + i;
        *loc = P7 ;
        loc = CWP[k+8] + i;
        *loc = P8 ;
        loc = CWP[k+9] + i;
        *loc = P9 ;
        loc = CWP[k+10] + i;
        *loc = P10 ;
        loc = CWP[k+11] + i;
        *loc = P11 ;
        loc = CWP[k+12] + i;
        *loc = P12 ;
        loc = CWP[k+13] + i;
        *loc = P13 ;
        loc = CWP[k+14] + i;
        *loc = P14 ;
        loc = CWP[k+15] + i;
        *loc = P15 ;
        loc = CWP[k+16] + i;
        *loc = P16 ;
        loc = CWP[k+17] + i;
        *loc = P17 ;
    }
    else
    {
        // Now check for parity equal to zero
        T0 = P0;
        T0 |= P1;
        T0 |= P2;
        T0 |= P3;
        T0 |= P4;
        T0 |= P5;
        T0 |= P6;
        T0 |= P7;
        T0 |= P8;
        T0 |= P9;
        T0 |= P10;
        T0 |= P11;
        T0 |= P12;
        T0 |= P13;
        T0 |= P14;
        T0 |= P15;
        T0 |= P16;
        T0 |= P17;
        if (T0 != 0)
        {
            unsigned char Sr[18], S[18], errLocs[18], errMags[18] ;
            unsigned char Lambda[18+1], B[18+1], T[18+1], Omega[18+1] ;
            
            if ( T0 & (0xff << 0) )
            {
                atomicAdd(&error_count, 1);
                Sr[0] = (P0>>0) & 0xff ;
                Sr[1] = (P1>>0) & 0xff ;
                Sr[2] = (P2>>0) & 0xff ;
                Sr[3] = (P3>>0) & 0xff ;
                Sr[4] = (P4>>0) & 0xff ;
                Sr[5] = (P5>>0) & 0xff ;
                Sr[6] = (P6>>0) & 0xff ;
                Sr[7] = (P7>>0) & 0xff ;
                Sr[8] = (P8>>0) & 0xff ;
                Sr[9] = (P9>>0) & 0xff ;
                Sr[10] = (P10>>0) & 0xff ;
                Sr[11] = (P11>>0) & 0xff ;
                Sr[12] = (P12>>0) & 0xff ;
                Sr[13] = (P13>>0) & 0xff ;
                Sr[14] = (P14>>0) & 0xff ;
                Sr[15] = (P15>>0) & 0xff ;
                Sr[16] = (P16>>0) & 0xff ;
                Sr[17] = (P17>>0) & 0xff ;
                rs_decode_palindromic_CUDA(Sr, S, 18, k, errLocs, errMags,
                                           0+(i<<2), Lambda, B, T, Omega) ;
            }
            if ( T0 & (0xff << 8) )
            {
                atomicAdd(&error_count, 1);
                Sr[0] = (P0>>8) & 0xff ;
                Sr[1] = (P1>>8) & 0xff ;
                Sr[2] = (P2>>8) & 0xff ;
                Sr[3] = (P3>>8) & 0xff ;
                Sr[4] = (P4>>8) & 0xff ;
                Sr[5] = (P5>>8) & 0xff ;
                Sr[6] = (P6>>8) & 0xff ;
                Sr[7] = (P7>>8) & 0xff ;
                Sr[8] = (P8>>8) & 0xff ;
                Sr[9] = (P9>>8) & 0xff ;
                Sr[10] = (P10>>8) & 0xff ;
                Sr[11] = (P11>>8) & 0xff ;
                Sr[12] = (P12>>8) & 0xff ;
                Sr[13] = (P13>>8) & 0xff ;
                Sr[14] = (P14>>8) & 0xff ;
                Sr[15] = (P15>>8) & 0xff ;
                Sr[16] = (P16>>8) & 0xff ;
                Sr[17] = (P17>>8) & 0xff ;
                rs_decode_palindromic_CUDA(Sr, S, 18, k, errLocs, errMags,
                                           1+(i<<2), Lambda, B, T, Omega) ;
            }
            if ( T0 & (0xff << 16) )
            {
                atomicAdd(&error_count, 1);
                Sr[0] = (P0>>16) & 0xff ;
                Sr[1] = (P1>>16) & 0xff ;
                Sr[2] = (P2>>16) & 0xff ;
                Sr[3] = (P3>>16) & 0xff ;
                Sr[4] = (P4>>16) & 0xff ;
                Sr[5] = (P5>>16) & 0xff ;
                Sr[6] = (P6>>16) & 0xff ;
                Sr[7] = (P7>>16) & 0xff ;
                Sr[8] = (P8>>16) & 0xff ;
                Sr[9] = (P9>>16) & 0xff ;
                Sr[10] = (P10>>16) & 0xff ;
                Sr[11] = (P11>>16) & 0xff ;
                Sr[12] = (P12>>16) & 0xff ;
                Sr[13] = (P13>>16) & 0xff ;
                Sr[14] = (P14>>16) & 0xff ;
                Sr[15] = (P15>>16) & 0xff ;
                Sr[16] = (P16>>16) & 0xff ;
                Sr[17] = (P17>>16) & 0xff ;
                rs_decode_palindromic_CUDA(Sr, S, 18, k, errLocs, errMags,
                                           2+(i<<2), Lambda, B, T, Omega) ;
            }
            if ( T0 & (0xff << 24) )
            {
                atomicAdd(&error_count, 1);
                Sr[0] = (P0>>24) & 0xff ;
                Sr[1] = (P1>>24) & 0xff ;
                Sr[2] = (P2>>24) & 0xff ;
                Sr[3] = (P3>>24) & 0xff ;
                Sr[4] = (P4>>24) & 0xff ;
                Sr[5] = (P5>>24) & 0xff ;
                Sr[6] = (P6>>24) & 0xff ;
                Sr[7] = (P7>>24) & 0xff ;
                Sr[8] = (P8>>24) & 0xff ;
                Sr[9] = (P9>>24) & 0xff ;
                Sr[10] = (P10>>24) & 0xff ;
                Sr[11] = (P11>>24) & 0xff ;
                Sr[12] = (P12>>24) & 0xff ;
                Sr[13] = (P13>>24) & 0xff ;
                Sr[14] = (P14>>24) & 0xff ;
                Sr[15] = (P15>>24) & 0xff ;
                Sr[16] = (P16>>24) & 0xff ;
                Sr[17] = (P17>>24) & 0xff ;
                rs_decode_palindromic_CUDA(Sr, S, 18, k, errLocs, errMags,
                                           3+(i<<2), Lambda, B, T, Omega) ;
            }
        }
    }
}

// ----------------------------------------------
// Parallel LFSR Sequencer for P = 19 Codewords
// ----------------------------------------------
__global__ void ParallelLFSRSequencer19_CUDA(int k, int decoder)
{
    // Find out place in the grid of threads
    int i = blockIdx.x * blockDim.x + threadIdx.x;

    // Pointer to develop original data address
    unsigned int* loc ;

    // Create the parity values, initialize to zero
    unsigned int P0 = 0 ;
    unsigned int P1 = 0 ;
    unsigned int P2 = 0 ;
    unsigned int P3 = 0 ;
    unsigned int P4 = 0 ;
    unsigned int P5 = 0 ;
    unsigned int P6 = 0 ;
    unsigned int P7 = 0 ;
    unsigned int P8 = 0 ;
    unsigned int P9 = 0 ;
    unsigned int P10 = 0 ;
    unsigned int P11 = 0 ;
    unsigned int P12 = 0 ;
    unsigned int P13 = 0 ;
    unsigned int P14 = 0 ;
    unsigned int P15 = 0 ;
    unsigned int P16 = 0 ;
    unsigned int P17 = 0 ;
    unsigned int P18 = 0 ;

    // Temporary values for self reciprocol polynomial
    unsigned int T0;
    unsigned int T1;
    unsigned int T2;
    unsigned int T3;
    unsigned int T4;
    unsigned int T5;
    unsigned int T6;
    unsigned int T7;
    unsigned int T8;

    // Overflow value for LFSR and carry mask
    unsigned int OV, MSK, OF, OOV ;

    // Adjust K if we are decoding
    k += decoder ? 19 : 0 ;

    // Loop through each symbol
    for ( int sym = 0 ; sym < k ; sym ++ )
    {
        loc = CWP[sym] + i;
        OV = *loc ^ P0 ;
        OOV = OV ;

        // Process bit 0
        T0  = OV;                      // Account for bit 0 of constant 69
        T3  = OV;                      // Account for bit 0 of constant ed
        T5  = OV;                      // Account for bit 0 of constant 25
        T8  = OV;                      // Account for bit 0 of constant 2f
        MSK = OV & 0x80808080 ;        // Create a mask of the MSBs
        OF = (MSK >> 7) * 0x1d ;       // Account for Overflow
        OV <<= 1 ;                     // Multiply the original data by 2
        OV &= 0xfefefefe ;             // Clear the MSBs
        OV ^= OF ;                     // Add in the overflow mask

        // Process bit 1
        T8 ^= OV;                      // Account for bit 1 of constant 2f
        MSK = OV & 0x80808080 ;        // Create a mask of the MSBs
        OF = (MSK >> 7) * 0x1d ;       // Account for Overflow
        OV <<= 1 ;                     // Multiply the original data by 2
        OV &= 0xfefefefe ;             // Clear the MSBs
        OV ^= OF ;                     // Add in the overflow mask

        // Process bit 2
        T1  = OV;                      // Account for bit 2 of constant 2c
        T3 ^= OV;                      // Account for bit 2 of constant ed
        T5 ^= OV;                      // Account for bit 2 of constant 25
        T6  = OV;                      // Account for bit 2 of constant b4
        T7  = OV;                      // Account for bit 2 of constant 1c
        T8 ^= OV;                      // Account for bit 2 of constant 2f
        MSK = OV & 0x80808080 ;        // Create a mask of the MSBs
        OF = (MSK >> 7) * 0x1d ;       // Account for Overflow
        OV <<= 1 ;                     // Multiply the original data by 2
        OV &= 0xfefefefe ;             // Clear the MSBs
        OV ^= OF ;                     // Add in the overflow mask

        // Process bit 3
        T0 ^= OV;                      // Account for bit 3 of constant 69
        T1 ^= OV;                      // Account for bit 3 of constant 2c
        T2  = OV;                      // Account for bit 3 of constant 78
        T3 ^= OV;                      // Account for bit 3 of constant ed
        T7 ^= OV;                      // Account for bit 3 of constant 1c
        T8 ^= OV;                      // Account for bit 3 of constant 2f
        MSK = OV & 0x80808080 ;        // Create a mask of the MSBs
        OF = (MSK >> 7) * 0x1d ;       // Account for Overflow
        OV <<= 1 ;                     // Multiply the original data by 2
        OV &= 0xfefefefe ;             // Clear the MSBs
        OV ^= OF ;                     // Add in the overflow mask

        // Process bit 4
        T2 ^= OV;                      // Account for bit 4 of constant 78
        T6 ^= OV;                      // Account for bit 4 of constant b4
        T7 ^= OV;                      // Account for bit 4 of constant 1c
        MSK = OV & 0x80808080 ;        // Create a mask of the MSBs
        OF = (MSK >> 7) * 0x1d ;       // Account for Overflow
        OV <<= 1 ;                     // Multiply the original data by 2
        OV &= 0xfefefefe ;             // Clear the MSBs
        OV ^= OF ;                     // Add in the overflow mask

        // Process bit 5
        T0 ^= OV;                      // Account for bit 5 of constant 69
        T1 ^= OV;                      // Account for bit 5 of constant 2c
        T2 ^= OV;                      // Account for bit 5 of constant 78
        T3 ^= OV;                      // Account for bit 5 of constant ed
        T5 ^= OV;                      // Account for bit 5 of constant 25
        T6 ^= OV;                      // Account for bit 5 of constant b4
        T8 ^= OV;                      // Account for bit 5 of constant 2f
        MSK = OV & 0x80808080 ;        // Create a mask of the MSBs
        OF = (MSK >> 7) * 0x1d ;       // Account for Overflow
        OV <<= 1 ;                     // Multiply the original data by 2
        OV &= 0xfefefefe ;             // Clear the MSBs
        OV ^= OF ;                     // Add in the overflow mask

        // Process bit 6
        T0 ^= OV;                      // Account for bit 6 of constant 69
        T2 ^= OV;                      // Account for bit 6 of constant 78
        T3 ^= OV;                      // Account for bit 6 of constant ed
        MSK = OV & 0x80808080 ;        // Create a mask of the MSBs
        OF = (MSK >> 7) * 0x1d ;       // Account for Overflow
        OV <<= 1 ;                     // Multiply the original data by 2
        OV &= 0xfefefefe ;             // Clear the MSBs
        OV ^= OF ;                     // Add in the overflow mask

        // Process bit 7
        T3 ^= OV;                      // Account for bit 7 of constant ed
        T4  = OV;                      // Account for bit 7 of constant 80
        T6 ^= OV;                      // Account for bit 7 of constant b4

        // Now process the LFSR
        P0 = P1 ^ T0 ;
        P1 = P2 ^ T1 ;
        P2 = P3 ^ T2 ;
        P3 = P4 ^ T3 ;
        P4 = P5 ^ T4 ;
        P5 = P6 ^ T5 ;
        P6 = P7 ^ T6 ;
        P7 = P8 ^ T7 ;
        P8 = P9 ^ T8 ;
        P9 = P10 ^ T8 ;
        P10 = P11 ^ T7 ;
        P11 = P12 ^ T6 ;
        P12 = P13 ^ T5 ;
        P13 = P14 ^ T4 ;
        P14 = P15 ^ T3 ;
        P15 = P16 ^ T2 ;
        P16 = P17 ^ T1 ;
        P17 = P18 ^ T0 ;
        P18 = OOV ;
    }

    if ( decoder == 0 )
    {
        // Now store computed check symbols back to memory
        loc = CWP[k+0] + i;
        *loc = P0 ;
        loc = CWP[k+1] + i;
        *loc = P1 ;
        loc = CWP[k+2] + i;
        *loc = P2 ;
        loc = CWP[k+3] + i;
        *loc = P3 ;
        loc = CWP[k+4] + i;
        *loc = P4 ;
        loc = CWP[k+5] + i;
        *loc = P5 ;
        loc = CWP[k+6] + i;
        *loc = P6 ;
        loc = CWP[k+7] + i;
        *loc = P7 ;
        loc = CWP[k+8] + i;
        *loc = P8 ;
        loc = CWP[k+9] + i;
        *loc = P9 ;
        loc = CWP[k+10] + i;
        *loc = P10 ;
        loc = CWP[k+11] + i;
        *loc = P11 ;
        loc = CWP[k+12] + i;
        *loc = P12 ;
        loc = CWP[k+13] + i;
        *loc = P13 ;
        loc = CWP[k+14] + i;
        *loc = P14 ;
        loc = CWP[k+15] + i;
        *loc = P15 ;
        loc = CWP[k+16] + i;
        *loc = P16 ;
        loc = CWP[k+17] + i;
        *loc = P17 ;
        loc = CWP[k+18] + i;
        *loc = P18 ;
    }
    else
    {
        // Now check for parity equal to zero
        T0 = P0;
        T0 |= P1;
        T0 |= P2;
        T0 |= P3;
        T0 |= P4;
        T0 |= P5;
        T0 |= P6;
        T0 |= P7;
        T0 |= P8;
        T0 |= P9;
        T0 |= P10;
        T0 |= P11;
        T0 |= P12;
        T0 |= P13;
        T0 |= P14;
        T0 |= P15;
        T0 |= P16;
        T0 |= P17;
        T0 |= P18;
        if (T0 != 0)
        {
            unsigned char Sr[19], S[19], errLocs[19], errMags[19] ;
            unsigned char Lambda[19+1], B[19+1], T[19+1], Omega[19+1] ;
            
            if ( T0 & (0xff << 0) )
            {
                atomicAdd(&error_count, 1);
                Sr[0] = (P0>>0) & 0xff ;
                Sr[1] = (P1>>0) & 0xff ;
                Sr[2] = (P2>>0) & 0xff ;
                Sr[3] = (P3>>0) & 0xff ;
                Sr[4] = (P4>>0) & 0xff ;
                Sr[5] = (P5>>0) & 0xff ;
                Sr[6] = (P6>>0) & 0xff ;
                Sr[7] = (P7>>0) & 0xff ;
                Sr[8] = (P8>>0) & 0xff ;
                Sr[9] = (P9>>0) & 0xff ;
                Sr[10] = (P10>>0) & 0xff ;
                Sr[11] = (P11>>0) & 0xff ;
                Sr[12] = (P12>>0) & 0xff ;
                Sr[13] = (P13>>0) & 0xff ;
                Sr[14] = (P14>>0) & 0xff ;
                Sr[15] = (P15>>0) & 0xff ;
                Sr[16] = (P16>>0) & 0xff ;
                Sr[17] = (P17>>0) & 0xff ;
                Sr[18] = (P18>>0) & 0xff ;
                rs_decode_palindromic_CUDA(Sr, S, 19, k, errLocs, errMags,
                                           0+(i<<2), Lambda, B, T, Omega) ;
            }
            if ( T0 & (0xff << 8) )
            {
                atomicAdd(&error_count, 1);
                Sr[0] = (P0>>8) & 0xff ;
                Sr[1] = (P1>>8) & 0xff ;
                Sr[2] = (P2>>8) & 0xff ;
                Sr[3] = (P3>>8) & 0xff ;
                Sr[4] = (P4>>8) & 0xff ;
                Sr[5] = (P5>>8) & 0xff ;
                Sr[6] = (P6>>8) & 0xff ;
                Sr[7] = (P7>>8) & 0xff ;
                Sr[8] = (P8>>8) & 0xff ;
                Sr[9] = (P9>>8) & 0xff ;
                Sr[10] = (P10>>8) & 0xff ;
                Sr[11] = (P11>>8) & 0xff ;
                Sr[12] = (P12>>8) & 0xff ;
                Sr[13] = (P13>>8) & 0xff ;
                Sr[14] = (P14>>8) & 0xff ;
                Sr[15] = (P15>>8) & 0xff ;
                Sr[16] = (P16>>8) & 0xff ;
                Sr[17] = (P17>>8) & 0xff ;
                Sr[18] = (P18>>8) & 0xff ;
                rs_decode_palindromic_CUDA(Sr, S, 19, k, errLocs, errMags,
                                           1+(i<<2), Lambda, B, T, Omega) ;
            }
            if ( T0 & (0xff << 16) )
            {
                atomicAdd(&error_count, 1);
                Sr[0] = (P0>>16) & 0xff ;
                Sr[1] = (P1>>16) & 0xff ;
                Sr[2] = (P2>>16) & 0xff ;
                Sr[3] = (P3>>16) & 0xff ;
                Sr[4] = (P4>>16) & 0xff ;
                Sr[5] = (P5>>16) & 0xff ;
                Sr[6] = (P6>>16) & 0xff ;
                Sr[7] = (P7>>16) & 0xff ;
                Sr[8] = (P8>>16) & 0xff ;
                Sr[9] = (P9>>16) & 0xff ;
                Sr[10] = (P10>>16) & 0xff ;
                Sr[11] = (P11>>16) & 0xff ;
                Sr[12] = (P12>>16) & 0xff ;
                Sr[13] = (P13>>16) & 0xff ;
                Sr[14] = (P14>>16) & 0xff ;
                Sr[15] = (P15>>16) & 0xff ;
                Sr[16] = (P16>>16) & 0xff ;
                Sr[17] = (P17>>16) & 0xff ;
                Sr[18] = (P18>>16) & 0xff ;
                rs_decode_palindromic_CUDA(Sr, S, 19, k, errLocs, errMags,
                                           2+(i<<2), Lambda, B, T, Omega) ;
            }
            if ( T0 & (0xff << 24) )
            {
                atomicAdd(&error_count, 1);
                Sr[0] = (P0>>24) & 0xff ;
                Sr[1] = (P1>>24) & 0xff ;
                Sr[2] = (P2>>24) & 0xff ;
                Sr[3] = (P3>>24) & 0xff ;
                Sr[4] = (P4>>24) & 0xff ;
                Sr[5] = (P5>>24) & 0xff ;
                Sr[6] = (P6>>24) & 0xff ;
                Sr[7] = (P7>>24) & 0xff ;
                Sr[8] = (P8>>24) & 0xff ;
                Sr[9] = (P9>>24) & 0xff ;
                Sr[10] = (P10>>24) & 0xff ;
                Sr[11] = (P11>>24) & 0xff ;
                Sr[12] = (P12>>24) & 0xff ;
                Sr[13] = (P13>>24) & 0xff ;
                Sr[14] = (P14>>24) & 0xff ;
                Sr[15] = (P15>>24) & 0xff ;
                Sr[16] = (P16>>24) & 0xff ;
                Sr[17] = (P17>>24) & 0xff ;
                Sr[18] = (P18>>24) & 0xff ;
                rs_decode_palindromic_CUDA(Sr, S, 19, k, errLocs, errMags,
                                           3+(i<<2), Lambda, B, T, Omega) ;
            }
        }
    }
}

// ----------------------------------------------
// Parallel LFSR Sequencer for P = 20 Codewords
// ----------------------------------------------
__global__ void ParallelLFSRSequencer20_CUDA(int k, int decoder)
{
    // Find out place in the grid of threads
    int i = blockIdx.x * blockDim.x + threadIdx.x;

    // Pointer to develop original data address
    unsigned int* loc ;

    // Create the parity values, initialize to zero
    unsigned int P0 = 0 ;
    unsigned int P1 = 0 ;
    unsigned int P2 = 0 ;
    unsigned int P3 = 0 ;
    unsigned int P4 = 0 ;
    unsigned int P5 = 0 ;
    unsigned int P6 = 0 ;
    unsigned int P7 = 0 ;
    unsigned int P8 = 0 ;
    unsigned int P9 = 0 ;
    unsigned int P10 = 0 ;
    unsigned int P11 = 0 ;
    unsigned int P12 = 0 ;
    unsigned int P13 = 0 ;
    unsigned int P14 = 0 ;
    unsigned int P15 = 0 ;
    unsigned int P16 = 0 ;
    unsigned int P17 = 0 ;
    unsigned int P18 = 0 ;
    unsigned int P19 = 0 ;

    // Temporary values for self reciprocol polynomial
    unsigned int T0;
    unsigned int T1;
    unsigned int T2;
    unsigned int T3;
    unsigned int T4;
    unsigned int T5;
    unsigned int T6;
    unsigned int T7;
    unsigned int T8;
    unsigned int T9;

    // Overflow value for LFSR and carry mask
    unsigned int OV, MSK, OF, OOV ;

    // Adjust K if we are decoding
    k += decoder ? 20 : 0 ;

    // Loop through each symbol
    for ( int sym = 0 ; sym < k ; sym ++ )
    {
        loc = CWP[sym] + i;
        OV = *loc ^ P0 ;
        OOV = OV ;

        // Process bit 0
        T0  = OV;                      // Account for bit 0 of constant a9
        T2  = OV;                      // Account for bit 0 of constant ab
        T3  = OV;                      // Account for bit 0 of constant cd
        T4  = OV;                      // Account for bit 0 of constant 15
        T6  = OV;                      // Account for bit 0 of constant 7f
        MSK = OV & 0x80808080 ;        // Create a mask of the MSBs
        OF = (MSK >> 7) * 0x1d ;       // Account for Overflow
        OV <<= 1 ;                     // Multiply the original data by 2
        OV &= 0xfefefefe ;             // Clear the MSBs
        OV ^= OF ;                     // Add in the overflow mask

        // Process bit 1
        T2 ^= OV;                      // Account for bit 1 of constant ab
        T6 ^= OV;                      // Account for bit 1 of constant 7f
        T7  = OV;                      // Account for bit 1 of constant ce
        T8  = OV;                      // Account for bit 1 of constant fa
        T9  = OV;                      // Account for bit 1 of constant d6
        MSK = OV & 0x80808080 ;        // Create a mask of the MSBs
        OF = (MSK >> 7) * 0x1d ;       // Account for Overflow
        OV <<= 1 ;                     // Multiply the original data by 2
        OV &= 0xfefefefe ;             // Clear the MSBs
        OV ^= OF ;                     // Add in the overflow mask

        // Process bit 2
        T1  = OV;                      // Account for bit 2 of constant d4
        T3 ^= OV;                      // Account for bit 2 of constant cd
        T4 ^= OV;                      // Account for bit 2 of constant 15
        T5  = OV;                      // Account for bit 2 of constant 34
        T6 ^= OV;                      // Account for bit 2 of constant 7f
        T7 ^= OV;                      // Account for bit 2 of constant ce
        T9 ^= OV;                      // Account for bit 2 of constant d6
        MSK = OV & 0x80808080 ;        // Create a mask of the MSBs
        OF = (MSK >> 7) * 0x1d ;       // Account for Overflow
        OV <<= 1 ;                     // Multiply the original data by 2
        OV &= 0xfefefefe ;             // Clear the MSBs
        OV ^= OF ;                     // Add in the overflow mask

        // Process bit 3
        T0 ^= OV;                      // Account for bit 3 of constant a9
        T2 ^= OV;                      // Account for bit 3 of constant ab
        T3 ^= OV;                      // Account for bit 3 of constant cd
        T6 ^= OV;                      // Account for bit 3 of constant 7f
        T7 ^= OV;                      // Account for bit 3 of constant ce
        T8 ^= OV;                      // Account for bit 3 of constant fa
        MSK = OV & 0x80808080 ;        // Create a mask of the MSBs
        OF = (MSK >> 7) * 0x1d ;       // Account for Overflow
        OV <<= 1 ;                     // Multiply the original data by 2
        OV &= 0xfefefefe ;             // Clear the MSBs
        OV ^= OF ;                     // Add in the overflow mask

        // Process bit 4
        T1 ^= OV;                      // Account for bit 4 of constant d4
        T4 ^= OV;                      // Account for bit 4 of constant 15
        T5 ^= OV;                      // Account for bit 4 of constant 34
        T6 ^= OV;                      // Account for bit 4 of constant 7f
        T8 ^= OV;                      // Account for bit 4 of constant fa
        T9 ^= OV;                      // Account for bit 4 of constant d6
        MSK = OV & 0x80808080 ;        // Create a mask of the MSBs
        OF = (MSK >> 7) * 0x1d ;       // Account for Overflow
        OV <<= 1 ;                     // Multiply the original data by 2
        OV &= 0xfefefefe ;             // Clear the MSBs
        OV ^= OF ;                     // Add in the overflow mask

        // Process bit 5
        T0 ^= OV;                      // Account for bit 5 of constant a9
        T2 ^= OV;                      // Account for bit 5 of constant ab
        T5 ^= OV;                      // Account for bit 5 of constant 34
        T6 ^= OV;                      // Account for bit 5 of constant 7f
        T8 ^= OV;                      // Account for bit 5 of constant fa
        MSK = OV & 0x80808080 ;        // Create a mask of the MSBs
        OF = (MSK >> 7) * 0x1d ;       // Account for Overflow
        OV <<= 1 ;                     // Multiply the original data by 2
        OV &= 0xfefefefe ;             // Clear the MSBs
        OV ^= OF ;                     // Add in the overflow mask

        // Process bit 6
        T1 ^= OV;                      // Account for bit 6 of constant d4
        T3 ^= OV;                      // Account for bit 6 of constant cd
        T6 ^= OV;                      // Account for bit 6 of constant 7f
        T7 ^= OV;                      // Account for bit 6 of constant ce
        T8 ^= OV;                      // Account for bit 6 of constant fa
        T9 ^= OV;                      // Account for bit 6 of constant d6
        MSK = OV & 0x80808080 ;        // Create a mask of the MSBs
        OF = (MSK >> 7) * 0x1d ;       // Account for Overflow
        OV <<= 1 ;                     // Multiply the original data by 2
        OV &= 0xfefefefe ;             // Clear the MSBs
        OV ^= OF ;                     // Add in the overflow mask

        // Process bit 7
        T0 ^= OV;                      // Account for bit 7 of constant a9
        T1 ^= OV;                      // Account for bit 7 of constant d4
        T2 ^= OV;                      // Account for bit 7 of constant ab
        T3 ^= OV;                      // Account for bit 7 of constant cd
        T7 ^= OV;                      // Account for bit 7 of constant ce
        T8 ^= OV;                      // Account for bit 7 of constant fa
        T9 ^= OV;                      // Account for bit 7 of constant d6

        // Now process the LFSR
        P0 = P1 ^ T0 ;
        P1 = P2 ^ T1 ;
        P2 = P3 ^ T2 ;
        P3 = P4 ^ T3 ;
        P4 = P5 ^ T4 ;
        P5 = P6 ^ T5 ;
        P6 = P7 ^ T6 ;
        P7 = P8 ^ T7 ;
        P8 = P9 ^ T8 ;
        P9 = P10 ^ T9 ;
        P10 = P11 ^ T8 ;
        P11 = P12 ^ T7 ;
        P12 = P13 ^ T6 ;
        P13 = P14 ^ T5 ;
        P14 = P15 ^ T4 ;
        P15 = P16 ^ T3 ;
        P16 = P17 ^ T2 ;
        P17 = P18 ^ T1 ;
        P18 = P19 ^ T0 ;
        P19 = OOV ;
    }

    if ( decoder == 0 )
    {
        // Now store computed check symbols back to memory
        loc = CWP[k+0] + i;
        *loc = P0 ;
        loc = CWP[k+1] + i;
        *loc = P1 ;
        loc = CWP[k+2] + i;
        *loc = P2 ;
        loc = CWP[k+3] + i;
        *loc = P3 ;
        loc = CWP[k+4] + i;
        *loc = P4 ;
        loc = CWP[k+5] + i;
        *loc = P5 ;
        loc = CWP[k+6] + i;
        *loc = P6 ;
        loc = CWP[k+7] + i;
        *loc = P7 ;
        loc = CWP[k+8] + i;
        *loc = P8 ;
        loc = CWP[k+9] + i;
        *loc = P9 ;
        loc = CWP[k+10] + i;
        *loc = P10 ;
        loc = CWP[k+11] + i;
        *loc = P11 ;
        loc = CWP[k+12] + i;
        *loc = P12 ;
        loc = CWP[k+13] + i;
        *loc = P13 ;
        loc = CWP[k+14] + i;
        *loc = P14 ;
        loc = CWP[k+15] + i;
        *loc = P15 ;
        loc = CWP[k+16] + i;
        *loc = P16 ;
        loc = CWP[k+17] + i;
        *loc = P17 ;
        loc = CWP[k+18] + i;
        *loc = P18 ;
        loc = CWP[k+19] + i;
        *loc = P19 ;
    }
    else
    {
        // Now check for parity equal to zero
        T0 = P0;
        T0 |= P1;
        T0 |= P2;
        T0 |= P3;
        T0 |= P4;
        T0 |= P5;
        T0 |= P6;
        T0 |= P7;
        T0 |= P8;
        T0 |= P9;
        T0 |= P10;
        T0 |= P11;
        T0 |= P12;
        T0 |= P13;
        T0 |= P14;
        T0 |= P15;
        T0 |= P16;
        T0 |= P17;
        T0 |= P18;
        T0 |= P19;
        if (T0 != 0)
        {
            unsigned char Sr[20], S[20], errLocs[20], errMags[20] ;
            unsigned char Lambda[20+1], B[20+1], T[20+1], Omega[20+1] ;
            
            if ( T0 & (0xff << 0) )
            {
                atomicAdd(&error_count, 1);
                Sr[0] = (P0>>0) & 0xff ;
                Sr[1] = (P1>>0) & 0xff ;
                Sr[2] = (P2>>0) & 0xff ;
                Sr[3] = (P3>>0) & 0xff ;
                Sr[4] = (P4>>0) & 0xff ;
                Sr[5] = (P5>>0) & 0xff ;
                Sr[6] = (P6>>0) & 0xff ;
                Sr[7] = (P7>>0) & 0xff ;
                Sr[8] = (P8>>0) & 0xff ;
                Sr[9] = (P9>>0) & 0xff ;
                Sr[10] = (P10>>0) & 0xff ;
                Sr[11] = (P11>>0) & 0xff ;
                Sr[12] = (P12>>0) & 0xff ;
                Sr[13] = (P13>>0) & 0xff ;
                Sr[14] = (P14>>0) & 0xff ;
                Sr[15] = (P15>>0) & 0xff ;
                Sr[16] = (P16>>0) & 0xff ;
                Sr[17] = (P17>>0) & 0xff ;
                Sr[18] = (P18>>0) & 0xff ;
                Sr[19] = (P19>>0) & 0xff ;
                rs_decode_palindromic_CUDA(Sr, S, 20, k, errLocs, errMags,
                                           0+(i<<2), Lambda, B, T, Omega) ;
            }
            if ( T0 & (0xff << 8) )
            {
                atomicAdd(&error_count, 1);
                Sr[0] = (P0>>8) & 0xff ;
                Sr[1] = (P1>>8) & 0xff ;
                Sr[2] = (P2>>8) & 0xff ;
                Sr[3] = (P3>>8) & 0xff ;
                Sr[4] = (P4>>8) & 0xff ;
                Sr[5] = (P5>>8) & 0xff ;
                Sr[6] = (P6>>8) & 0xff ;
                Sr[7] = (P7>>8) & 0xff ;
                Sr[8] = (P8>>8) & 0xff ;
                Sr[9] = (P9>>8) & 0xff ;
                Sr[10] = (P10>>8) & 0xff ;
                Sr[11] = (P11>>8) & 0xff ;
                Sr[12] = (P12>>8) & 0xff ;
                Sr[13] = (P13>>8) & 0xff ;
                Sr[14] = (P14>>8) & 0xff ;
                Sr[15] = (P15>>8) & 0xff ;
                Sr[16] = (P16>>8) & 0xff ;
                Sr[17] = (P17>>8) & 0xff ;
                Sr[18] = (P18>>8) & 0xff ;
                Sr[19] = (P19>>8) & 0xff ;
                rs_decode_palindromic_CUDA(Sr, S, 20, k, errLocs, errMags,
                                           1+(i<<2), Lambda, B, T, Omega) ;
            }
            if ( T0 & (0xff << 16) )
            {
                atomicAdd(&error_count, 1);
                Sr[0] = (P0>>16) & 0xff ;
                Sr[1] = (P1>>16) & 0xff ;
                Sr[2] = (P2>>16) & 0xff ;
                Sr[3] = (P3>>16) & 0xff ;
                Sr[4] = (P4>>16) & 0xff ;
                Sr[5] = (P5>>16) & 0xff ;
                Sr[6] = (P6>>16) & 0xff ;
                Sr[7] = (P7>>16) & 0xff ;
                Sr[8] = (P8>>16) & 0xff ;
                Sr[9] = (P9>>16) & 0xff ;
                Sr[10] = (P10>>16) & 0xff ;
                Sr[11] = (P11>>16) & 0xff ;
                Sr[12] = (P12>>16) & 0xff ;
                Sr[13] = (P13>>16) & 0xff ;
                Sr[14] = (P14>>16) & 0xff ;
                Sr[15] = (P15>>16) & 0xff ;
                Sr[16] = (P16>>16) & 0xff ;
                Sr[17] = (P17>>16) & 0xff ;
                Sr[18] = (P18>>16) & 0xff ;
                Sr[19] = (P19>>16) & 0xff ;
                rs_decode_palindromic_CUDA(Sr, S, 20, k, errLocs, errMags,
                                           2+(i<<2), Lambda, B, T, Omega) ;
            }
            if ( T0 & (0xff << 24) )
            {
                atomicAdd(&error_count, 1);
                Sr[0] = (P0>>24) & 0xff ;
                Sr[1] = (P1>>24) & 0xff ;
                Sr[2] = (P2>>24) & 0xff ;
                Sr[3] = (P3>>24) & 0xff ;
                Sr[4] = (P4>>24) & 0xff ;
                Sr[5] = (P5>>24) & 0xff ;
                Sr[6] = (P6>>24) & 0xff ;
                Sr[7] = (P7>>24) & 0xff ;
                Sr[8] = (P8>>24) & 0xff ;
                Sr[9] = (P9>>24) & 0xff ;
                Sr[10] = (P10>>24) & 0xff ;
                Sr[11] = (P11>>24) & 0xff ;
                Sr[12] = (P12>>24) & 0xff ;
                Sr[13] = (P13>>24) & 0xff ;
                Sr[14] = (P14>>24) & 0xff ;
                Sr[15] = (P15>>24) & 0xff ;
                Sr[16] = (P16>>24) & 0xff ;
                Sr[17] = (P17>>24) & 0xff ;
                Sr[18] = (P18>>24) & 0xff ;
                Sr[19] = (P19>>24) & 0xff ;
                rs_decode_palindromic_CUDA(Sr, S, 20, k, errLocs, errMags,
                                           3+(i<<2), Lambda, B, T, Omega) ;
            }
        }
    }
}

// ----------------------------------------------
// Parallel LFSR Sequencer for P = 21 Codewords
// ----------------------------------------------
__global__ void ParallelLFSRSequencer21_CUDA(int k, int decoder)
{
    // Find out place in the grid of threads
    int i = blockIdx.x * blockDim.x + threadIdx.x;

    // Pointer to develop original data address
    unsigned int* loc ;

    // Create the parity values, initialize to zero
    unsigned int P0 = 0 ;
    unsigned int P1 = 0 ;
    unsigned int P2 = 0 ;
    unsigned int P3 = 0 ;
    unsigned int P4 = 0 ;
    unsigned int P5 = 0 ;
    unsigned int P6 = 0 ;
    unsigned int P7 = 0 ;
    unsigned int P8 = 0 ;
    unsigned int P9 = 0 ;
    unsigned int P10 = 0 ;
    unsigned int P11 = 0 ;
    unsigned int P12 = 0 ;
    unsigned int P13 = 0 ;
    unsigned int P14 = 0 ;
    unsigned int P15 = 0 ;
    unsigned int P16 = 0 ;
    unsigned int P17 = 0 ;
    unsigned int P18 = 0 ;
    unsigned int P19 = 0 ;
    unsigned int P20 = 0 ;

    // Temporary values for self reciprocol polynomial
    unsigned int T0;
    unsigned int T1;
    unsigned int T2;
    unsigned int T3;
    unsigned int T4;
    unsigned int T5;
    unsigned int T6;
    unsigned int T7;
    unsigned int T8;
    unsigned int T9;

    // Overflow value for LFSR and carry mask
    unsigned int OV, MSK, OF, OOV ;

    // Adjust K if we are decoding
    k += decoder ? 21 : 0 ;

    // Loop through each symbol
    for ( int sym = 0 ; sym < k ; sym ++ )
    {
        loc = CWP[sym] + i;
        OV = *loc ^ P0 ;
        OOV = OV ;

        // Process bit 0
        T2  = OV;                      // Account for bit 0 of constant 89
        T5  = OV;                      // Account for bit 0 of constant fd
        T6  = OV;                      // Account for bit 0 of constant ad
        T7  = OV;                      // Account for bit 0 of constant 2d
        T9  = OV;                      // Account for bit 0 of constant 11
        MSK = OV & 0x80808080 ;        // Create a mask of the MSBs
        OF = (MSK >> 7) * 0x1d ;       // Account for Overflow
        OV <<= 1 ;                     // Multiply the original data by 2
        OV &= 0xfefefefe ;             // Clear the MSBs
        OV ^= OF ;                     // Add in the overflow mask

        // Process bit 1
        T1  = OV;                      // Account for bit 1 of constant f2
        T3  = OV;                      // Account for bit 1 of constant a6
        MSK = OV & 0x80808080 ;        // Create a mask of the MSBs
        OF = (MSK >> 7) * 0x1d ;       // Account for Overflow
        OV <<= 1 ;                     // Multiply the original data by 2
        OV &= 0xfefefefe ;             // Clear the MSBs
        OV ^= OF ;                     // Add in the overflow mask

        // Process bit 2
        T0  = OV;                      // Account for bit 2 of constant f4
        T3 ^= OV;                      // Account for bit 2 of constant a6
        T4  = OV;                      // Account for bit 2 of constant 5c
        T5 ^= OV;                      // Account for bit 2 of constant fd
        T6 ^= OV;                      // Account for bit 2 of constant ad
        T7 ^= OV;                      // Account for bit 2 of constant 2d
        T8  = OV;                      // Account for bit 2 of constant 6c
        MSK = OV & 0x80808080 ;        // Create a mask of the MSBs
        OF = (MSK >> 7) * 0x1d ;       // Account for Overflow
        OV <<= 1 ;                     // Multiply the original data by 2
        OV &= 0xfefefefe ;             // Clear the MSBs
        OV ^= OF ;                     // Add in the overflow mask

        // Process bit 3
        T2 ^= OV;                      // Account for bit 3 of constant 89
        T4 ^= OV;                      // Account for bit 3 of constant 5c
        T5 ^= OV;                      // Account for bit 3 of constant fd
        T6 ^= OV;                      // Account for bit 3 of constant ad
        T7 ^= OV;                      // Account for bit 3 of constant 2d
        T8 ^= OV;                      // Account for bit 3 of constant 6c
        MSK = OV & 0x80808080 ;        // Create a mask of the MSBs
        OF = (MSK >> 7) * 0x1d ;       // Account for Overflow
        OV <<= 1 ;                     // Multiply the original data by 2
        OV &= 0xfefefefe ;             // Clear the MSBs
        OV ^= OF ;                     // Add in the overflow mask

        // Process bit 4
        T0 ^= OV;                      // Account for bit 4 of constant f4
        T1 ^= OV;                      // Account for bit 4 of constant f2
        T4 ^= OV;                      // Account for bit 4 of constant 5c
        T5 ^= OV;                      // Account for bit 4 of constant fd
        T9 ^= OV;                      // Account for bit 4 of constant 11
        MSK = OV & 0x80808080 ;        // Create a mask of the MSBs
        OF = (MSK >> 7) * 0x1d ;       // Account for Overflow
        OV <<= 1 ;                     // Multiply the original data by 2
        OV &= 0xfefefefe ;             // Clear the MSBs
        OV ^= OF ;                     // Add in the overflow mask

        // Process bit 5
        T0 ^= OV;                      // Account for bit 5 of constant f4
        T1 ^= OV;                      // Account for bit 5 of constant f2
        T3 ^= OV;                      // Account for bit 5 of constant a6
        T5 ^= OV;                      // Account for bit 5 of constant fd
        T6 ^= OV;                      // Account for bit 5 of constant ad
        T7 ^= OV;                      // Account for bit 5 of constant 2d
        T8 ^= OV;                      // Account for bit 5 of constant 6c
        MSK = OV & 0x80808080 ;        // Create a mask of the MSBs
        OF = (MSK >> 7) * 0x1d ;       // Account for Overflow
        OV <<= 1 ;                     // Multiply the original data by 2
        OV &= 0xfefefefe ;             // Clear the MSBs
        OV ^= OF ;                     // Add in the overflow mask

        // Process bit 6
        T0 ^= OV;                      // Account for bit 6 of constant f4
        T1 ^= OV;                      // Account for bit 6 of constant f2
        T4 ^= OV;                      // Account for bit 6 of constant 5c
        T5 ^= OV;                      // Account for bit 6 of constant fd
        T8 ^= OV;                      // Account for bit 6 of constant 6c
        MSK = OV & 0x80808080 ;        // Create a mask of the MSBs
        OF = (MSK >> 7) * 0x1d ;       // Account for Overflow
        OV <<= 1 ;                     // Multiply the original data by 2
        OV &= 0xfefefefe ;             // Clear the MSBs
        OV ^= OF ;                     // Add in the overflow mask

        // Process bit 7
        T0 ^= OV;                      // Account for bit 7 of constant f4
        T1 ^= OV;                      // Account for bit 7 of constant f2
        T2 ^= OV;                      // Account for bit 7 of constant 89
        T3 ^= OV;                      // Account for bit 7 of constant a6
        T5 ^= OV;                      // Account for bit 7 of constant fd
        T6 ^= OV;                      // Account for bit 7 of constant ad

        // Now process the LFSR
        P0 = P1 ^ T0 ;
        P1 = P2 ^ T1 ;
        P2 = P3 ^ T2 ;
        P3 = P4 ^ T3 ;
        P4 = P5 ^ T4 ;
        P5 = P6 ^ T5 ;
        P6 = P7 ^ T6 ;
        P7 = P8 ^ T7 ;
        P8 = P9 ^ T8 ;
        P9 = P10 ^ T9 ;
        P10 = P11 ^ T9 ;
        P11 = P12 ^ T8 ;
        P12 = P13 ^ T7 ;
        P13 = P14 ^ T6 ;
        P14 = P15 ^ T5 ;
        P15 = P16 ^ T4 ;
        P16 = P17 ^ T3 ;
        P17 = P18 ^ T2 ;
        P18 = P19 ^ T1 ;
        P19 = P20 ^ T0 ;
        P20 = OOV ;
    }

    if ( decoder == 0 )
    {
        // Now store computed check symbols back to memory
        loc = CWP[k+0] + i;
        *loc = P0 ;
        loc = CWP[k+1] + i;
        *loc = P1 ;
        loc = CWP[k+2] + i;
        *loc = P2 ;
        loc = CWP[k+3] + i;
        *loc = P3 ;
        loc = CWP[k+4] + i;
        *loc = P4 ;
        loc = CWP[k+5] + i;
        *loc = P5 ;
        loc = CWP[k+6] + i;
        *loc = P6 ;
        loc = CWP[k+7] + i;
        *loc = P7 ;
        loc = CWP[k+8] + i;
        *loc = P8 ;
        loc = CWP[k+9] + i;
        *loc = P9 ;
        loc = CWP[k+10] + i;
        *loc = P10 ;
        loc = CWP[k+11] + i;
        *loc = P11 ;
        loc = CWP[k+12] + i;
        *loc = P12 ;
        loc = CWP[k+13] + i;
        *loc = P13 ;
        loc = CWP[k+14] + i;
        *loc = P14 ;
        loc = CWP[k+15] + i;
        *loc = P15 ;
        loc = CWP[k+16] + i;
        *loc = P16 ;
        loc = CWP[k+17] + i;
        *loc = P17 ;
        loc = CWP[k+18] + i;
        *loc = P18 ;
        loc = CWP[k+19] + i;
        *loc = P19 ;
        loc = CWP[k+20] + i;
        *loc = P20 ;
    }
    else
    {
        // Now check for parity equal to zero
        T0 = P0;
        T0 |= P1;
        T0 |= P2;
        T0 |= P3;
        T0 |= P4;
        T0 |= P5;
        T0 |= P6;
        T0 |= P7;
        T0 |= P8;
        T0 |= P9;
        T0 |= P10;
        T0 |= P11;
        T0 |= P12;
        T0 |= P13;
        T0 |= P14;
        T0 |= P15;
        T0 |= P16;
        T0 |= P17;
        T0 |= P18;
        T0 |= P19;
        T0 |= P20;
        if (T0 != 0)
        {
            unsigned char Sr[21], S[21], errLocs[21], errMags[21] ;
            unsigned char Lambda[21+1], B[21+1], T[21+1], Omega[21+1] ;
            
            if ( T0 & (0xff << 0) )
            {
                atomicAdd(&error_count, 1);
                Sr[0] = (P0>>0) & 0xff ;
                Sr[1] = (P1>>0) & 0xff ;
                Sr[2] = (P2>>0) & 0xff ;
                Sr[3] = (P3>>0) & 0xff ;
                Sr[4] = (P4>>0) & 0xff ;
                Sr[5] = (P5>>0) & 0xff ;
                Sr[6] = (P6>>0) & 0xff ;
                Sr[7] = (P7>>0) & 0xff ;
                Sr[8] = (P8>>0) & 0xff ;
                Sr[9] = (P9>>0) & 0xff ;
                Sr[10] = (P10>>0) & 0xff ;
                Sr[11] = (P11>>0) & 0xff ;
                Sr[12] = (P12>>0) & 0xff ;
                Sr[13] = (P13>>0) & 0xff ;
                Sr[14] = (P14>>0) & 0xff ;
                Sr[15] = (P15>>0) & 0xff ;
                Sr[16] = (P16>>0) & 0xff ;
                Sr[17] = (P17>>0) & 0xff ;
                Sr[18] = (P18>>0) & 0xff ;
                Sr[19] = (P19>>0) & 0xff ;
                Sr[20] = (P20>>0) & 0xff ;
                rs_decode_palindromic_CUDA(Sr, S, 21, k, errLocs, errMags,
                                           0+(i<<2), Lambda, B, T, Omega) ;
            }
            if ( T0 & (0xff << 8) )
            {
                atomicAdd(&error_count, 1);
                Sr[0] = (P0>>8) & 0xff ;
                Sr[1] = (P1>>8) & 0xff ;
                Sr[2] = (P2>>8) & 0xff ;
                Sr[3] = (P3>>8) & 0xff ;
                Sr[4] = (P4>>8) & 0xff ;
                Sr[5] = (P5>>8) & 0xff ;
                Sr[6] = (P6>>8) & 0xff ;
                Sr[7] = (P7>>8) & 0xff ;
                Sr[8] = (P8>>8) & 0xff ;
                Sr[9] = (P9>>8) & 0xff ;
                Sr[10] = (P10>>8) & 0xff ;
                Sr[11] = (P11>>8) & 0xff ;
                Sr[12] = (P12>>8) & 0xff ;
                Sr[13] = (P13>>8) & 0xff ;
                Sr[14] = (P14>>8) & 0xff ;
                Sr[15] = (P15>>8) & 0xff ;
                Sr[16] = (P16>>8) & 0xff ;
                Sr[17] = (P17>>8) & 0xff ;
                Sr[18] = (P18>>8) & 0xff ;
                Sr[19] = (P19>>8) & 0xff ;
                Sr[20] = (P20>>8) & 0xff ;
                rs_decode_palindromic_CUDA(Sr, S, 21, k, errLocs, errMags,
                                           1+(i<<2), Lambda, B, T, Omega) ;
            }
            if ( T0 & (0xff << 16) )
            {
                atomicAdd(&error_count, 1);
                Sr[0] = (P0>>16) & 0xff ;
                Sr[1] = (P1>>16) & 0xff ;
                Sr[2] = (P2>>16) & 0xff ;
                Sr[3] = (P3>>16) & 0xff ;
                Sr[4] = (P4>>16) & 0xff ;
                Sr[5] = (P5>>16) & 0xff ;
                Sr[6] = (P6>>16) & 0xff ;
                Sr[7] = (P7>>16) & 0xff ;
                Sr[8] = (P8>>16) & 0xff ;
                Sr[9] = (P9>>16) & 0xff ;
                Sr[10] = (P10>>16) & 0xff ;
                Sr[11] = (P11>>16) & 0xff ;
                Sr[12] = (P12>>16) & 0xff ;
                Sr[13] = (P13>>16) & 0xff ;
                Sr[14] = (P14>>16) & 0xff ;
                Sr[15] = (P15>>16) & 0xff ;
                Sr[16] = (P16>>16) & 0xff ;
                Sr[17] = (P17>>16) & 0xff ;
                Sr[18] = (P18>>16) & 0xff ;
                Sr[19] = (P19>>16) & 0xff ;
                Sr[20] = (P20>>16) & 0xff ;
                rs_decode_palindromic_CUDA(Sr, S, 21, k, errLocs, errMags,
                                           2+(i<<2), Lambda, B, T, Omega) ;
            }
            if ( T0 & (0xff << 24) )
            {
                atomicAdd(&error_count, 1);
                Sr[0] = (P0>>24) & 0xff ;
                Sr[1] = (P1>>24) & 0xff ;
                Sr[2] = (P2>>24) & 0xff ;
                Sr[3] = (P3>>24) & 0xff ;
                Sr[4] = (P4>>24) & 0xff ;
                Sr[5] = (P5>>24) & 0xff ;
                Sr[6] = (P6>>24) & 0xff ;
                Sr[7] = (P7>>24) & 0xff ;
                Sr[8] = (P8>>24) & 0xff ;
                Sr[9] = (P9>>24) & 0xff ;
                Sr[10] = (P10>>24) & 0xff ;
                Sr[11] = (P11>>24) & 0xff ;
                Sr[12] = (P12>>24) & 0xff ;
                Sr[13] = (P13>>24) & 0xff ;
                Sr[14] = (P14>>24) & 0xff ;
                Sr[15] = (P15>>24) & 0xff ;
                Sr[16] = (P16>>24) & 0xff ;
                Sr[17] = (P17>>24) & 0xff ;
                Sr[18] = (P18>>24) & 0xff ;
                Sr[19] = (P19>>24) & 0xff ;
                Sr[20] = (P20>>24) & 0xff ;
                rs_decode_palindromic_CUDA(Sr, S, 21, k, errLocs, errMags,
                                           3+(i<<2), Lambda, B, T, Omega) ;
            }
        }
    }
}

// ----------------------------------------------
// Parallel LFSR Sequencer for P = 22 Codewords
// ----------------------------------------------
__global__ void ParallelLFSRSequencer22_CUDA(int k, int decoder)
{
    // Find out place in the grid of threads
    int i = blockIdx.x * blockDim.x + threadIdx.x;

    // Pointer to develop original data address
    unsigned int* loc ;

    // Create the parity values, initialize to zero
    unsigned int P0 = 0 ;
    unsigned int P1 = 0 ;
    unsigned int P2 = 0 ;
    unsigned int P3 = 0 ;
    unsigned int P4 = 0 ;
    unsigned int P5 = 0 ;
    unsigned int P6 = 0 ;
    unsigned int P7 = 0 ;
    unsigned int P8 = 0 ;
    unsigned int P9 = 0 ;
    unsigned int P10 = 0 ;
    unsigned int P11 = 0 ;
    unsigned int P12 = 0 ;
    unsigned int P13 = 0 ;
    unsigned int P14 = 0 ;
    unsigned int P15 = 0 ;
    unsigned int P16 = 0 ;
    unsigned int P17 = 0 ;
    unsigned int P18 = 0 ;
    unsigned int P19 = 0 ;
    unsigned int P20 = 0 ;
    unsigned int P21 = 0 ;

    // Temporary values for self reciprocol polynomial
    unsigned int T0;
    unsigned int T1;
    unsigned int T2;
    unsigned int T3;
    unsigned int T4;
    unsigned int T5;
    unsigned int T6;
    unsigned int T7;
    unsigned int T8;
    unsigned int T9;
    unsigned int T10;

    // Overflow value for LFSR and carry mask
    unsigned int OV, MSK, OF, OOV ;

    // Adjust K if we are decoding
    k += decoder ? 22 : 0 ;

    // Loop through each symbol
    for ( int sym = 0 ; sym < k ; sym ++ )
    {
        loc = CWP[sym] + i;
        OV = *loc ^ P0 ;
        OOV = OV ;

        // Process bit 0
        T0  = OV;                      // Account for bit 0 of constant 65
        T1  = OV;                      // Account for bit 0 of constant 55
        T6  = OV;                      // Account for bit 0 of constant 99
        T7  = OV;                      // Account for bit 0 of constant 3
        T10  = OV;                      // Account for bit 0 of constant 79
        MSK = OV & 0x80808080 ;        // Create a mask of the MSBs
        OF = (MSK >> 7) * 0x1d ;       // Account for Overflow
        OV <<= 1 ;                     // Multiply the original data by 2
        OV &= 0xfefefefe ;             // Clear the MSBs
        OV ^= OF ;                     // Add in the overflow mask

        // Process bit 1
        T2  = OV;                      // Account for bit 1 of constant fe
        T5  = OV;                      // Account for bit 1 of constant 7e
        T7 ^= OV;                      // Account for bit 1 of constant 3
        T8  = OV;                      // Account for bit 1 of constant 8e
        MSK = OV & 0x80808080 ;        // Create a mask of the MSBs
        OF = (MSK >> 7) * 0x1d ;       // Account for Overflow
        OV <<= 1 ;                     // Multiply the original data by 2
        OV &= 0xfefefefe ;             // Clear the MSBs
        OV ^= OF ;                     // Add in the overflow mask

        // Process bit 2
        T0 ^= OV;                      // Account for bit 2 of constant 65
        T1 ^= OV;                      // Account for bit 2 of constant 55
        T2 ^= OV;                      // Account for bit 2 of constant fe
        T3  = OV;                      // Account for bit 2 of constant 1c
        T4  = OV;                      // Account for bit 2 of constant fc
        T5 ^= OV;                      // Account for bit 2 of constant 7e
        T8 ^= OV;                      // Account for bit 2 of constant 8e
        MSK = OV & 0x80808080 ;        // Create a mask of the MSBs
        OF = (MSK >> 7) * 0x1d ;       // Account for Overflow
        OV <<= 1 ;                     // Multiply the original data by 2
        OV &= 0xfefefefe ;             // Clear the MSBs
        OV ^= OF ;                     // Add in the overflow mask

        // Process bit 3
        T2 ^= OV;                      // Account for bit 3 of constant fe
        T3 ^= OV;                      // Account for bit 3 of constant 1c
        T4 ^= OV;                      // Account for bit 3 of constant fc
        T5 ^= OV;                      // Account for bit 3 of constant 7e
        T6 ^= OV;                      // Account for bit 3 of constant 99
        T8 ^= OV;                      // Account for bit 3 of constant 8e
        T10 ^= OV;                      // Account for bit 3 of constant 79
        MSK = OV & 0x80808080 ;        // Create a mask of the MSBs
        OF = (MSK >> 7) * 0x1d ;       // Account for Overflow
        OV <<= 1 ;                     // Multiply the original data by 2
        OV &= 0xfefefefe ;             // Clear the MSBs
        OV ^= OF ;                     // Add in the overflow mask

        // Process bit 4
        T1 ^= OV;                      // Account for bit 4 of constant 55
        T2 ^= OV;                      // Account for bit 4 of constant fe
        T3 ^= OV;                      // Account for bit 4 of constant 1c
        T4 ^= OV;                      // Account for bit 4 of constant fc
        T5 ^= OV;                      // Account for bit 4 of constant 7e
        T6 ^= OV;                      // Account for bit 4 of constant 99
        T10 ^= OV;                      // Account for bit 4 of constant 79
        MSK = OV & 0x80808080 ;        // Create a mask of the MSBs
        OF = (MSK >> 7) * 0x1d ;       // Account for Overflow
        OV <<= 1 ;                     // Multiply the original data by 2
        OV &= 0xfefefefe ;             // Clear the MSBs
        OV ^= OF ;                     // Add in the overflow mask

        // Process bit 5
        T0 ^= OV;                      // Account for bit 5 of constant 65
        T2 ^= OV;                      // Account for bit 5 of constant fe
        T4 ^= OV;                      // Account for bit 5 of constant fc
        T5 ^= OV;                      // Account for bit 5 of constant 7e
        T9  = OV;                      // Account for bit 5 of constant e0
        T10 ^= OV;                      // Account for bit 5 of constant 79
        MSK = OV & 0x80808080 ;        // Create a mask of the MSBs
        OF = (MSK >> 7) * 0x1d ;       // Account for Overflow
        OV <<= 1 ;                     // Multiply the original data by 2
        OV &= 0xfefefefe ;             // Clear the MSBs
        OV ^= OF ;                     // Add in the overflow mask

        // Process bit 6
        T0 ^= OV;                      // Account for bit 6 of constant 65
        T1 ^= OV;                      // Account for bit 6 of constant 55
        T2 ^= OV;                      // Account for bit 6 of constant fe
        T4 ^= OV;                      // Account for bit 6 of constant fc
        T5 ^= OV;                      // Account for bit 6 of constant 7e
        T9 ^= OV;                      // Account for bit 6 of constant e0
        T10 ^= OV;                      // Account for bit 6 of constant 79
        MSK = OV & 0x80808080 ;        // Create a mask of the MSBs
        OF = (MSK >> 7) * 0x1d ;       // Account for Overflow
        OV <<= 1 ;                     // Multiply the original data by 2
        OV &= 0xfefefefe ;             // Clear the MSBs
        OV ^= OF ;                     // Add in the overflow mask

        // Process bit 7
        T2 ^= OV;                      // Account for bit 7 of constant fe
        T4 ^= OV;                      // Account for bit 7 of constant fc
        T6 ^= OV;                      // Account for bit 7 of constant 99
        T8 ^= OV;                      // Account for bit 7 of constant 8e
        T9 ^= OV;                      // Account for bit 7 of constant e0

        // Now process the LFSR
        P0 = P1 ^ T0 ;
        P1 = P2 ^ T1 ;
        P2 = P3 ^ T2 ;
        P3 = P4 ^ T3 ;
        P4 = P5 ^ T4 ;
        P5 = P6 ^ T5 ;
        P6 = P7 ^ T6 ;
        P7 = P8 ^ T7 ;
        P8 = P9 ^ T8 ;
        P9 = P10 ^ T9 ;
        P10 = P11 ^ T10 ;
        P11 = P12 ^ T9 ;
        P12 = P13 ^ T8 ;
        P13 = P14 ^ T7 ;
        P14 = P15 ^ T6 ;
        P15 = P16 ^ T5 ;
        P16 = P17 ^ T4 ;
        P17 = P18 ^ T3 ;
        P18 = P19 ^ T2 ;
        P19 = P20 ^ T1 ;
        P20 = P21 ^ T0 ;
        P21 = OOV ;
    }

    if ( decoder == 0 )
    {
        // Now store computed check symbols back to memory
        loc = CWP[k+0] + i;
        *loc = P0 ;
        loc = CWP[k+1] + i;
        *loc = P1 ;
        loc = CWP[k+2] + i;
        *loc = P2 ;
        loc = CWP[k+3] + i;
        *loc = P3 ;
        loc = CWP[k+4] + i;
        *loc = P4 ;
        loc = CWP[k+5] + i;
        *loc = P5 ;
        loc = CWP[k+6] + i;
        *loc = P6 ;
        loc = CWP[k+7] + i;
        *loc = P7 ;
        loc = CWP[k+8] + i;
        *loc = P8 ;
        loc = CWP[k+9] + i;
        *loc = P9 ;
        loc = CWP[k+10] + i;
        *loc = P10 ;
        loc = CWP[k+11] + i;
        *loc = P11 ;
        loc = CWP[k+12] + i;
        *loc = P12 ;
        loc = CWP[k+13] + i;
        *loc = P13 ;
        loc = CWP[k+14] + i;
        *loc = P14 ;
        loc = CWP[k+15] + i;
        *loc = P15 ;
        loc = CWP[k+16] + i;
        *loc = P16 ;
        loc = CWP[k+17] + i;
        *loc = P17 ;
        loc = CWP[k+18] + i;
        *loc = P18 ;
        loc = CWP[k+19] + i;
        *loc = P19 ;
        loc = CWP[k+20] + i;
        *loc = P20 ;
        loc = CWP[k+21] + i;
        *loc = P21 ;
    }
    else
    {
        // Now check for parity equal to zero
        T0 = P0;
        T0 |= P1;
        T0 |= P2;
        T0 |= P3;
        T0 |= P4;
        T0 |= P5;
        T0 |= P6;
        T0 |= P7;
        T0 |= P8;
        T0 |= P9;
        T0 |= P10;
        T0 |= P11;
        T0 |= P12;
        T0 |= P13;
        T0 |= P14;
        T0 |= P15;
        T0 |= P16;
        T0 |= P17;
        T0 |= P18;
        T0 |= P19;
        T0 |= P20;
        T0 |= P21;
        if (T0 != 0)
        {
            unsigned char Sr[22], S[22], errLocs[22], errMags[22] ;
            unsigned char Lambda[22+1], B[22+1], T[22+1], Omega[22+1] ;
            
            if ( T0 & (0xff << 0) )
            {
                atomicAdd(&error_count, 1);
                Sr[0] = (P0>>0) & 0xff ;
                Sr[1] = (P1>>0) & 0xff ;
                Sr[2] = (P2>>0) & 0xff ;
                Sr[3] = (P3>>0) & 0xff ;
                Sr[4] = (P4>>0) & 0xff ;
                Sr[5] = (P5>>0) & 0xff ;
                Sr[6] = (P6>>0) & 0xff ;
                Sr[7] = (P7>>0) & 0xff ;
                Sr[8] = (P8>>0) & 0xff ;
                Sr[9] = (P9>>0) & 0xff ;
                Sr[10] = (P10>>0) & 0xff ;
                Sr[11] = (P11>>0) & 0xff ;
                Sr[12] = (P12>>0) & 0xff ;
                Sr[13] = (P13>>0) & 0xff ;
                Sr[14] = (P14>>0) & 0xff ;
                Sr[15] = (P15>>0) & 0xff ;
                Sr[16] = (P16>>0) & 0xff ;
                Sr[17] = (P17>>0) & 0xff ;
                Sr[18] = (P18>>0) & 0xff ;
                Sr[19] = (P19>>0) & 0xff ;
                Sr[20] = (P20>>0) & 0xff ;
                Sr[21] = (P21>>0) & 0xff ;
                rs_decode_palindromic_CUDA(Sr, S, 22, k, errLocs, errMags,
                                           0+(i<<2), Lambda, B, T, Omega) ;
            }
            if ( T0 & (0xff << 8) )
            {
                atomicAdd(&error_count, 1);
                Sr[0] = (P0>>8) & 0xff ;
                Sr[1] = (P1>>8) & 0xff ;
                Sr[2] = (P2>>8) & 0xff ;
                Sr[3] = (P3>>8) & 0xff ;
                Sr[4] = (P4>>8) & 0xff ;
                Sr[5] = (P5>>8) & 0xff ;
                Sr[6] = (P6>>8) & 0xff ;
                Sr[7] = (P7>>8) & 0xff ;
                Sr[8] = (P8>>8) & 0xff ;
                Sr[9] = (P9>>8) & 0xff ;
                Sr[10] = (P10>>8) & 0xff ;
                Sr[11] = (P11>>8) & 0xff ;
                Sr[12] = (P12>>8) & 0xff ;
                Sr[13] = (P13>>8) & 0xff ;
                Sr[14] = (P14>>8) & 0xff ;
                Sr[15] = (P15>>8) & 0xff ;
                Sr[16] = (P16>>8) & 0xff ;
                Sr[17] = (P17>>8) & 0xff ;
                Sr[18] = (P18>>8) & 0xff ;
                Sr[19] = (P19>>8) & 0xff ;
                Sr[20] = (P20>>8) & 0xff ;
                Sr[21] = (P21>>8) & 0xff ;
                rs_decode_palindromic_CUDA(Sr, S, 22, k, errLocs, errMags,
                                           1+(i<<2), Lambda, B, T, Omega) ;
            }
            if ( T0 & (0xff << 16) )
            {
                atomicAdd(&error_count, 1);
                Sr[0] = (P0>>16) & 0xff ;
                Sr[1] = (P1>>16) & 0xff ;
                Sr[2] = (P2>>16) & 0xff ;
                Sr[3] = (P3>>16) & 0xff ;
                Sr[4] = (P4>>16) & 0xff ;
                Sr[5] = (P5>>16) & 0xff ;
                Sr[6] = (P6>>16) & 0xff ;
                Sr[7] = (P7>>16) & 0xff ;
                Sr[8] = (P8>>16) & 0xff ;
                Sr[9] = (P9>>16) & 0xff ;
                Sr[10] = (P10>>16) & 0xff ;
                Sr[11] = (P11>>16) & 0xff ;
                Sr[12] = (P12>>16) & 0xff ;
                Sr[13] = (P13>>16) & 0xff ;
                Sr[14] = (P14>>16) & 0xff ;
                Sr[15] = (P15>>16) & 0xff ;
                Sr[16] = (P16>>16) & 0xff ;
                Sr[17] = (P17>>16) & 0xff ;
                Sr[18] = (P18>>16) & 0xff ;
                Sr[19] = (P19>>16) & 0xff ;
                Sr[20] = (P20>>16) & 0xff ;
                Sr[21] = (P21>>16) & 0xff ;
                rs_decode_palindromic_CUDA(Sr, S, 22, k, errLocs, errMags,
                                           2+(i<<2), Lambda, B, T, Omega) ;
            }
            if ( T0 & (0xff << 24) )
            {
                atomicAdd(&error_count, 1);
                Sr[0] = (P0>>24) & 0xff ;
                Sr[1] = (P1>>24) & 0xff ;
                Sr[2] = (P2>>24) & 0xff ;
                Sr[3] = (P3>>24) & 0xff ;
                Sr[4] = (P4>>24) & 0xff ;
                Sr[5] = (P5>>24) & 0xff ;
                Sr[6] = (P6>>24) & 0xff ;
                Sr[7] = (P7>>24) & 0xff ;
                Sr[8] = (P8>>24) & 0xff ;
                Sr[9] = (P9>>24) & 0xff ;
                Sr[10] = (P10>>24) & 0xff ;
                Sr[11] = (P11>>24) & 0xff ;
                Sr[12] = (P12>>24) & 0xff ;
                Sr[13] = (P13>>24) & 0xff ;
                Sr[14] = (P14>>24) & 0xff ;
                Sr[15] = (P15>>24) & 0xff ;
                Sr[16] = (P16>>24) & 0xff ;
                Sr[17] = (P17>>24) & 0xff ;
                Sr[18] = (P18>>24) & 0xff ;
                Sr[19] = (P19>>24) & 0xff ;
                Sr[20] = (P20>>24) & 0xff ;
                Sr[21] = (P21>>24) & 0xff ;
                rs_decode_palindromic_CUDA(Sr, S, 22, k, errLocs, errMags,
                                           3+(i<<2), Lambda, B, T, Omega) ;
            }
        }
    }
}

// ----------------------------------------------
// Parallel LFSR Sequencer for P = 23 Codewords
// ----------------------------------------------
__global__ void ParallelLFSRSequencer23_CUDA(int k, int decoder)
{
    // Find out place in the grid of threads
    int i = blockIdx.x * blockDim.x + threadIdx.x;

    // Pointer to develop original data address
    unsigned int* loc ;

    // Create the parity values, initialize to zero
    unsigned int P0 = 0 ;
    unsigned int P1 = 0 ;
    unsigned int P2 = 0 ;
    unsigned int P3 = 0 ;
    unsigned int P4 = 0 ;
    unsigned int P5 = 0 ;
    unsigned int P6 = 0 ;
    unsigned int P7 = 0 ;
    unsigned int P8 = 0 ;
    unsigned int P9 = 0 ;
    unsigned int P10 = 0 ;
    unsigned int P11 = 0 ;
    unsigned int P12 = 0 ;
    unsigned int P13 = 0 ;
    unsigned int P14 = 0 ;
    unsigned int P15 = 0 ;
    unsigned int P16 = 0 ;
    unsigned int P17 = 0 ;
    unsigned int P18 = 0 ;
    unsigned int P19 = 0 ;
    unsigned int P20 = 0 ;
    unsigned int P21 = 0 ;
    unsigned int P22 = 0 ;

    // Temporary values for self reciprocol polynomial
    unsigned int T0;
    unsigned int T1;
    unsigned int T2;
    unsigned int T3;
    unsigned int T4;
    unsigned int T5;
    unsigned int T6;
    unsigned int T7;
    unsigned int T8;
    unsigned int T9;
    unsigned int T10;

    // Overflow value for LFSR and carry mask
    unsigned int OV, MSK, OF, OOV ;

    // Adjust K if we are decoding
    k += decoder ? 23 : 0 ;

    // Loop through each symbol
    for ( int sym = 0 ; sym < k ; sym ++ )
    {
        loc = CWP[sym] + i;
        OV = *loc ^ P0 ;
        OOV = OV ;

        // Process bit 0
        T1  = OV;                      // Account for bit 0 of constant fd
        T2  = OV;                      // Account for bit 0 of constant 1f
        T3  = OV;                      // Account for bit 0 of constant 23
        T6  = OV;                      // Account for bit 0 of constant 7d
        T7  = OV;                      // Account for bit 0 of constant 95
        T8  = OV;                      // Account for bit 0 of constant 71
        MSK = OV & 0x80808080 ;        // Create a mask of the MSBs
        OF = (MSK >> 7) * 0x1d ;       // Account for Overflow
        OV <<= 1 ;                     // Multiply the original data by 2
        OV &= 0xfefefefe ;             // Clear the MSBs
        OV ^= OF ;                     // Add in the overflow mask

        // Process bit 1
        T0  = OV;                      // Account for bit 1 of constant e6
        T2 ^= OV;                      // Account for bit 1 of constant 1f
        T3 ^= OV;                      // Account for bit 1 of constant 23
        T4  = OV;                      // Account for bit 1 of constant 36
        T5  = OV;                      // Account for bit 1 of constant 4a
        T9  = OV;                      // Account for bit 1 of constant 6a
        T10  = OV;                      // Account for bit 1 of constant 52
        MSK = OV & 0x80808080 ;        // Create a mask of the MSBs
        OF = (MSK >> 7) * 0x1d ;       // Account for Overflow
        OV <<= 1 ;                     // Multiply the original data by 2
        OV &= 0xfefefefe ;             // Clear the MSBs
        OV ^= OF ;                     // Add in the overflow mask

        // Process bit 2
        T0 ^= OV;                      // Account for bit 2 of constant e6
        T1 ^= OV;                      // Account for bit 2 of constant fd
        T2 ^= OV;                      // Account for bit 2 of constant 1f
        T4 ^= OV;                      // Account for bit 2 of constant 36
        T6 ^= OV;                      // Account for bit 2 of constant 7d
        T7 ^= OV;                      // Account for bit 2 of constant 95
        MSK = OV & 0x80808080 ;        // Create a mask of the MSBs
        OF = (MSK >> 7) * 0x1d ;       // Account for Overflow
        OV <<= 1 ;                     // Multiply the original data by 2
        OV &= 0xfefefefe ;             // Clear the MSBs
        OV ^= OF ;                     // Add in the overflow mask

        // Process bit 3
        T1 ^= OV;                      // Account for bit 3 of constant fd
        T2 ^= OV;                      // Account for bit 3 of constant 1f
        T5 ^= OV;                      // Account for bit 3 of constant 4a
        T6 ^= OV;                      // Account for bit 3 of constant 7d
        T9 ^= OV;                      // Account for bit 3 of constant 6a
        MSK = OV & 0x80808080 ;        // Create a mask of the MSBs
        OF = (MSK >> 7) * 0x1d ;       // Account for Overflow
        OV <<= 1 ;                     // Multiply the original data by 2
        OV &= 0xfefefefe ;             // Clear the MSBs
        OV ^= OF ;                     // Add in the overflow mask

        // Process bit 4
        T1 ^= OV;                      // Account for bit 4 of constant fd
        T2 ^= OV;                      // Account for bit 4 of constant 1f
        T4 ^= OV;                      // Account for bit 4 of constant 36
        T6 ^= OV;                      // Account for bit 4 of constant 7d
        T7 ^= OV;                      // Account for bit 4 of constant 95
        T8 ^= OV;                      // Account for bit 4 of constant 71
        T10 ^= OV;                      // Account for bit 4 of constant 52
        MSK = OV & 0x80808080 ;        // Create a mask of the MSBs
        OF = (MSK >> 7) * 0x1d ;       // Account for Overflow
        OV <<= 1 ;                     // Multiply the original data by 2
        OV &= 0xfefefefe ;             // Clear the MSBs
        OV ^= OF ;                     // Add in the overflow mask

        // Process bit 5
        T0 ^= OV;                      // Account for bit 5 of constant e6
        T1 ^= OV;                      // Account for bit 5 of constant fd
        T3 ^= OV;                      // Account for bit 5 of constant 23
        T4 ^= OV;                      // Account for bit 5 of constant 36
        T6 ^= OV;                      // Account for bit 5 of constant 7d
        T8 ^= OV;                      // Account for bit 5 of constant 71
        T9 ^= OV;                      // Account for bit 5 of constant 6a
        MSK = OV & 0x80808080 ;        // Create a mask of the MSBs
        OF = (MSK >> 7) * 0x1d ;       // Account for Overflow
        OV <<= 1 ;                     // Multiply the original data by 2
        OV &= 0xfefefefe ;             // Clear the MSBs
        OV ^= OF ;                     // Add in the overflow mask

        // Process bit 6
        T0 ^= OV;                      // Account for bit 6 of constant e6
        T1 ^= OV;                      // Account for bit 6 of constant fd
        T5 ^= OV;                      // Account for bit 6 of constant 4a
        T6 ^= OV;                      // Account for bit 6 of constant 7d
        T8 ^= OV;                      // Account for bit 6 of constant 71
        T9 ^= OV;                      // Account for bit 6 of constant 6a
        T10 ^= OV;                      // Account for bit 6 of constant 52
        MSK = OV & 0x80808080 ;        // Create a mask of the MSBs
        OF = (MSK >> 7) * 0x1d ;       // Account for Overflow
        OV <<= 1 ;                     // Multiply the original data by 2
        OV &= 0xfefefefe ;             // Clear the MSBs
        OV ^= OF ;                     // Add in the overflow mask

        // Process bit 7
        T0 ^= OV;                      // Account for bit 7 of constant e6
        T1 ^= OV;                      // Account for bit 7 of constant fd
        T7 ^= OV;                      // Account for bit 7 of constant 95

        // Now process the LFSR
        P0 = P1 ^ T0 ;
        P1 = P2 ^ T1 ;
        P2 = P3 ^ T2 ;
        P3 = P4 ^ T3 ;
        P4 = P5 ^ T4 ;
        P5 = P6 ^ T5 ;
        P6 = P7 ^ T6 ;
        P7 = P8 ^ T7 ;
        P8 = P9 ^ T8 ;
        P9 = P10 ^ T9 ;
        P10 = P11 ^ T10 ;
        P11 = P12 ^ T10 ;
        P12 = P13 ^ T9 ;
        P13 = P14 ^ T8 ;
        P14 = P15 ^ T7 ;
        P15 = P16 ^ T6 ;
        P16 = P17 ^ T5 ;
        P17 = P18 ^ T4 ;
        P18 = P19 ^ T3 ;
        P19 = P20 ^ T2 ;
        P20 = P21 ^ T1 ;
        P21 = P22 ^ T0 ;
        P22 = OOV ;
    }

    if ( decoder == 0 )
    {
        // Now store computed check symbols back to memory
        loc = CWP[k+0] + i;
        *loc = P0 ;
        loc = CWP[k+1] + i;
        *loc = P1 ;
        loc = CWP[k+2] + i;
        *loc = P2 ;
        loc = CWP[k+3] + i;
        *loc = P3 ;
        loc = CWP[k+4] + i;
        *loc = P4 ;
        loc = CWP[k+5] + i;
        *loc = P5 ;
        loc = CWP[k+6] + i;
        *loc = P6 ;
        loc = CWP[k+7] + i;
        *loc = P7 ;
        loc = CWP[k+8] + i;
        *loc = P8 ;
        loc = CWP[k+9] + i;
        *loc = P9 ;
        loc = CWP[k+10] + i;
        *loc = P10 ;
        loc = CWP[k+11] + i;
        *loc = P11 ;
        loc = CWP[k+12] + i;
        *loc = P12 ;
        loc = CWP[k+13] + i;
        *loc = P13 ;
        loc = CWP[k+14] + i;
        *loc = P14 ;
        loc = CWP[k+15] + i;
        *loc = P15 ;
        loc = CWP[k+16] + i;
        *loc = P16 ;
        loc = CWP[k+17] + i;
        *loc = P17 ;
        loc = CWP[k+18] + i;
        *loc = P18 ;
        loc = CWP[k+19] + i;
        *loc = P19 ;
        loc = CWP[k+20] + i;
        *loc = P20 ;
        loc = CWP[k+21] + i;
        *loc = P21 ;
        loc = CWP[k+22] + i;
        *loc = P22 ;
    }
    else
    {
        // Now check for parity equal to zero
        T0 = P0;
        T0 |= P1;
        T0 |= P2;
        T0 |= P3;
        T0 |= P4;
        T0 |= P5;
        T0 |= P6;
        T0 |= P7;
        T0 |= P8;
        T0 |= P9;
        T0 |= P10;
        T0 |= P11;
        T0 |= P12;
        T0 |= P13;
        T0 |= P14;
        T0 |= P15;
        T0 |= P16;
        T0 |= P17;
        T0 |= P18;
        T0 |= P19;
        T0 |= P20;
        T0 |= P21;
        T0 |= P22;
        if (T0 != 0)
        {
            unsigned char Sr[23], S[23], errLocs[23], errMags[23] ;
            unsigned char Lambda[23+1], B[23+1], T[23+1], Omega[23+1] ;
            
            if ( T0 & (0xff << 0) )
            {
                atomicAdd(&error_count, 1);
                Sr[0] = (P0>>0) & 0xff ;
                Sr[1] = (P1>>0) & 0xff ;
                Sr[2] = (P2>>0) & 0xff ;
                Sr[3] = (P3>>0) & 0xff ;
                Sr[4] = (P4>>0) & 0xff ;
                Sr[5] = (P5>>0) & 0xff ;
                Sr[6] = (P6>>0) & 0xff ;
                Sr[7] = (P7>>0) & 0xff ;
                Sr[8] = (P8>>0) & 0xff ;
                Sr[9] = (P9>>0) & 0xff ;
                Sr[10] = (P10>>0) & 0xff ;
                Sr[11] = (P11>>0) & 0xff ;
                Sr[12] = (P12>>0) & 0xff ;
                Sr[13] = (P13>>0) & 0xff ;
                Sr[14] = (P14>>0) & 0xff ;
                Sr[15] = (P15>>0) & 0xff ;
                Sr[16] = (P16>>0) & 0xff ;
                Sr[17] = (P17>>0) & 0xff ;
                Sr[18] = (P18>>0) & 0xff ;
                Sr[19] = (P19>>0) & 0xff ;
                Sr[20] = (P20>>0) & 0xff ;
                Sr[21] = (P21>>0) & 0xff ;
                Sr[22] = (P22>>0) & 0xff ;
                rs_decode_palindromic_CUDA(Sr, S, 23, k, errLocs, errMags,
                                           0+(i<<2), Lambda, B, T, Omega) ;
            }
            if ( T0 & (0xff << 8) )
            {
                atomicAdd(&error_count, 1);
                Sr[0] = (P0>>8) & 0xff ;
                Sr[1] = (P1>>8) & 0xff ;
                Sr[2] = (P2>>8) & 0xff ;
                Sr[3] = (P3>>8) & 0xff ;
                Sr[4] = (P4>>8) & 0xff ;
                Sr[5] = (P5>>8) & 0xff ;
                Sr[6] = (P6>>8) & 0xff ;
                Sr[7] = (P7>>8) & 0xff ;
                Sr[8] = (P8>>8) & 0xff ;
                Sr[9] = (P9>>8) & 0xff ;
                Sr[10] = (P10>>8) & 0xff ;
                Sr[11] = (P11>>8) & 0xff ;
                Sr[12] = (P12>>8) & 0xff ;
                Sr[13] = (P13>>8) & 0xff ;
                Sr[14] = (P14>>8) & 0xff ;
                Sr[15] = (P15>>8) & 0xff ;
                Sr[16] = (P16>>8) & 0xff ;
                Sr[17] = (P17>>8) & 0xff ;
                Sr[18] = (P18>>8) & 0xff ;
                Sr[19] = (P19>>8) & 0xff ;
                Sr[20] = (P20>>8) & 0xff ;
                Sr[21] = (P21>>8) & 0xff ;
                Sr[22] = (P22>>8) & 0xff ;
                rs_decode_palindromic_CUDA(Sr, S, 23, k, errLocs, errMags,
                                           1+(i<<2), Lambda, B, T, Omega) ;
            }
            if ( T0 & (0xff << 16) )
            {
                atomicAdd(&error_count, 1);
                Sr[0] = (P0>>16) & 0xff ;
                Sr[1] = (P1>>16) & 0xff ;
                Sr[2] = (P2>>16) & 0xff ;
                Sr[3] = (P3>>16) & 0xff ;
                Sr[4] = (P4>>16) & 0xff ;
                Sr[5] = (P5>>16) & 0xff ;
                Sr[6] = (P6>>16) & 0xff ;
                Sr[7] = (P7>>16) & 0xff ;
                Sr[8] = (P8>>16) & 0xff ;
                Sr[9] = (P9>>16) & 0xff ;
                Sr[10] = (P10>>16) & 0xff ;
                Sr[11] = (P11>>16) & 0xff ;
                Sr[12] = (P12>>16) & 0xff ;
                Sr[13] = (P13>>16) & 0xff ;
                Sr[14] = (P14>>16) & 0xff ;
                Sr[15] = (P15>>16) & 0xff ;
                Sr[16] = (P16>>16) & 0xff ;
                Sr[17] = (P17>>16) & 0xff ;
                Sr[18] = (P18>>16) & 0xff ;
                Sr[19] = (P19>>16) & 0xff ;
                Sr[20] = (P20>>16) & 0xff ;
                Sr[21] = (P21>>16) & 0xff ;
                Sr[22] = (P22>>16) & 0xff ;
                rs_decode_palindromic_CUDA(Sr, S, 23, k, errLocs, errMags,
                                           2+(i<<2), Lambda, B, T, Omega) ;
            }
            if ( T0 & (0xff << 24) )
            {
                atomicAdd(&error_count, 1);
                Sr[0] = (P0>>24) & 0xff ;
                Sr[1] = (P1>>24) & 0xff ;
                Sr[2] = (P2>>24) & 0xff ;
                Sr[3] = (P3>>24) & 0xff ;
                Sr[4] = (P4>>24) & 0xff ;
                Sr[5] = (P5>>24) & 0xff ;
                Sr[6] = (P6>>24) & 0xff ;
                Sr[7] = (P7>>24) & 0xff ;
                Sr[8] = (P8>>24) & 0xff ;
                Sr[9] = (P9>>24) & 0xff ;
                Sr[10] = (P10>>24) & 0xff ;
                Sr[11] = (P11>>24) & 0xff ;
                Sr[12] = (P12>>24) & 0xff ;
                Sr[13] = (P13>>24) & 0xff ;
                Sr[14] = (P14>>24) & 0xff ;
                Sr[15] = (P15>>24) & 0xff ;
                Sr[16] = (P16>>24) & 0xff ;
                Sr[17] = (P17>>24) & 0xff ;
                Sr[18] = (P18>>24) & 0xff ;
                Sr[19] = (P19>>24) & 0xff ;
                Sr[20] = (P20>>24) & 0xff ;
                Sr[21] = (P21>>24) & 0xff ;
                Sr[22] = (P22>>24) & 0xff ;
                rs_decode_palindromic_CUDA(Sr, S, 23, k, errLocs, errMags,
                                           3+(i<<2), Lambda, B, T, Omega) ;
            }
        }
    }
}

// ----------------------------------------------
// Parallel LFSR Sequencer for P = 24 Codewords
// ----------------------------------------------
__global__ void ParallelLFSRSequencer24_CUDA(int k, int decoder)
{
    // Find out place in the grid of threads
    int i = blockIdx.x * blockDim.x + threadIdx.x;

    // Pointer to develop original data address
    unsigned int* loc ;

    // Create the parity values, initialize to zero
    unsigned int P0 = 0 ;
    unsigned int P1 = 0 ;
    unsigned int P2 = 0 ;
    unsigned int P3 = 0 ;
    unsigned int P4 = 0 ;
    unsigned int P5 = 0 ;
    unsigned int P6 = 0 ;
    unsigned int P7 = 0 ;
    unsigned int P8 = 0 ;
    unsigned int P9 = 0 ;
    unsigned int P10 = 0 ;
    unsigned int P11 = 0 ;
    unsigned int P12 = 0 ;
    unsigned int P13 = 0 ;
    unsigned int P14 = 0 ;
    unsigned int P15 = 0 ;
    unsigned int P16 = 0 ;
    unsigned int P17 = 0 ;
    unsigned int P18 = 0 ;
    unsigned int P19 = 0 ;
    unsigned int P20 = 0 ;
    unsigned int P21 = 0 ;
    unsigned int P22 = 0 ;
    unsigned int P23 = 0 ;

    // Temporary values for self reciprocol polynomial
    unsigned int T0;
    unsigned int T1;
    unsigned int T2;
    unsigned int T3;
    unsigned int T4;
    unsigned int T5;
    unsigned int T6;
    unsigned int T7;
    unsigned int T8;
    unsigned int T9;
    unsigned int T10;
    unsigned int T11;

    // Overflow value for LFSR and carry mask
    unsigned int OV, MSK, OF, OOV ;

    // Adjust K if we are decoding
    k += decoder ? 24 : 0 ;

    // Loop through each symbol
    for ( int sym = 0 ; sym < k ; sym ++ )
    {
        loc = CWP[sym] + i;
        OV = *loc ^ P0 ;
        OOV = OV ;

        // Process bit 0
        T0  = OV;                      // Account for bit 0 of constant df
        T1  = OV;                      // Account for bit 0 of constant 43
        T2  = OV;                      // Account for bit 0 of constant 3d
        T6  = OV;                      // Account for bit 0 of constant 77
        T10  = OV;                      // Account for bit 0 of constant d7
        T11  = OV;                      // Account for bit 0 of constant 13
        MSK = OV & 0x80808080 ;        // Create a mask of the MSBs
        OF = (MSK >> 7) * 0x1d ;       // Account for Overflow
        OV <<= 1 ;                     // Multiply the original data by 2
        OV &= 0xfefefefe ;             // Clear the MSBs
        OV ^= OF ;                     // Add in the overflow mask

        // Process bit 1
        T0 ^= OV;                      // Account for bit 1 of constant df
        T1 ^= OV;                      // Account for bit 1 of constant 43
        T4  = OV;                      // Account for bit 1 of constant 6
        T5  = OV;                      // Account for bit 1 of constant 46
        T6 ^= OV;                      // Account for bit 1 of constant 77
        T7  = OV;                      // Account for bit 1 of constant 4e
        T9  = OV;                      // Account for bit 1 of constant be
        T10 ^= OV;                      // Account for bit 1 of constant d7
        T11 ^= OV;                      // Account for bit 1 of constant 13
        MSK = OV & 0x80808080 ;        // Create a mask of the MSBs
        OF = (MSK >> 7) * 0x1d ;       // Account for Overflow
        OV <<= 1 ;                     // Multiply the original data by 2
        OV &= 0xfefefefe ;             // Clear the MSBs
        OV ^= OF ;                     // Add in the overflow mask

        // Process bit 2
        T0 ^= OV;                      // Account for bit 2 of constant df
        T2 ^= OV;                      // Account for bit 2 of constant 3d
        T3  = OV;                      // Account for bit 2 of constant 4
        T4 ^= OV;                      // Account for bit 2 of constant 6
        T5 ^= OV;                      // Account for bit 2 of constant 46
        T6 ^= OV;                      // Account for bit 2 of constant 77
        T7 ^= OV;                      // Account for bit 2 of constant 4e
        T8  = OV;                      // Account for bit 2 of constant c4
        T9 ^= OV;                      // Account for bit 2 of constant be
        T10 ^= OV;                      // Account for bit 2 of constant d7
        MSK = OV & 0x80808080 ;        // Create a mask of the MSBs
        OF = (MSK >> 7) * 0x1d ;       // Account for Overflow
        OV <<= 1 ;                     // Multiply the original data by 2
        OV &= 0xfefefefe ;             // Clear the MSBs
        OV ^= OF ;                     // Add in the overflow mask

        // Process bit 3
        T0 ^= OV;                      // Account for bit 3 of constant df
        T2 ^= OV;                      // Account for bit 3 of constant 3d
        T7 ^= OV;                      // Account for bit 3 of constant 4e
        T9 ^= OV;                      // Account for bit 3 of constant be
        MSK = OV & 0x80808080 ;        // Create a mask of the MSBs
        OF = (MSK >> 7) * 0x1d ;       // Account for Overflow
        OV <<= 1 ;                     // Multiply the original data by 2
        OV &= 0xfefefefe ;             // Clear the MSBs
        OV ^= OF ;                     // Add in the overflow mask

        // Process bit 4
        T0 ^= OV;                      // Account for bit 4 of constant df
        T2 ^= OV;                      // Account for bit 4 of constant 3d
        T6 ^= OV;                      // Account for bit 4 of constant 77
        T9 ^= OV;                      // Account for bit 4 of constant be
        T10 ^= OV;                      // Account for bit 4 of constant d7
        T11 ^= OV;                      // Account for bit 4 of constant 13
        MSK = OV & 0x80808080 ;        // Create a mask of the MSBs
        OF = (MSK >> 7) * 0x1d ;       // Account for Overflow
        OV <<= 1 ;                     // Multiply the original data by 2
        OV &= 0xfefefefe ;             // Clear the MSBs
        OV ^= OF ;                     // Add in the overflow mask

        // Process bit 5
        T2 ^= OV;                      // Account for bit 5 of constant 3d
        T6 ^= OV;                      // Account for bit 5 of constant 77
        T9 ^= OV;                      // Account for bit 5 of constant be
        MSK = OV & 0x80808080 ;        // Create a mask of the MSBs
        OF = (MSK >> 7) * 0x1d ;       // Account for Overflow
        OV <<= 1 ;                     // Multiply the original data by 2
        OV &= 0xfefefefe ;             // Clear the MSBs
        OV ^= OF ;                     // Add in the overflow mask

        // Process bit 6
        T0 ^= OV;                      // Account for bit 6 of constant df
        T1 ^= OV;                      // Account for bit 6 of constant 43
        T5 ^= OV;                      // Account for bit 6 of constant 46
        T6 ^= OV;                      // Account for bit 6 of constant 77
        T7 ^= OV;                      // Account for bit 6 of constant 4e
        T8 ^= OV;                      // Account for bit 6 of constant c4
        T10 ^= OV;                      // Account for bit 6 of constant d7
        MSK = OV & 0x80808080 ;        // Create a mask of the MSBs
        OF = (MSK >> 7) * 0x1d ;       // Account for Overflow
        OV <<= 1 ;                     // Multiply the original data by 2
        OV &= 0xfefefefe ;             // Clear the MSBs
        OV ^= OF ;                     // Add in the overflow mask

        // Process bit 7
        T0 ^= OV;                      // Account for bit 7 of constant df
        T8 ^= OV;                      // Account for bit 7 of constant c4
        T9 ^= OV;                      // Account for bit 7 of constant be
        T10 ^= OV;                      // Account for bit 7 of constant d7

        // Now process the LFSR
        P0 = P1 ^ T0 ;
        P1 = P2 ^ T1 ;
        P2 = P3 ^ T2 ;
        P3 = P4 ^ T3 ;
        P4 = P5 ^ T4 ;
        P5 = P6 ^ T5 ;
        P6 = P7 ^ T6 ;
        P7 = P8 ^ T7 ;
        P8 = P9 ^ T8 ;
        P9 = P10 ^ T9 ;
        P10 = P11 ^ T10 ;
        P11 = P12 ^ T11 ;
        P12 = P13 ^ T10 ;
        P13 = P14 ^ T9 ;
        P14 = P15 ^ T8 ;
        P15 = P16 ^ T7 ;
        P16 = P17 ^ T6 ;
        P17 = P18 ^ T5 ;
        P18 = P19 ^ T4 ;
        P19 = P20 ^ T3 ;
        P20 = P21 ^ T2 ;
        P21 = P22 ^ T1 ;
        P22 = P23 ^ T0 ;
        P23 = OOV ;
    }

    if ( decoder == 0 )
    {
        // Now store computed check symbols back to memory
        loc = CWP[k+0] + i;
        *loc = P0 ;
        loc = CWP[k+1] + i;
        *loc = P1 ;
        loc = CWP[k+2] + i;
        *loc = P2 ;
        loc = CWP[k+3] + i;
        *loc = P3 ;
        loc = CWP[k+4] + i;
        *loc = P4 ;
        loc = CWP[k+5] + i;
        *loc = P5 ;
        loc = CWP[k+6] + i;
        *loc = P6 ;
        loc = CWP[k+7] + i;
        *loc = P7 ;
        loc = CWP[k+8] + i;
        *loc = P8 ;
        loc = CWP[k+9] + i;
        *loc = P9 ;
        loc = CWP[k+10] + i;
        *loc = P10 ;
        loc = CWP[k+11] + i;
        *loc = P11 ;
        loc = CWP[k+12] + i;
        *loc = P12 ;
        loc = CWP[k+13] + i;
        *loc = P13 ;
        loc = CWP[k+14] + i;
        *loc = P14 ;
        loc = CWP[k+15] + i;
        *loc = P15 ;
        loc = CWP[k+16] + i;
        *loc = P16 ;
        loc = CWP[k+17] + i;
        *loc = P17 ;
        loc = CWP[k+18] + i;
        *loc = P18 ;
        loc = CWP[k+19] + i;
        *loc = P19 ;
        loc = CWP[k+20] + i;
        *loc = P20 ;
        loc = CWP[k+21] + i;
        *loc = P21 ;
        loc = CWP[k+22] + i;
        *loc = P22 ;
        loc = CWP[k+23] + i;
        *loc = P23 ;
    }
    else
    {
        // Now check for parity equal to zero
        T0 = P0;
        T0 |= P1;
        T0 |= P2;
        T0 |= P3;
        T0 |= P4;
        T0 |= P5;
        T0 |= P6;
        T0 |= P7;
        T0 |= P8;
        T0 |= P9;
        T0 |= P10;
        T0 |= P11;
        T0 |= P12;
        T0 |= P13;
        T0 |= P14;
        T0 |= P15;
        T0 |= P16;
        T0 |= P17;
        T0 |= P18;
        T0 |= P19;
        T0 |= P20;
        T0 |= P21;
        T0 |= P22;
        T0 |= P23;
        if (T0 != 0)
        {
            unsigned char Sr[24], S[24], errLocs[24], errMags[24] ;
            unsigned char Lambda[24+1], B[24+1], T[24+1], Omega[24+1] ;
            
            if ( T0 & (0xff << 0) )
            {
                atomicAdd(&error_count, 1);
                Sr[0] = (P0>>0) & 0xff ;
                Sr[1] = (P1>>0) & 0xff ;
                Sr[2] = (P2>>0) & 0xff ;
                Sr[3] = (P3>>0) & 0xff ;
                Sr[4] = (P4>>0) & 0xff ;
                Sr[5] = (P5>>0) & 0xff ;
                Sr[6] = (P6>>0) & 0xff ;
                Sr[7] = (P7>>0) & 0xff ;
                Sr[8] = (P8>>0) & 0xff ;
                Sr[9] = (P9>>0) & 0xff ;
                Sr[10] = (P10>>0) & 0xff ;
                Sr[11] = (P11>>0) & 0xff ;
                Sr[12] = (P12>>0) & 0xff ;
                Sr[13] = (P13>>0) & 0xff ;
                Sr[14] = (P14>>0) & 0xff ;
                Sr[15] = (P15>>0) & 0xff ;
                Sr[16] = (P16>>0) & 0xff ;
                Sr[17] = (P17>>0) & 0xff ;
                Sr[18] = (P18>>0) & 0xff ;
                Sr[19] = (P19>>0) & 0xff ;
                Sr[20] = (P20>>0) & 0xff ;
                Sr[21] = (P21>>0) & 0xff ;
                Sr[22] = (P22>>0) & 0xff ;
                Sr[23] = (P23>>0) & 0xff ;
                rs_decode_palindromic_CUDA(Sr, S, 24, k, errLocs, errMags,
                                           0+(i<<2), Lambda, B, T, Omega) ;
            }
            if ( T0 & (0xff << 8) )
            {
                atomicAdd(&error_count, 1);
                Sr[0] = (P0>>8) & 0xff ;
                Sr[1] = (P1>>8) & 0xff ;
                Sr[2] = (P2>>8) & 0xff ;
                Sr[3] = (P3>>8) & 0xff ;
                Sr[4] = (P4>>8) & 0xff ;
                Sr[5] = (P5>>8) & 0xff ;
                Sr[6] = (P6>>8) & 0xff ;
                Sr[7] = (P7>>8) & 0xff ;
                Sr[8] = (P8>>8) & 0xff ;
                Sr[9] = (P9>>8) & 0xff ;
                Sr[10] = (P10>>8) & 0xff ;
                Sr[11] = (P11>>8) & 0xff ;
                Sr[12] = (P12>>8) & 0xff ;
                Sr[13] = (P13>>8) & 0xff ;
                Sr[14] = (P14>>8) & 0xff ;
                Sr[15] = (P15>>8) & 0xff ;
                Sr[16] = (P16>>8) & 0xff ;
                Sr[17] = (P17>>8) & 0xff ;
                Sr[18] = (P18>>8) & 0xff ;
                Sr[19] = (P19>>8) & 0xff ;
                Sr[20] = (P20>>8) & 0xff ;
                Sr[21] = (P21>>8) & 0xff ;
                Sr[22] = (P22>>8) & 0xff ;
                Sr[23] = (P23>>8) & 0xff ;
                rs_decode_palindromic_CUDA(Sr, S, 24, k, errLocs, errMags,
                                           1+(i<<2), Lambda, B, T, Omega) ;
            }
            if ( T0 & (0xff << 16) )
            {
                atomicAdd(&error_count, 1);
                Sr[0] = (P0>>16) & 0xff ;
                Sr[1] = (P1>>16) & 0xff ;
                Sr[2] = (P2>>16) & 0xff ;
                Sr[3] = (P3>>16) & 0xff ;
                Sr[4] = (P4>>16) & 0xff ;
                Sr[5] = (P5>>16) & 0xff ;
                Sr[6] = (P6>>16) & 0xff ;
                Sr[7] = (P7>>16) & 0xff ;
                Sr[8] = (P8>>16) & 0xff ;
                Sr[9] = (P9>>16) & 0xff ;
                Sr[10] = (P10>>16) & 0xff ;
                Sr[11] = (P11>>16) & 0xff ;
                Sr[12] = (P12>>16) & 0xff ;
                Sr[13] = (P13>>16) & 0xff ;
                Sr[14] = (P14>>16) & 0xff ;
                Sr[15] = (P15>>16) & 0xff ;
                Sr[16] = (P16>>16) & 0xff ;
                Sr[17] = (P17>>16) & 0xff ;
                Sr[18] = (P18>>16) & 0xff ;
                Sr[19] = (P19>>16) & 0xff ;
                Sr[20] = (P20>>16) & 0xff ;
                Sr[21] = (P21>>16) & 0xff ;
                Sr[22] = (P22>>16) & 0xff ;
                Sr[23] = (P23>>16) & 0xff ;
                rs_decode_palindromic_CUDA(Sr, S, 24, k, errLocs, errMags,
                                           2+(i<<2), Lambda, B, T, Omega) ;
            }
            if ( T0 & (0xff << 24) )
            {
                atomicAdd(&error_count, 1);
                Sr[0] = (P0>>24) & 0xff ;
                Sr[1] = (P1>>24) & 0xff ;
                Sr[2] = (P2>>24) & 0xff ;
                Sr[3] = (P3>>24) & 0xff ;
                Sr[4] = (P4>>24) & 0xff ;
                Sr[5] = (P5>>24) & 0xff ;
                Sr[6] = (P6>>24) & 0xff ;
                Sr[7] = (P7>>24) & 0xff ;
                Sr[8] = (P8>>24) & 0xff ;
                Sr[9] = (P9>>24) & 0xff ;
                Sr[10] = (P10>>24) & 0xff ;
                Sr[11] = (P11>>24) & 0xff ;
                Sr[12] = (P12>>24) & 0xff ;
                Sr[13] = (P13>>24) & 0xff ;
                Sr[14] = (P14>>24) & 0xff ;
                Sr[15] = (P15>>24) & 0xff ;
                Sr[16] = (P16>>24) & 0xff ;
                Sr[17] = (P17>>24) & 0xff ;
                Sr[18] = (P18>>24) & 0xff ;
                Sr[19] = (P19>>24) & 0xff ;
                Sr[20] = (P20>>24) & 0xff ;
                Sr[21] = (P21>>24) & 0xff ;
                Sr[22] = (P22>>24) & 0xff ;
                Sr[23] = (P23>>24) & 0xff ;
                rs_decode_palindromic_CUDA(Sr, S, 24, k, errLocs, errMags,
                                           3+(i<<2), Lambda, B, T, Omega) ;
            }
        }
    }
}

// ----------------------------------------------
// Parallel LFSR Sequencer for P = 25 Codewords
// ----------------------------------------------
__global__ void ParallelLFSRSequencer25_CUDA(int k, int decoder)
{
    // Find out place in the grid of threads
    int i = blockIdx.x * blockDim.x + threadIdx.x;

    // Pointer to develop original data address
    unsigned int* loc ;

    // Create the parity values, initialize to zero
    unsigned int P0 = 0 ;
    unsigned int P1 = 0 ;
    unsigned int P2 = 0 ;
    unsigned int P3 = 0 ;
    unsigned int P4 = 0 ;
    unsigned int P5 = 0 ;
    unsigned int P6 = 0 ;
    unsigned int P7 = 0 ;
    unsigned int P8 = 0 ;
    unsigned int P9 = 0 ;
    unsigned int P10 = 0 ;
    unsigned int P11 = 0 ;
    unsigned int P12 = 0 ;
    unsigned int P13 = 0 ;
    unsigned int P14 = 0 ;
    unsigned int P15 = 0 ;
    unsigned int P16 = 0 ;
    unsigned int P17 = 0 ;
    unsigned int P18 = 0 ;
    unsigned int P19 = 0 ;
    unsigned int P20 = 0 ;
    unsigned int P21 = 0 ;
    unsigned int P22 = 0 ;
    unsigned int P23 = 0 ;
    unsigned int P24 = 0 ;

    // Temporary values for self reciprocol polynomial
    unsigned int T0;
    unsigned int T1;
    unsigned int T2;
    unsigned int T3;
    unsigned int T4;
    unsigned int T5;
    unsigned int T6;
    unsigned int T7;
    unsigned int T8;
    unsigned int T9;
    unsigned int T10;
    unsigned int T11;

    // Overflow value for LFSR and carry mask
    unsigned int OV, MSK, OF, OOV ;

    // Adjust K if we are decoding
    k += decoder ? 25 : 0 ;

    // Loop through each symbol
    for ( int sym = 0 ; sym < k ; sym ++ )
    {
        loc = CWP[sym] + i;
        OV = *loc ^ P0 ;
        OOV = OV ;

        // Process bit 0
        T1  = OV;                      // Account for bit 0 of constant d5
        T2  = OV;                      // Account for bit 0 of constant 3b
        T3  = OV;                      // Account for bit 0 of constant cf
        T4  = OV;                      // Account for bit 0 of constant 67
        T7  = OV;                      // Account for bit 0 of constant 2b
        T8  = OV;                      // Account for bit 0 of constant bf
        T10  = OV;                      // Account for bit 0 of constant 2f
        T11  = OV;                      // Account for bit 0 of constant 91
        MSK = OV & 0x80808080 ;        // Create a mask of the MSBs
        OF = (MSK >> 7) * 0x1d ;       // Account for Overflow
        OV <<= 1 ;                     // Multiply the original data by 2
        OV &= 0xfefefefe ;             // Clear the MSBs
        OV ^= OF ;                     // Add in the overflow mask

        // Process bit 1
        T0  = OV;                      // Account for bit 1 of constant 56
        T2 ^= OV;                      // Account for bit 1 of constant 3b
        T3 ^= OV;                      // Account for bit 1 of constant cf
        T4 ^= OV;                      // Account for bit 1 of constant 67
        T5  = OV;                      // Account for bit 1 of constant a2
        T7 ^= OV;                      // Account for bit 1 of constant 2b
        T8 ^= OV;                      // Account for bit 1 of constant bf
        T10 ^= OV;                      // Account for bit 1 of constant 2f
        MSK = OV & 0x80808080 ;        // Create a mask of the MSBs
        OF = (MSK >> 7) * 0x1d ;       // Account for Overflow
        OV <<= 1 ;                     // Multiply the original data by 2
        OV &= 0xfefefefe ;             // Clear the MSBs
        OV ^= OF ;                     // Add in the overflow mask

        // Process bit 2
        T0 ^= OV;                      // Account for bit 2 of constant 56
        T1 ^= OV;                      // Account for bit 2 of constant d5
        T3 ^= OV;                      // Account for bit 2 of constant cf
        T4 ^= OV;                      // Account for bit 2 of constant 67
        T6  = OV;                      // Account for bit 2 of constant c4
        T8 ^= OV;                      // Account for bit 2 of constant bf
        T10 ^= OV;                      // Account for bit 2 of constant 2f
        MSK = OV & 0x80808080 ;        // Create a mask of the MSBs
        OF = (MSK >> 7) * 0x1d ;       // Account for Overflow
        OV <<= 1 ;                     // Multiply the original data by 2
        OV &= 0xfefefefe ;             // Clear the MSBs
        OV ^= OF ;                     // Add in the overflow mask

        // Process bit 3
        T2 ^= OV;                      // Account for bit 3 of constant 3b
        T3 ^= OV;                      // Account for bit 3 of constant cf
        T7 ^= OV;                      // Account for bit 3 of constant 2b
        T8 ^= OV;                      // Account for bit 3 of constant bf
        T9  = OV;                      // Account for bit 3 of constant 18
        T10 ^= OV;                      // Account for bit 3 of constant 2f
        MSK = OV & 0x80808080 ;        // Create a mask of the MSBs
        OF = (MSK >> 7) * 0x1d ;       // Account for Overflow
        OV <<= 1 ;                     // Multiply the original data by 2
        OV &= 0xfefefefe ;             // Clear the MSBs
        OV ^= OF ;                     // Add in the overflow mask

        // Process bit 4
        T0 ^= OV;                      // Account for bit 4 of constant 56
        T1 ^= OV;                      // Account for bit 4 of constant d5
        T2 ^= OV;                      // Account for bit 4 of constant 3b
        T8 ^= OV;                      // Account for bit 4 of constant bf
        T9 ^= OV;                      // Account for bit 4 of constant 18
        T11 ^= OV;                      // Account for bit 4 of constant 91
        MSK = OV & 0x80808080 ;        // Create a mask of the MSBs
        OF = (MSK >> 7) * 0x1d ;       // Account for Overflow
        OV <<= 1 ;                     // Multiply the original data by 2
        OV &= 0xfefefefe ;             // Clear the MSBs
        OV ^= OF ;                     // Add in the overflow mask

        // Process bit 5
        T2 ^= OV;                      // Account for bit 5 of constant 3b
        T4 ^= OV;                      // Account for bit 5 of constant 67
        T5 ^= OV;                      // Account for bit 5 of constant a2
        T7 ^= OV;                      // Account for bit 5 of constant 2b
        T8 ^= OV;                      // Account for bit 5 of constant bf
        T10 ^= OV;                      // Account for bit 5 of constant 2f
        MSK = OV & 0x80808080 ;        // Create a mask of the MSBs
        OF = (MSK >> 7) * 0x1d ;       // Account for Overflow
        OV <<= 1 ;                     // Multiply the original data by 2
        OV &= 0xfefefefe ;             // Clear the MSBs
        OV ^= OF ;                     // Add in the overflow mask

        // Process bit 6
        T0 ^= OV;                      // Account for bit 6 of constant 56
        T1 ^= OV;                      // Account for bit 6 of constant d5
        T3 ^= OV;                      // Account for bit 6 of constant cf
        T4 ^= OV;                      // Account for bit 6 of constant 67
        T6 ^= OV;                      // Account for bit 6 of constant c4
        MSK = OV & 0x80808080 ;        // Create a mask of the MSBs
        OF = (MSK >> 7) * 0x1d ;       // Account for Overflow
        OV <<= 1 ;                     // Multiply the original data by 2
        OV &= 0xfefefefe ;             // Clear the MSBs
        OV ^= OF ;                     // Add in the overflow mask

        // Process bit 7
        T1 ^= OV;                      // Account for bit 7 of constant d5
        T3 ^= OV;                      // Account for bit 7 of constant cf
        T5 ^= OV;                      // Account for bit 7 of constant a2
        T6 ^= OV;                      // Account for bit 7 of constant c4
        T8 ^= OV;                      // Account for bit 7 of constant bf
        T11 ^= OV;                      // Account for bit 7 of constant 91

        // Now process the LFSR
        P0 = P1 ^ T0 ;
        P1 = P2 ^ T1 ;
        P2 = P3 ^ T2 ;
        P3 = P4 ^ T3 ;
        P4 = P5 ^ T4 ;
        P5 = P6 ^ T5 ;
        P6 = P7 ^ T6 ;
        P7 = P8 ^ T7 ;
        P8 = P9 ^ T8 ;
        P9 = P10 ^ T9 ;
        P10 = P11 ^ T10 ;
        P11 = P12 ^ T11 ;
        P12 = P13 ^ T11 ;
        P13 = P14 ^ T10 ;
        P14 = P15 ^ T9 ;
        P15 = P16 ^ T8 ;
        P16 = P17 ^ T7 ;
        P17 = P18 ^ T6 ;
        P18 = P19 ^ T5 ;
        P19 = P20 ^ T4 ;
        P20 = P21 ^ T3 ;
        P21 = P22 ^ T2 ;
        P22 = P23 ^ T1 ;
        P23 = P24 ^ T0 ;
        P24 = OOV ;
    }

    if ( decoder == 0 )
    {
        // Now store computed check symbols back to memory
        loc = CWP[k+0] + i;
        *loc = P0 ;
        loc = CWP[k+1] + i;
        *loc = P1 ;
        loc = CWP[k+2] + i;
        *loc = P2 ;
        loc = CWP[k+3] + i;
        *loc = P3 ;
        loc = CWP[k+4] + i;
        *loc = P4 ;
        loc = CWP[k+5] + i;
        *loc = P5 ;
        loc = CWP[k+6] + i;
        *loc = P6 ;
        loc = CWP[k+7] + i;
        *loc = P7 ;
        loc = CWP[k+8] + i;
        *loc = P8 ;
        loc = CWP[k+9] + i;
        *loc = P9 ;
        loc = CWP[k+10] + i;
        *loc = P10 ;
        loc = CWP[k+11] + i;
        *loc = P11 ;
        loc = CWP[k+12] + i;
        *loc = P12 ;
        loc = CWP[k+13] + i;
        *loc = P13 ;
        loc = CWP[k+14] + i;
        *loc = P14 ;
        loc = CWP[k+15] + i;
        *loc = P15 ;
        loc = CWP[k+16] + i;
        *loc = P16 ;
        loc = CWP[k+17] + i;
        *loc = P17 ;
        loc = CWP[k+18] + i;
        *loc = P18 ;
        loc = CWP[k+19] + i;
        *loc = P19 ;
        loc = CWP[k+20] + i;
        *loc = P20 ;
        loc = CWP[k+21] + i;
        *loc = P21 ;
        loc = CWP[k+22] + i;
        *loc = P22 ;
        loc = CWP[k+23] + i;
        *loc = P23 ;
        loc = CWP[k+24] + i;
        *loc = P24 ;
    }
    else
    {
        // Now check for parity equal to zero
        T0 = P0;
        T0 |= P1;
        T0 |= P2;
        T0 |= P3;
        T0 |= P4;
        T0 |= P5;
        T0 |= P6;
        T0 |= P7;
        T0 |= P8;
        T0 |= P9;
        T0 |= P10;
        T0 |= P11;
        T0 |= P12;
        T0 |= P13;
        T0 |= P14;
        T0 |= P15;
        T0 |= P16;
        T0 |= P17;
        T0 |= P18;
        T0 |= P19;
        T0 |= P20;
        T0 |= P21;
        T0 |= P22;
        T0 |= P23;
        T0 |= P24;
        if (T0 != 0)
        {
            unsigned char Sr[25], S[25], errLocs[25], errMags[25] ;
            unsigned char Lambda[25+1], B[25+1], T[25+1], Omega[25+1] ;
            
            if ( T0 & (0xff << 0) )
            {
                atomicAdd(&error_count, 1);
                Sr[0] = (P0>>0) & 0xff ;
                Sr[1] = (P1>>0) & 0xff ;
                Sr[2] = (P2>>0) & 0xff ;
                Sr[3] = (P3>>0) & 0xff ;
                Sr[4] = (P4>>0) & 0xff ;
                Sr[5] = (P5>>0) & 0xff ;
                Sr[6] = (P6>>0) & 0xff ;
                Sr[7] = (P7>>0) & 0xff ;
                Sr[8] = (P8>>0) & 0xff ;
                Sr[9] = (P9>>0) & 0xff ;
                Sr[10] = (P10>>0) & 0xff ;
                Sr[11] = (P11>>0) & 0xff ;
                Sr[12] = (P12>>0) & 0xff ;
                Sr[13] = (P13>>0) & 0xff ;
                Sr[14] = (P14>>0) & 0xff ;
                Sr[15] = (P15>>0) & 0xff ;
                Sr[16] = (P16>>0) & 0xff ;
                Sr[17] = (P17>>0) & 0xff ;
                Sr[18] = (P18>>0) & 0xff ;
                Sr[19] = (P19>>0) & 0xff ;
                Sr[20] = (P20>>0) & 0xff ;
                Sr[21] = (P21>>0) & 0xff ;
                Sr[22] = (P22>>0) & 0xff ;
                Sr[23] = (P23>>0) & 0xff ;
                Sr[24] = (P24>>0) & 0xff ;
                rs_decode_palindromic_CUDA(Sr, S, 25, k, errLocs, errMags,
                                           0+(i<<2), Lambda, B, T, Omega) ;
            }
            if ( T0 & (0xff << 8) )
            {
                atomicAdd(&error_count, 1);
                Sr[0] = (P0>>8) & 0xff ;
                Sr[1] = (P1>>8) & 0xff ;
                Sr[2] = (P2>>8) & 0xff ;
                Sr[3] = (P3>>8) & 0xff ;
                Sr[4] = (P4>>8) & 0xff ;
                Sr[5] = (P5>>8) & 0xff ;
                Sr[6] = (P6>>8) & 0xff ;
                Sr[7] = (P7>>8) & 0xff ;
                Sr[8] = (P8>>8) & 0xff ;
                Sr[9] = (P9>>8) & 0xff ;
                Sr[10] = (P10>>8) & 0xff ;
                Sr[11] = (P11>>8) & 0xff ;
                Sr[12] = (P12>>8) & 0xff ;
                Sr[13] = (P13>>8) & 0xff ;
                Sr[14] = (P14>>8) & 0xff ;
                Sr[15] = (P15>>8) & 0xff ;
                Sr[16] = (P16>>8) & 0xff ;
                Sr[17] = (P17>>8) & 0xff ;
                Sr[18] = (P18>>8) & 0xff ;
                Sr[19] = (P19>>8) & 0xff ;
                Sr[20] = (P20>>8) & 0xff ;
                Sr[21] = (P21>>8) & 0xff ;
                Sr[22] = (P22>>8) & 0xff ;
                Sr[23] = (P23>>8) & 0xff ;
                Sr[24] = (P24>>8) & 0xff ;
                rs_decode_palindromic_CUDA(Sr, S, 25, k, errLocs, errMags,
                                           1+(i<<2), Lambda, B, T, Omega) ;
            }
            if ( T0 & (0xff << 16) )
            {
                atomicAdd(&error_count, 1);
                Sr[0] = (P0>>16) & 0xff ;
                Sr[1] = (P1>>16) & 0xff ;
                Sr[2] = (P2>>16) & 0xff ;
                Sr[3] = (P3>>16) & 0xff ;
                Sr[4] = (P4>>16) & 0xff ;
                Sr[5] = (P5>>16) & 0xff ;
                Sr[6] = (P6>>16) & 0xff ;
                Sr[7] = (P7>>16) & 0xff ;
                Sr[8] = (P8>>16) & 0xff ;
                Sr[9] = (P9>>16) & 0xff ;
                Sr[10] = (P10>>16) & 0xff ;
                Sr[11] = (P11>>16) & 0xff ;
                Sr[12] = (P12>>16) & 0xff ;
                Sr[13] = (P13>>16) & 0xff ;
                Sr[14] = (P14>>16) & 0xff ;
                Sr[15] = (P15>>16) & 0xff ;
                Sr[16] = (P16>>16) & 0xff ;
                Sr[17] = (P17>>16) & 0xff ;
                Sr[18] = (P18>>16) & 0xff ;
                Sr[19] = (P19>>16) & 0xff ;
                Sr[20] = (P20>>16) & 0xff ;
                Sr[21] = (P21>>16) & 0xff ;
                Sr[22] = (P22>>16) & 0xff ;
                Sr[23] = (P23>>16) & 0xff ;
                Sr[24] = (P24>>16) & 0xff ;
                rs_decode_palindromic_CUDA(Sr, S, 25, k, errLocs, errMags,
                                           2+(i<<2), Lambda, B, T, Omega) ;
            }
            if ( T0 & (0xff << 24) )
            {
                atomicAdd(&error_count, 1);
                Sr[0] = (P0>>24) & 0xff ;
                Sr[1] = (P1>>24) & 0xff ;
                Sr[2] = (P2>>24) & 0xff ;
                Sr[3] = (P3>>24) & 0xff ;
                Sr[4] = (P4>>24) & 0xff ;
                Sr[5] = (P5>>24) & 0xff ;
                Sr[6] = (P6>>24) & 0xff ;
                Sr[7] = (P7>>24) & 0xff ;
                Sr[8] = (P8>>24) & 0xff ;
                Sr[9] = (P9>>24) & 0xff ;
                Sr[10] = (P10>>24) & 0xff ;
                Sr[11] = (P11>>24) & 0xff ;
                Sr[12] = (P12>>24) & 0xff ;
                Sr[13] = (P13>>24) & 0xff ;
                Sr[14] = (P14>>24) & 0xff ;
                Sr[15] = (P15>>24) & 0xff ;
                Sr[16] = (P16>>24) & 0xff ;
                Sr[17] = (P17>>24) & 0xff ;
                Sr[18] = (P18>>24) & 0xff ;
                Sr[19] = (P19>>24) & 0xff ;
                Sr[20] = (P20>>24) & 0xff ;
                Sr[21] = (P21>>24) & 0xff ;
                Sr[22] = (P22>>24) & 0xff ;
                Sr[23] = (P23>>24) & 0xff ;
                Sr[24] = (P24>>24) & 0xff ;
                rs_decode_palindromic_CUDA(Sr, S, 25, k, errLocs, errMags,
                                           3+(i<<2), Lambda, B, T, Omega) ;
            }
        }
    }
}

// ----------------------------------------------
// Parallel LFSR Sequencer for P = 26 Codewords
// ----------------------------------------------
__global__ void ParallelLFSRSequencer26_CUDA(int k, int decoder)
{
    // Find out place in the grid of threads
    int i = blockIdx.x * blockDim.x + threadIdx.x;

    // Pointer to develop original data address
    unsigned int* loc ;

    // Create the parity values, initialize to zero
    unsigned int P0 = 0 ;
    unsigned int P1 = 0 ;
    unsigned int P2 = 0 ;
    unsigned int P3 = 0 ;
    unsigned int P4 = 0 ;
    unsigned int P5 = 0 ;
    unsigned int P6 = 0 ;
    unsigned int P7 = 0 ;
    unsigned int P8 = 0 ;
    unsigned int P9 = 0 ;
    unsigned int P10 = 0 ;
    unsigned int P11 = 0 ;
    unsigned int P12 = 0 ;
    unsigned int P13 = 0 ;
    unsigned int P14 = 0 ;
    unsigned int P15 = 0 ;
    unsigned int P16 = 0 ;
    unsigned int P17 = 0 ;
    unsigned int P18 = 0 ;
    unsigned int P19 = 0 ;
    unsigned int P20 = 0 ;
    unsigned int P21 = 0 ;
    unsigned int P22 = 0 ;
    unsigned int P23 = 0 ;
    unsigned int P24 = 0 ;
    unsigned int P25 = 0 ;

    // Temporary values for self reciprocol polynomial
    unsigned int T0;
    unsigned int T1;
    unsigned int T2;
    unsigned int T3;
    unsigned int T4;
    unsigned int T5;
    unsigned int T6;
    unsigned int T7;
    unsigned int T8;
    unsigned int T9;
    unsigned int T10;
    unsigned int T11;
    unsigned int T12;

    // Overflow value for LFSR and carry mask
    unsigned int OV, MSK, OF, OOV ;

    // Adjust K if we are decoding
    k += decoder ? 26 : 0 ;

    // Loop through each symbol
    for ( int sym = 0 ; sym < k ; sym ++ )
    {
        loc = CWP[sym] + i;
        OV = *loc ^ P0 ;
        OOV = OV ;

        // Process bit 0
        T0  = OV;                      // Account for bit 0 of constant 27
        T1  = OV;                      // Account for bit 0 of constant 11
        T2  = OV;                      // Account for bit 0 of constant 1b
        T6  = OV;                      // Account for bit 0 of constant b7
        T7  = OV;                      // Account for bit 0 of constant 7b
        T11  = OV;                      // Account for bit 0 of constant 6d
        MSK = OV & 0x80808080 ;        // Create a mask of the MSBs
        OF = (MSK >> 7) * 0x1d ;       // Account for Overflow
        OV <<= 1 ;                     // Multiply the original data by 2
        OV &= 0xfefefefe ;             // Clear the MSBs
        OV ^= OF ;                     // Add in the overflow mask

        // Process bit 1
        T0 ^= OV;                      // Account for bit 1 of constant 27
        T2 ^= OV;                      // Account for bit 1 of constant 1b
        T3  = OV;                      // Account for bit 1 of constant a6
        T6 ^= OV;                      // Account for bit 1 of constant b7
        T7 ^= OV;                      // Account for bit 1 of constant 7b
        T8  = OV;                      // Account for bit 1 of constant e6
        T9  = OV;                      // Account for bit 1 of constant 1e
        T10  = OV;                      // Account for bit 1 of constant 22
        T12  = OV;                      // Account for bit 1 of constant 2e
        MSK = OV & 0x80808080 ;        // Create a mask of the MSBs
        OF = (MSK >> 7) * 0x1d ;       // Account for Overflow
        OV <<= 1 ;                     // Multiply the original data by 2
        OV &= 0xfefefefe ;             // Clear the MSBs
        OV ^= OF ;                     // Add in the overflow mask

        // Process bit 2
        T0 ^= OV;                      // Account for bit 2 of constant 27
        T3 ^= OV;                      // Account for bit 2 of constant a6
        T4  = OV;                      // Account for bit 2 of constant fc
        T6 ^= OV;                      // Account for bit 2 of constant b7
        T8 ^= OV;                      // Account for bit 2 of constant e6
        T9 ^= OV;                      // Account for bit 2 of constant 1e
        T11 ^= OV;                      // Account for bit 2 of constant 6d
        T12 ^= OV;                      // Account for bit 2 of constant 2e
        MSK = OV & 0x80808080 ;        // Create a mask of the MSBs
        OF = (MSK >> 7) * 0x1d ;       // Account for Overflow
        OV <<= 1 ;                     // Multiply the original data by 2
        OV &= 0xfefefefe ;             // Clear the MSBs
        OV ^= OF ;                     // Add in the overflow mask

        // Process bit 3
        T2 ^= OV;                      // Account for bit 3 of constant 1b
        T4 ^= OV;                      // Account for bit 3 of constant fc
        T5  = OV;                      // Account for bit 3 of constant 68
        T7 ^= OV;                      // Account for bit 3 of constant 7b
        T9 ^= OV;                      // Account for bit 3 of constant 1e
        T11 ^= OV;                      // Account for bit 3 of constant 6d
        T12 ^= OV;                      // Account for bit 3 of constant 2e
        MSK = OV & 0x80808080 ;        // Create a mask of the MSBs
        OF = (MSK >> 7) * 0x1d ;       // Account for Overflow
        OV <<= 1 ;                     // Multiply the original data by 2
        OV &= 0xfefefefe ;             // Clear the MSBs
        OV ^= OF ;                     // Add in the overflow mask

        // Process bit 4
        T1 ^= OV;                      // Account for bit 4 of constant 11
        T2 ^= OV;                      // Account for bit 4 of constant 1b
        T4 ^= OV;                      // Account for bit 4 of constant fc
        T6 ^= OV;                      // Account for bit 4 of constant b7
        T7 ^= OV;                      // Account for bit 4 of constant 7b
        T9 ^= OV;                      // Account for bit 4 of constant 1e
        MSK = OV & 0x80808080 ;        // Create a mask of the MSBs
        OF = (MSK >> 7) * 0x1d ;       // Account for Overflow
        OV <<= 1 ;                     // Multiply the original data by 2
        OV &= 0xfefefefe ;             // Clear the MSBs
        OV ^= OF ;                     // Add in the overflow mask

        // Process bit 5
        T0 ^= OV;                      // Account for bit 5 of constant 27
        T3 ^= OV;                      // Account for bit 5 of constant a6
        T4 ^= OV;                      // Account for bit 5 of constant fc
        T5 ^= OV;                      // Account for bit 5 of constant 68
        T6 ^= OV;                      // Account for bit 5 of constant b7
        T7 ^= OV;                      // Account for bit 5 of constant 7b
        T8 ^= OV;                      // Account for bit 5 of constant e6
        T10 ^= OV;                      // Account for bit 5 of constant 22
        T11 ^= OV;                      // Account for bit 5 of constant 6d
        T12 ^= OV;                      // Account for bit 5 of constant 2e
        MSK = OV & 0x80808080 ;        // Create a mask of the MSBs
        OF = (MSK >> 7) * 0x1d ;       // Account for Overflow
        OV <<= 1 ;                     // Multiply the original data by 2
        OV &= 0xfefefefe ;             // Clear the MSBs
        OV ^= OF ;                     // Add in the overflow mask

        // Process bit 6
        T4 ^= OV;                      // Account for bit 6 of constant fc
        T5 ^= OV;                      // Account for bit 6 of constant 68
        T7 ^= OV;                      // Account for bit 6 of constant 7b
        T8 ^= OV;                      // Account for bit 6 of constant e6
        T11 ^= OV;                      // Account for bit 6 of constant 6d
        MSK = OV & 0x80808080 ;        // Create a mask of the MSBs
        OF = (MSK >> 7) * 0x1d ;       // Account for Overflow
        OV <<= 1 ;                     // Multiply the original data by 2
        OV &= 0xfefefefe ;             // Clear the MSBs
        OV ^= OF ;                     // Add in the overflow mask

        // Process bit 7
        T3 ^= OV;                      // Account for bit 7 of constant a6
        T4 ^= OV;                      // Account for bit 7 of constant fc
        T6 ^= OV;                      // Account for bit 7 of constant b7
        T8 ^= OV;                      // Account for bit 7 of constant e6

        // Now process the LFSR
        P0 = P1 ^ T0 ;
        P1 = P2 ^ T1 ;
        P2 = P3 ^ T2 ;
        P3 = P4 ^ T3 ;
        P4 = P5 ^ T4 ;
        P5 = P6 ^ T5 ;
        P6 = P7 ^ T6 ;
        P7 = P8 ^ T7 ;
        P8 = P9 ^ T8 ;
        P9 = P10 ^ T9 ;
        P10 = P11 ^ T10 ;
        P11 = P12 ^ T11 ;
        P12 = P13 ^ T12 ;
        P13 = P14 ^ T11 ;
        P14 = P15 ^ T10 ;
        P15 = P16 ^ T9 ;
        P16 = P17 ^ T8 ;
        P17 = P18 ^ T7 ;
        P18 = P19 ^ T6 ;
        P19 = P20 ^ T5 ;
        P20 = P21 ^ T4 ;
        P21 = P22 ^ T3 ;
        P22 = P23 ^ T2 ;
        P23 = P24 ^ T1 ;
        P24 = P25 ^ T0 ;
        P25 = OOV ;
    }

    if ( decoder == 0 )
    {
        // Now store computed check symbols back to memory
        loc = CWP[k+0] + i;
        *loc = P0 ;
        loc = CWP[k+1] + i;
        *loc = P1 ;
        loc = CWP[k+2] + i;
        *loc = P2 ;
        loc = CWP[k+3] + i;
        *loc = P3 ;
        loc = CWP[k+4] + i;
        *loc = P4 ;
        loc = CWP[k+5] + i;
        *loc = P5 ;
        loc = CWP[k+6] + i;
        *loc = P6 ;
        loc = CWP[k+7] + i;
        *loc = P7 ;
        loc = CWP[k+8] + i;
        *loc = P8 ;
        loc = CWP[k+9] + i;
        *loc = P9 ;
        loc = CWP[k+10] + i;
        *loc = P10 ;
        loc = CWP[k+11] + i;
        *loc = P11 ;
        loc = CWP[k+12] + i;
        *loc = P12 ;
        loc = CWP[k+13] + i;
        *loc = P13 ;
        loc = CWP[k+14] + i;
        *loc = P14 ;
        loc = CWP[k+15] + i;
        *loc = P15 ;
        loc = CWP[k+16] + i;
        *loc = P16 ;
        loc = CWP[k+17] + i;
        *loc = P17 ;
        loc = CWP[k+18] + i;
        *loc = P18 ;
        loc = CWP[k+19] + i;
        *loc = P19 ;
        loc = CWP[k+20] + i;
        *loc = P20 ;
        loc = CWP[k+21] + i;
        *loc = P21 ;
        loc = CWP[k+22] + i;
        *loc = P22 ;
        loc = CWP[k+23] + i;
        *loc = P23 ;
        loc = CWP[k+24] + i;
        *loc = P24 ;
        loc = CWP[k+25] + i;
        *loc = P25 ;
    }
    else
    {
        // Now check for parity equal to zero
        T0 = P0;
        T0 |= P1;
        T0 |= P2;
        T0 |= P3;
        T0 |= P4;
        T0 |= P5;
        T0 |= P6;
        T0 |= P7;
        T0 |= P8;
        T0 |= P9;
        T0 |= P10;
        T0 |= P11;
        T0 |= P12;
        T0 |= P13;
        T0 |= P14;
        T0 |= P15;
        T0 |= P16;
        T0 |= P17;
        T0 |= P18;
        T0 |= P19;
        T0 |= P20;
        T0 |= P21;
        T0 |= P22;
        T0 |= P23;
        T0 |= P24;
        T0 |= P25;
        if (T0 != 0)
        {
            unsigned char Sr[26], S[26], errLocs[26], errMags[26] ;
            unsigned char Lambda[26+1], B[26+1], T[26+1], Omega[26+1] ;
            
            if ( T0 & (0xff << 0) )
            {
                atomicAdd(&error_count, 1);
                Sr[0] = (P0>>0) & 0xff ;
                Sr[1] = (P1>>0) & 0xff ;
                Sr[2] = (P2>>0) & 0xff ;
                Sr[3] = (P3>>0) & 0xff ;
                Sr[4] = (P4>>0) & 0xff ;
                Sr[5] = (P5>>0) & 0xff ;
                Sr[6] = (P6>>0) & 0xff ;
                Sr[7] = (P7>>0) & 0xff ;
                Sr[8] = (P8>>0) & 0xff ;
                Sr[9] = (P9>>0) & 0xff ;
                Sr[10] = (P10>>0) & 0xff ;
                Sr[11] = (P11>>0) & 0xff ;
                Sr[12] = (P12>>0) & 0xff ;
                Sr[13] = (P13>>0) & 0xff ;
                Sr[14] = (P14>>0) & 0xff ;
                Sr[15] = (P15>>0) & 0xff ;
                Sr[16] = (P16>>0) & 0xff ;
                Sr[17] = (P17>>0) & 0xff ;
                Sr[18] = (P18>>0) & 0xff ;
                Sr[19] = (P19>>0) & 0xff ;
                Sr[20] = (P20>>0) & 0xff ;
                Sr[21] = (P21>>0) & 0xff ;
                Sr[22] = (P22>>0) & 0xff ;
                Sr[23] = (P23>>0) & 0xff ;
                Sr[24] = (P24>>0) & 0xff ;
                Sr[25] = (P25>>0) & 0xff ;
                rs_decode_palindromic_CUDA(Sr, S, 26, k, errLocs, errMags,
                                           0+(i<<2), Lambda, B, T, Omega) ;
            }
            if ( T0 & (0xff << 8) )
            {
                atomicAdd(&error_count, 1);
                Sr[0] = (P0>>8) & 0xff ;
                Sr[1] = (P1>>8) & 0xff ;
                Sr[2] = (P2>>8) & 0xff ;
                Sr[3] = (P3>>8) & 0xff ;
                Sr[4] = (P4>>8) & 0xff ;
                Sr[5] = (P5>>8) & 0xff ;
                Sr[6] = (P6>>8) & 0xff ;
                Sr[7] = (P7>>8) & 0xff ;
                Sr[8] = (P8>>8) & 0xff ;
                Sr[9] = (P9>>8) & 0xff ;
                Sr[10] = (P10>>8) & 0xff ;
                Sr[11] = (P11>>8) & 0xff ;
                Sr[12] = (P12>>8) & 0xff ;
                Sr[13] = (P13>>8) & 0xff ;
                Sr[14] = (P14>>8) & 0xff ;
                Sr[15] = (P15>>8) & 0xff ;
                Sr[16] = (P16>>8) & 0xff ;
                Sr[17] = (P17>>8) & 0xff ;
                Sr[18] = (P18>>8) & 0xff ;
                Sr[19] = (P19>>8) & 0xff ;
                Sr[20] = (P20>>8) & 0xff ;
                Sr[21] = (P21>>8) & 0xff ;
                Sr[22] = (P22>>8) & 0xff ;
                Sr[23] = (P23>>8) & 0xff ;
                Sr[24] = (P24>>8) & 0xff ;
                Sr[25] = (P25>>8) & 0xff ;
                rs_decode_palindromic_CUDA(Sr, S, 26, k, errLocs, errMags,
                                           1+(i<<2), Lambda, B, T, Omega) ;
            }
            if ( T0 & (0xff << 16) )
            {
                atomicAdd(&error_count, 1);
                Sr[0] = (P0>>16) & 0xff ;
                Sr[1] = (P1>>16) & 0xff ;
                Sr[2] = (P2>>16) & 0xff ;
                Sr[3] = (P3>>16) & 0xff ;
                Sr[4] = (P4>>16) & 0xff ;
                Sr[5] = (P5>>16) & 0xff ;
                Sr[6] = (P6>>16) & 0xff ;
                Sr[7] = (P7>>16) & 0xff ;
                Sr[8] = (P8>>16) & 0xff ;
                Sr[9] = (P9>>16) & 0xff ;
                Sr[10] = (P10>>16) & 0xff ;
                Sr[11] = (P11>>16) & 0xff ;
                Sr[12] = (P12>>16) & 0xff ;
                Sr[13] = (P13>>16) & 0xff ;
                Sr[14] = (P14>>16) & 0xff ;
                Sr[15] = (P15>>16) & 0xff ;
                Sr[16] = (P16>>16) & 0xff ;
                Sr[17] = (P17>>16) & 0xff ;
                Sr[18] = (P18>>16) & 0xff ;
                Sr[19] = (P19>>16) & 0xff ;
                Sr[20] = (P20>>16) & 0xff ;
                Sr[21] = (P21>>16) & 0xff ;
                Sr[22] = (P22>>16) & 0xff ;
                Sr[23] = (P23>>16) & 0xff ;
                Sr[24] = (P24>>16) & 0xff ;
                Sr[25] = (P25>>16) & 0xff ;
                rs_decode_palindromic_CUDA(Sr, S, 26, k, errLocs, errMags,
                                           2+(i<<2), Lambda, B, T, Omega) ;
            }
            if ( T0 & (0xff << 24) )
            {
                atomicAdd(&error_count, 1);
                Sr[0] = (P0>>24) & 0xff ;
                Sr[1] = (P1>>24) & 0xff ;
                Sr[2] = (P2>>24) & 0xff ;
                Sr[3] = (P3>>24) & 0xff ;
                Sr[4] = (P4>>24) & 0xff ;
                Sr[5] = (P5>>24) & 0xff ;
                Sr[6] = (P6>>24) & 0xff ;
                Sr[7] = (P7>>24) & 0xff ;
                Sr[8] = (P8>>24) & 0xff ;
                Sr[9] = (P9>>24) & 0xff ;
                Sr[10] = (P10>>24) & 0xff ;
                Sr[11] = (P11>>24) & 0xff ;
                Sr[12] = (P12>>24) & 0xff ;
                Sr[13] = (P13>>24) & 0xff ;
                Sr[14] = (P14>>24) & 0xff ;
                Sr[15] = (P15>>24) & 0xff ;
                Sr[16] = (P16>>24) & 0xff ;
                Sr[17] = (P17>>24) & 0xff ;
                Sr[18] = (P18>>24) & 0xff ;
                Sr[19] = (P19>>24) & 0xff ;
                Sr[20] = (P20>>24) & 0xff ;
                Sr[21] = (P21>>24) & 0xff ;
                Sr[22] = (P22>>24) & 0xff ;
                Sr[23] = (P23>>24) & 0xff ;
                Sr[24] = (P24>>24) & 0xff ;
                Sr[25] = (P25>>24) & 0xff ;
                rs_decode_palindromic_CUDA(Sr, S, 26, k, errLocs, errMags,
                                           3+(i<<2), Lambda, B, T, Omega) ;
            }
        }
    }
}

// ----------------------------------------------
// Parallel LFSR Sequencer for P = 27 Codewords
// ----------------------------------------------
__global__ void ParallelLFSRSequencer27_CUDA(int k, int decoder)
{
    // Find out place in the grid of threads
    int i = blockIdx.x * blockDim.x + threadIdx.x;

    // Pointer to develop original data address
    unsigned int* loc ;

    // Create the parity values, initialize to zero
    unsigned int P0 = 0 ;
    unsigned int P1 = 0 ;
    unsigned int P2 = 0 ;
    unsigned int P3 = 0 ;
    unsigned int P4 = 0 ;
    unsigned int P5 = 0 ;
    unsigned int P6 = 0 ;
    unsigned int P7 = 0 ;
    unsigned int P8 = 0 ;
    unsigned int P9 = 0 ;
    unsigned int P10 = 0 ;
    unsigned int P11 = 0 ;
    unsigned int P12 = 0 ;
    unsigned int P13 = 0 ;
    unsigned int P14 = 0 ;
    unsigned int P15 = 0 ;
    unsigned int P16 = 0 ;
    unsigned int P17 = 0 ;
    unsigned int P18 = 0 ;
    unsigned int P19 = 0 ;
    unsigned int P20 = 0 ;
    unsigned int P21 = 0 ;
    unsigned int P22 = 0 ;
    unsigned int P23 = 0 ;
    unsigned int P24 = 0 ;
    unsigned int P25 = 0 ;
    unsigned int P26 = 0 ;

    // Temporary values for self reciprocol polynomial
    unsigned int T0;
    unsigned int T1;
    unsigned int T2;
    unsigned int T3;
    unsigned int T4;
    unsigned int T5;
    unsigned int T6;
    unsigned int T7;
    unsigned int T8;
    unsigned int T9;
    unsigned int T10;
    unsigned int T11;
    unsigned int T12;

    // Overflow value for LFSR and carry mask
    unsigned int OV, MSK, OF, OOV ;

    // Adjust K if we are decoding
    k += decoder ? 27 : 0 ;

    // Loop through each symbol
    for ( int sym = 0 ; sym < k ; sym ++ )
    {
        loc = CWP[sym] + i;
        OV = *loc ^ P0 ;
        OOV = OV ;

        // Process bit 0
        T0  = OV;                      // Account for bit 0 of constant 61
        T3  = OV;                      // Account for bit 0 of constant 1f
        T4  = OV;                      // Account for bit 0 of constant bf
        T5  = OV;                      // Account for bit 0 of constant 7
        T6  = OV;                      // Account for bit 0 of constant e1
        T8  = OV;                      // Account for bit 0 of constant 29
        T11  = OV;                      // Account for bit 0 of constant 7
        MSK = OV & 0x80808080 ;        // Create a mask of the MSBs
        OF = (MSK >> 7) * 0x1d ;       // Account for Overflow
        OV <<= 1 ;                     // Multiply the original data by 2
        OV &= 0xfefefefe ;             // Clear the MSBs
        OV ^= OF ;                     // Add in the overflow mask

        // Process bit 1
        T2  = OV;                      // Account for bit 1 of constant 12
        T3 ^= OV;                      // Account for bit 1 of constant 1f
        T4 ^= OV;                      // Account for bit 1 of constant bf
        T5 ^= OV;                      // Account for bit 1 of constant 7
        T7  = OV;                      // Account for bit 1 of constant 96
        T10  = OV;                      // Account for bit 1 of constant 62
        T11 ^= OV;                      // Account for bit 1 of constant 7
        MSK = OV & 0x80808080 ;        // Create a mask of the MSBs
        OF = (MSK >> 7) * 0x1d ;       // Account for Overflow
        OV <<= 1 ;                     // Multiply the original data by 2
        OV &= 0xfefefefe ;             // Clear the MSBs
        OV ^= OF ;                     // Add in the overflow mask

        // Process bit 2
        T3 ^= OV;                      // Account for bit 2 of constant 1f
        T4 ^= OV;                      // Account for bit 2 of constant bf
        T5 ^= OV;                      // Account for bit 2 of constant 7
        T7 ^= OV;                      // Account for bit 2 of constant 96
        T11 ^= OV;                      // Account for bit 2 of constant 7
        T12  = OV;                      // Account for bit 2 of constant 5c
        MSK = OV & 0x80808080 ;        // Create a mask of the MSBs
        OF = (MSK >> 7) * 0x1d ;       // Account for Overflow
        OV <<= 1 ;                     // Multiply the original data by 2
        OV &= 0xfefefefe ;             // Clear the MSBs
        OV ^= OF ;                     // Add in the overflow mask

        // Process bit 3
        T3 ^= OV;                      // Account for bit 3 of constant 1f
        T4 ^= OV;                      // Account for bit 3 of constant bf
        T8 ^= OV;                      // Account for bit 3 of constant 29
        T9  = OV;                      // Account for bit 3 of constant 68
        T12 ^= OV;                      // Account for bit 3 of constant 5c
        MSK = OV & 0x80808080 ;        // Create a mask of the MSBs
        OF = (MSK >> 7) * 0x1d ;       // Account for Overflow
        OV <<= 1 ;                     // Multiply the original data by 2
        OV &= 0xfefefefe ;             // Clear the MSBs
        OV ^= OF ;                     // Add in the overflow mask

        // Process bit 4
        T1  = OV;                      // Account for bit 4 of constant 70
        T2 ^= OV;                      // Account for bit 4 of constant 12
        T3 ^= OV;                      // Account for bit 4 of constant 1f
        T4 ^= OV;                      // Account for bit 4 of constant bf
        T7 ^= OV;                      // Account for bit 4 of constant 96
        T12 ^= OV;                      // Account for bit 4 of constant 5c
        MSK = OV & 0x80808080 ;        // Create a mask of the MSBs
        OF = (MSK >> 7) * 0x1d ;       // Account for Overflow
        OV <<= 1 ;                     // Multiply the original data by 2
        OV &= 0xfefefefe ;             // Clear the MSBs
        OV ^= OF ;                     // Add in the overflow mask

        // Process bit 5
        T0 ^= OV;                      // Account for bit 5 of constant 61
        T1 ^= OV;                      // Account for bit 5 of constant 70
        T4 ^= OV;                      // Account for bit 5 of constant bf
        T6 ^= OV;                      // Account for bit 5 of constant e1
        T8 ^= OV;                      // Account for bit 5 of constant 29
        T9 ^= OV;                      // Account for bit 5 of constant 68
        T10 ^= OV;                      // Account for bit 5 of constant 62
        MSK = OV & 0x80808080 ;        // Create a mask of the MSBs
        OF = (MSK >> 7) * 0x1d ;       // Account for Overflow
        OV <<= 1 ;                     // Multiply the original data by 2
        OV &= 0xfefefefe ;             // Clear the MSBs
        OV ^= OF ;                     // Add in the overflow mask

        // Process bit 6
        T0 ^= OV;                      // Account for bit 6 of constant 61
        T1 ^= OV;                      // Account for bit 6 of constant 70
        T6 ^= OV;                      // Account for bit 6 of constant e1
        T9 ^= OV;                      // Account for bit 6 of constant 68
        T10 ^= OV;                      // Account for bit 6 of constant 62
        T12 ^= OV;                      // Account for bit 6 of constant 5c
        MSK = OV & 0x80808080 ;        // Create a mask of the MSBs
        OF = (MSK >> 7) * 0x1d ;       // Account for Overflow
        OV <<= 1 ;                     // Multiply the original data by 2
        OV &= 0xfefefefe ;             // Clear the MSBs
        OV ^= OF ;                     // Add in the overflow mask

        // Process bit 7
        T4 ^= OV;                      // Account for bit 7 of constant bf
        T6 ^= OV;                      // Account for bit 7 of constant e1
        T7 ^= OV;                      // Account for bit 7 of constant 96

        // Now process the LFSR
        P0 = P1 ^ T0 ;
        P1 = P2 ^ T1 ;
        P2 = P3 ^ T2 ;
        P3 = P4 ^ T3 ;
        P4 = P5 ^ T4 ;
        P5 = P6 ^ T5 ;
        P6 = P7 ^ T6 ;
        P7 = P8 ^ T7 ;
        P8 = P9 ^ T8 ;
        P9 = P10 ^ T9 ;
        P10 = P11 ^ T10 ;
        P11 = P12 ^ T11 ;
        P12 = P13 ^ T12 ;
        P13 = P14 ^ T12 ;
        P14 = P15 ^ T11 ;
        P15 = P16 ^ T10 ;
        P16 = P17 ^ T9 ;
        P17 = P18 ^ T8 ;
        P18 = P19 ^ T7 ;
        P19 = P20 ^ T6 ;
        P20 = P21 ^ T5 ;
        P21 = P22 ^ T4 ;
        P22 = P23 ^ T3 ;
        P23 = P24 ^ T2 ;
        P24 = P25 ^ T1 ;
        P25 = P26 ^ T0 ;
        P26 = OOV ;
    }

    if ( decoder == 0 )
    {
        // Now store computed check symbols back to memory
        loc = CWP[k+0] + i;
        *loc = P0 ;
        loc = CWP[k+1] + i;
        *loc = P1 ;
        loc = CWP[k+2] + i;
        *loc = P2 ;
        loc = CWP[k+3] + i;
        *loc = P3 ;
        loc = CWP[k+4] + i;
        *loc = P4 ;
        loc = CWP[k+5] + i;
        *loc = P5 ;
        loc = CWP[k+6] + i;
        *loc = P6 ;
        loc = CWP[k+7] + i;
        *loc = P7 ;
        loc = CWP[k+8] + i;
        *loc = P8 ;
        loc = CWP[k+9] + i;
        *loc = P9 ;
        loc = CWP[k+10] + i;
        *loc = P10 ;
        loc = CWP[k+11] + i;
        *loc = P11 ;
        loc = CWP[k+12] + i;
        *loc = P12 ;
        loc = CWP[k+13] + i;
        *loc = P13 ;
        loc = CWP[k+14] + i;
        *loc = P14 ;
        loc = CWP[k+15] + i;
        *loc = P15 ;
        loc = CWP[k+16] + i;
        *loc = P16 ;
        loc = CWP[k+17] + i;
        *loc = P17 ;
        loc = CWP[k+18] + i;
        *loc = P18 ;
        loc = CWP[k+19] + i;
        *loc = P19 ;
        loc = CWP[k+20] + i;
        *loc = P20 ;
        loc = CWP[k+21] + i;
        *loc = P21 ;
        loc = CWP[k+22] + i;
        *loc = P22 ;
        loc = CWP[k+23] + i;
        *loc = P23 ;
        loc = CWP[k+24] + i;
        *loc = P24 ;
        loc = CWP[k+25] + i;
        *loc = P25 ;
        loc = CWP[k+26] + i;
        *loc = P26 ;
    }
    else
    {
        // Now check for parity equal to zero
        T0 = P0;
        T0 |= P1;
        T0 |= P2;
        T0 |= P3;
        T0 |= P4;
        T0 |= P5;
        T0 |= P6;
        T0 |= P7;
        T0 |= P8;
        T0 |= P9;
        T0 |= P10;
        T0 |= P11;
        T0 |= P12;
        T0 |= P13;
        T0 |= P14;
        T0 |= P15;
        T0 |= P16;
        T0 |= P17;
        T0 |= P18;
        T0 |= P19;
        T0 |= P20;
        T0 |= P21;
        T0 |= P22;
        T0 |= P23;
        T0 |= P24;
        T0 |= P25;
        T0 |= P26;
        if (T0 != 0)
        {
            unsigned char Sr[27], S[27], errLocs[27], errMags[27] ;
            unsigned char Lambda[27+1], B[27+1], T[27+1], Omega[27+1] ;
            
            if ( T0 & (0xff << 0) )
            {
                atomicAdd(&error_count, 1);
                Sr[0] = (P0>>0) & 0xff ;
                Sr[1] = (P1>>0) & 0xff ;
                Sr[2] = (P2>>0) & 0xff ;
                Sr[3] = (P3>>0) & 0xff ;
                Sr[4] = (P4>>0) & 0xff ;
                Sr[5] = (P5>>0) & 0xff ;
                Sr[6] = (P6>>0) & 0xff ;
                Sr[7] = (P7>>0) & 0xff ;
                Sr[8] = (P8>>0) & 0xff ;
                Sr[9] = (P9>>0) & 0xff ;
                Sr[10] = (P10>>0) & 0xff ;
                Sr[11] = (P11>>0) & 0xff ;
                Sr[12] = (P12>>0) & 0xff ;
                Sr[13] = (P13>>0) & 0xff ;
                Sr[14] = (P14>>0) & 0xff ;
                Sr[15] = (P15>>0) & 0xff ;
                Sr[16] = (P16>>0) & 0xff ;
                Sr[17] = (P17>>0) & 0xff ;
                Sr[18] = (P18>>0) & 0xff ;
                Sr[19] = (P19>>0) & 0xff ;
                Sr[20] = (P20>>0) & 0xff ;
                Sr[21] = (P21>>0) & 0xff ;
                Sr[22] = (P22>>0) & 0xff ;
                Sr[23] = (P23>>0) & 0xff ;
                Sr[24] = (P24>>0) & 0xff ;
                Sr[25] = (P25>>0) & 0xff ;
                Sr[26] = (P26>>0) & 0xff ;
                rs_decode_palindromic_CUDA(Sr, S, 27, k, errLocs, errMags,
                                           0+(i<<2), Lambda, B, T, Omega) ;
            }
            if ( T0 & (0xff << 8) )
            {
                atomicAdd(&error_count, 1);
                Sr[0] = (P0>>8) & 0xff ;
                Sr[1] = (P1>>8) & 0xff ;
                Sr[2] = (P2>>8) & 0xff ;
                Sr[3] = (P3>>8) & 0xff ;
                Sr[4] = (P4>>8) & 0xff ;
                Sr[5] = (P5>>8) & 0xff ;
                Sr[6] = (P6>>8) & 0xff ;
                Sr[7] = (P7>>8) & 0xff ;
                Sr[8] = (P8>>8) & 0xff ;
                Sr[9] = (P9>>8) & 0xff ;
                Sr[10] = (P10>>8) & 0xff ;
                Sr[11] = (P11>>8) & 0xff ;
                Sr[12] = (P12>>8) & 0xff ;
                Sr[13] = (P13>>8) & 0xff ;
                Sr[14] = (P14>>8) & 0xff ;
                Sr[15] = (P15>>8) & 0xff ;
                Sr[16] = (P16>>8) & 0xff ;
                Sr[17] = (P17>>8) & 0xff ;
                Sr[18] = (P18>>8) & 0xff ;
                Sr[19] = (P19>>8) & 0xff ;
                Sr[20] = (P20>>8) & 0xff ;
                Sr[21] = (P21>>8) & 0xff ;
                Sr[22] = (P22>>8) & 0xff ;
                Sr[23] = (P23>>8) & 0xff ;
                Sr[24] = (P24>>8) & 0xff ;
                Sr[25] = (P25>>8) & 0xff ;
                Sr[26] = (P26>>8) & 0xff ;
                rs_decode_palindromic_CUDA(Sr, S, 27, k, errLocs, errMags,
                                           1+(i<<2), Lambda, B, T, Omega) ;
            }
            if ( T0 & (0xff << 16) )
            {
                atomicAdd(&error_count, 1);
                Sr[0] = (P0>>16) & 0xff ;
                Sr[1] = (P1>>16) & 0xff ;
                Sr[2] = (P2>>16) & 0xff ;
                Sr[3] = (P3>>16) & 0xff ;
                Sr[4] = (P4>>16) & 0xff ;
                Sr[5] = (P5>>16) & 0xff ;
                Sr[6] = (P6>>16) & 0xff ;
                Sr[7] = (P7>>16) & 0xff ;
                Sr[8] = (P8>>16) & 0xff ;
                Sr[9] = (P9>>16) & 0xff ;
                Sr[10] = (P10>>16) & 0xff ;
                Sr[11] = (P11>>16) & 0xff ;
                Sr[12] = (P12>>16) & 0xff ;
                Sr[13] = (P13>>16) & 0xff ;
                Sr[14] = (P14>>16) & 0xff ;
                Sr[15] = (P15>>16) & 0xff ;
                Sr[16] = (P16>>16) & 0xff ;
                Sr[17] = (P17>>16) & 0xff ;
                Sr[18] = (P18>>16) & 0xff ;
                Sr[19] = (P19>>16) & 0xff ;
                Sr[20] = (P20>>16) & 0xff ;
                Sr[21] = (P21>>16) & 0xff ;
                Sr[22] = (P22>>16) & 0xff ;
                Sr[23] = (P23>>16) & 0xff ;
                Sr[24] = (P24>>16) & 0xff ;
                Sr[25] = (P25>>16) & 0xff ;
                Sr[26] = (P26>>16) & 0xff ;
                rs_decode_palindromic_CUDA(Sr, S, 27, k, errLocs, errMags,
                                           2+(i<<2), Lambda, B, T, Omega) ;
            }
            if ( T0 & (0xff << 24) )
            {
                atomicAdd(&error_count, 1);
                Sr[0] = (P0>>24) & 0xff ;
                Sr[1] = (P1>>24) & 0xff ;
                Sr[2] = (P2>>24) & 0xff ;
                Sr[3] = (P3>>24) & 0xff ;
                Sr[4] = (P4>>24) & 0xff ;
                Sr[5] = (P5>>24) & 0xff ;
                Sr[6] = (P6>>24) & 0xff ;
                Sr[7] = (P7>>24) & 0xff ;
                Sr[8] = (P8>>24) & 0xff ;
                Sr[9] = (P9>>24) & 0xff ;
                Sr[10] = (P10>>24) & 0xff ;
                Sr[11] = (P11>>24) & 0xff ;
                Sr[12] = (P12>>24) & 0xff ;
                Sr[13] = (P13>>24) & 0xff ;
                Sr[14] = (P14>>24) & 0xff ;
                Sr[15] = (P15>>24) & 0xff ;
                Sr[16] = (P16>>24) & 0xff ;
                Sr[17] = (P17>>24) & 0xff ;
                Sr[18] = (P18>>24) & 0xff ;
                Sr[19] = (P19>>24) & 0xff ;
                Sr[20] = (P20>>24) & 0xff ;
                Sr[21] = (P21>>24) & 0xff ;
                Sr[22] = (P22>>24) & 0xff ;
                Sr[23] = (P23>>24) & 0xff ;
                Sr[24] = (P24>>24) & 0xff ;
                Sr[25] = (P25>>24) & 0xff ;
                Sr[26] = (P26>>24) & 0xff ;
                rs_decode_palindromic_CUDA(Sr, S, 27, k, errLocs, errMags,
                                           3+(i<<2), Lambda, B, T, Omega) ;
            }
        }
    }
}

// ----------------------------------------------
// Parallel LFSR Sequencer for P = 28 Codewords
// ----------------------------------------------
__global__ void ParallelLFSRSequencer28_CUDA(int k, int decoder)
{
    // Find out place in the grid of threads
    int i = blockIdx.x * blockDim.x + threadIdx.x;

    // Pointer to develop original data address
    unsigned int* loc ;

    // Create the parity values, initialize to zero
    unsigned int P0 = 0 ;
    unsigned int P1 = 0 ;
    unsigned int P2 = 0 ;
    unsigned int P3 = 0 ;
    unsigned int P4 = 0 ;
    unsigned int P5 = 0 ;
    unsigned int P6 = 0 ;
    unsigned int P7 = 0 ;
    unsigned int P8 = 0 ;
    unsigned int P9 = 0 ;
    unsigned int P10 = 0 ;
    unsigned int P11 = 0 ;
    unsigned int P12 = 0 ;
    unsigned int P13 = 0 ;
    unsigned int P14 = 0 ;
    unsigned int P15 = 0 ;
    unsigned int P16 = 0 ;
    unsigned int P17 = 0 ;
    unsigned int P18 = 0 ;
    unsigned int P19 = 0 ;
    unsigned int P20 = 0 ;
    unsigned int P21 = 0 ;
    unsigned int P22 = 0 ;
    unsigned int P23 = 0 ;
    unsigned int P24 = 0 ;
    unsigned int P25 = 0 ;
    unsigned int P26 = 0 ;
    unsigned int P27 = 0 ;

    // Temporary values for self reciprocol polynomial
    unsigned int T0;
    unsigned int T1;
    unsigned int T2;
    unsigned int T3;
    unsigned int T4;
    unsigned int T5;
    unsigned int T6;
    unsigned int T7;
    unsigned int T8;
    unsigned int T9;
    unsigned int T10;
    unsigned int T11;
    unsigned int T12;
    unsigned int T13;

    // Overflow value for LFSR and carry mask
    unsigned int OV, MSK, OF, OOV ;

    // Adjust K if we are decoding
    k += decoder ? 28 : 0 ;

    // Loop through each symbol
    for ( int sym = 0 ; sym < k ; sym ++ )
    {
        loc = CWP[sym] + i;
        OV = *loc ^ P0 ;
        OOV = OV ;

        // Process bit 0
        T2  = OV;                      // Account for bit 0 of constant 9d
        T3  = OV;                      // Account for bit 0 of constant 5
        T6  = OV;                      // Account for bit 0 of constant 15
        T7  = OV;                      // Account for bit 0 of constant 65
        T10  = OV;                      // Account for bit 0 of constant f1
        T12  = OV;                      // Account for bit 0 of constant d5
        MSK = OV & 0x80808080 ;        // Create a mask of the MSBs
        OF = (MSK >> 7) * 0x1d ;       // Account for Overflow
        OV <<= 1 ;                     // Multiply the original data by 2
        OV &= 0xfefefefe ;             // Clear the MSBs
        OV ^= OF ;                     // Add in the overflow mask

        // Process bit 1
        T5  = OV;                      // Account for bit 1 of constant 92
        T13  = OV;                      // Account for bit 1 of constant b6
        MSK = OV & 0x80808080 ;        // Create a mask of the MSBs
        OF = (MSK >> 7) * 0x1d ;       // Account for Overflow
        OV <<= 1 ;                     // Multiply the original data by 2
        OV &= 0xfefefefe ;             // Clear the MSBs
        OV ^= OF ;                     // Add in the overflow mask

        // Process bit 2
        T0  = OV;                      // Account for bit 2 of constant c
        T2 ^= OV;                      // Account for bit 2 of constant 9d
        T3 ^= OV;                      // Account for bit 2 of constant 5
        T6 ^= OV;                      // Account for bit 2 of constant 15
        T7 ^= OV;                      // Account for bit 2 of constant 65
        T11  = OV;                      // Account for bit 2 of constant 2c
        T12 ^= OV;                      // Account for bit 2 of constant d5
        T13 ^= OV;                      // Account for bit 2 of constant b6
        MSK = OV & 0x80808080 ;        // Create a mask of the MSBs
        OF = (MSK >> 7) * 0x1d ;       // Account for Overflow
        OV <<= 1 ;                     // Multiply the original data by 2
        OV &= 0xfefefefe ;             // Clear the MSBs
        OV ^= OF ;                     // Add in the overflow mask

        // Process bit 3
        T0 ^= OV;                      // Account for bit 3 of constant c
        T1  = OV;                      // Account for bit 3 of constant c8
        T2 ^= OV;                      // Account for bit 3 of constant 9d
        T8  = OV;                      // Account for bit 3 of constant f8
        T11 ^= OV;                      // Account for bit 3 of constant 2c
        MSK = OV & 0x80808080 ;        // Create a mask of the MSBs
        OF = (MSK >> 7) * 0x1d ;       // Account for Overflow
        OV <<= 1 ;                     // Multiply the original data by 2
        OV &= 0xfefefefe ;             // Clear the MSBs
        OV ^= OF ;                     // Add in the overflow mask

        // Process bit 4
        T2 ^= OV;                      // Account for bit 4 of constant 9d
        T4  = OV;                      // Account for bit 4 of constant 30
        T5 ^= OV;                      // Account for bit 4 of constant 92
        T6 ^= OV;                      // Account for bit 4 of constant 15
        T8 ^= OV;                      // Account for bit 4 of constant f8
        T10 ^= OV;                      // Account for bit 4 of constant f1
        T12 ^= OV;                      // Account for bit 4 of constant d5
        T13 ^= OV;                      // Account for bit 4 of constant b6
        MSK = OV & 0x80808080 ;        // Create a mask of the MSBs
        OF = (MSK >> 7) * 0x1d ;       // Account for Overflow
        OV <<= 1 ;                     // Multiply the original data by 2
        OV &= 0xfefefefe ;             // Clear the MSBs
        OV ^= OF ;                     // Add in the overflow mask

        // Process bit 5
        T4 ^= OV;                      // Account for bit 5 of constant 30
        T7 ^= OV;                      // Account for bit 5 of constant 65
        T8 ^= OV;                      // Account for bit 5 of constant f8
        T9  = OV;                      // Account for bit 5 of constant a0
        T10 ^= OV;                      // Account for bit 5 of constant f1
        T11 ^= OV;                      // Account for bit 5 of constant 2c
        T13 ^= OV;                      // Account for bit 5 of constant b6
        MSK = OV & 0x80808080 ;        // Create a mask of the MSBs
        OF = (MSK >> 7) * 0x1d ;       // Account for Overflow
        OV <<= 1 ;                     // Multiply the original data by 2
        OV &= 0xfefefefe ;             // Clear the MSBs
        OV ^= OF ;                     // Add in the overflow mask

        // Process bit 6
        T1 ^= OV;                      // Account for bit 6 of constant c8
        T7 ^= OV;                      // Account for bit 6 of constant 65
        T8 ^= OV;                      // Account for bit 6 of constant f8
        T10 ^= OV;                      // Account for bit 6 of constant f1
        T12 ^= OV;                      // Account for bit 6 of constant d5
        MSK = OV & 0x80808080 ;        // Create a mask of the MSBs
        OF = (MSK >> 7) * 0x1d ;       // Account for Overflow
        OV <<= 1 ;                     // Multiply the original data by 2
        OV &= 0xfefefefe ;             // Clear the MSBs
        OV ^= OF ;                     // Add in the overflow mask

        // Process bit 7
        T1 ^= OV;                      // Account for bit 7 of constant c8
        T2 ^= OV;                      // Account for bit 7 of constant 9d
        T5 ^= OV;                      // Account for bit 7 of constant 92
        T8 ^= OV;                      // Account for bit 7 of constant f8
        T9 ^= OV;                      // Account for bit 7 of constant a0
        T10 ^= OV;                      // Account for bit 7 of constant f1
        T12 ^= OV;                      // Account for bit 7 of constant d5
        T13 ^= OV;                      // Account for bit 7 of constant b6

        // Now process the LFSR
        P0 = P1 ^ T0 ;
        P1 = P2 ^ T1 ;
        P2 = P3 ^ T2 ;
        P3 = P4 ^ T3 ;
        P4 = P5 ^ T4 ;
        P5 = P6 ^ T5 ;
        P6 = P7 ^ T6 ;
        P7 = P8 ^ T7 ;
        P8 = P9 ^ T8 ;
        P9 = P10 ^ T9 ;
        P10 = P11 ^ T10 ;
        P11 = P12 ^ T11 ;
        P12 = P13 ^ T12 ;
        P13 = P14 ^ T13 ;
        P14 = P15 ^ T12 ;
        P15 = P16 ^ T11 ;
        P16 = P17 ^ T10 ;
        P17 = P18 ^ T9 ;
        P18 = P19 ^ T8 ;
        P19 = P20 ^ T7 ;
        P20 = P21 ^ T6 ;
        P21 = P22 ^ T5 ;
        P22 = P23 ^ T4 ;
        P23 = P24 ^ T3 ;
        P24 = P25 ^ T2 ;
        P25 = P26 ^ T1 ;
        P26 = P27 ^ T0 ;
        P27 = OOV ;
    }

    if ( decoder == 0 )
    {
        // Now store computed check symbols back to memory
        loc = CWP[k+0] + i;
        *loc = P0 ;
        loc = CWP[k+1] + i;
        *loc = P1 ;
        loc = CWP[k+2] + i;
        *loc = P2 ;
        loc = CWP[k+3] + i;
        *loc = P3 ;
        loc = CWP[k+4] + i;
        *loc = P4 ;
        loc = CWP[k+5] + i;
        *loc = P5 ;
        loc = CWP[k+6] + i;
        *loc = P6 ;
        loc = CWP[k+7] + i;
        *loc = P7 ;
        loc = CWP[k+8] + i;
        *loc = P8 ;
        loc = CWP[k+9] + i;
        *loc = P9 ;
        loc = CWP[k+10] + i;
        *loc = P10 ;
        loc = CWP[k+11] + i;
        *loc = P11 ;
        loc = CWP[k+12] + i;
        *loc = P12 ;
        loc = CWP[k+13] + i;
        *loc = P13 ;
        loc = CWP[k+14] + i;
        *loc = P14 ;
        loc = CWP[k+15] + i;
        *loc = P15 ;
        loc = CWP[k+16] + i;
        *loc = P16 ;
        loc = CWP[k+17] + i;
        *loc = P17 ;
        loc = CWP[k+18] + i;
        *loc = P18 ;
        loc = CWP[k+19] + i;
        *loc = P19 ;
        loc = CWP[k+20] + i;
        *loc = P20 ;
        loc = CWP[k+21] + i;
        *loc = P21 ;
        loc = CWP[k+22] + i;
        *loc = P22 ;
        loc = CWP[k+23] + i;
        *loc = P23 ;
        loc = CWP[k+24] + i;
        *loc = P24 ;
        loc = CWP[k+25] + i;
        *loc = P25 ;
        loc = CWP[k+26] + i;
        *loc = P26 ;
        loc = CWP[k+27] + i;
        *loc = P27 ;
    }
    else
    {
        // Now check for parity equal to zero
        T0 = P0;
        T0 |= P1;
        T0 |= P2;
        T0 |= P3;
        T0 |= P4;
        T0 |= P5;
        T0 |= P6;
        T0 |= P7;
        T0 |= P8;
        T0 |= P9;
        T0 |= P10;
        T0 |= P11;
        T0 |= P12;
        T0 |= P13;
        T0 |= P14;
        T0 |= P15;
        T0 |= P16;
        T0 |= P17;
        T0 |= P18;
        T0 |= P19;
        T0 |= P20;
        T0 |= P21;
        T0 |= P22;
        T0 |= P23;
        T0 |= P24;
        T0 |= P25;
        T0 |= P26;
        T0 |= P27;
        if (T0 != 0)
        {
            unsigned char Sr[28], S[28], errLocs[28], errMags[28] ;
            unsigned char Lambda[28+1], B[28+1], T[28+1], Omega[28+1] ;
            
            if ( T0 & (0xff << 0) )
            {
                atomicAdd(&error_count, 1);
                Sr[0] = (P0>>0) & 0xff ;
                Sr[1] = (P1>>0) & 0xff ;
                Sr[2] = (P2>>0) & 0xff ;
                Sr[3] = (P3>>0) & 0xff ;
                Sr[4] = (P4>>0) & 0xff ;
                Sr[5] = (P5>>0) & 0xff ;
                Sr[6] = (P6>>0) & 0xff ;
                Sr[7] = (P7>>0) & 0xff ;
                Sr[8] = (P8>>0) & 0xff ;
                Sr[9] = (P9>>0) & 0xff ;
                Sr[10] = (P10>>0) & 0xff ;
                Sr[11] = (P11>>0) & 0xff ;
                Sr[12] = (P12>>0) & 0xff ;
                Sr[13] = (P13>>0) & 0xff ;
                Sr[14] = (P14>>0) & 0xff ;
                Sr[15] = (P15>>0) & 0xff ;
                Sr[16] = (P16>>0) & 0xff ;
                Sr[17] = (P17>>0) & 0xff ;
                Sr[18] = (P18>>0) & 0xff ;
                Sr[19] = (P19>>0) & 0xff ;
                Sr[20] = (P20>>0) & 0xff ;
                Sr[21] = (P21>>0) & 0xff ;
                Sr[22] = (P22>>0) & 0xff ;
                Sr[23] = (P23>>0) & 0xff ;
                Sr[24] = (P24>>0) & 0xff ;
                Sr[25] = (P25>>0) & 0xff ;
                Sr[26] = (P26>>0) & 0xff ;
                Sr[27] = (P27>>0) & 0xff ;
                rs_decode_palindromic_CUDA(Sr, S, 28, k, errLocs, errMags,
                                           0+(i<<2), Lambda, B, T, Omega) ;
            }
            if ( T0 & (0xff << 8) )
            {
                atomicAdd(&error_count, 1);
                Sr[0] = (P0>>8) & 0xff ;
                Sr[1] = (P1>>8) & 0xff ;
                Sr[2] = (P2>>8) & 0xff ;
                Sr[3] = (P3>>8) & 0xff ;
                Sr[4] = (P4>>8) & 0xff ;
                Sr[5] = (P5>>8) & 0xff ;
                Sr[6] = (P6>>8) & 0xff ;
                Sr[7] = (P7>>8) & 0xff ;
                Sr[8] = (P8>>8) & 0xff ;
                Sr[9] = (P9>>8) & 0xff ;
                Sr[10] = (P10>>8) & 0xff ;
                Sr[11] = (P11>>8) & 0xff ;
                Sr[12] = (P12>>8) & 0xff ;
                Sr[13] = (P13>>8) & 0xff ;
                Sr[14] = (P14>>8) & 0xff ;
                Sr[15] = (P15>>8) & 0xff ;
                Sr[16] = (P16>>8) & 0xff ;
                Sr[17] = (P17>>8) & 0xff ;
                Sr[18] = (P18>>8) & 0xff ;
                Sr[19] = (P19>>8) & 0xff ;
                Sr[20] = (P20>>8) & 0xff ;
                Sr[21] = (P21>>8) & 0xff ;
                Sr[22] = (P22>>8) & 0xff ;
                Sr[23] = (P23>>8) & 0xff ;
                Sr[24] = (P24>>8) & 0xff ;
                Sr[25] = (P25>>8) & 0xff ;
                Sr[26] = (P26>>8) & 0xff ;
                Sr[27] = (P27>>8) & 0xff ;
                rs_decode_palindromic_CUDA(Sr, S, 28, k, errLocs, errMags,
                                           1+(i<<2), Lambda, B, T, Omega) ;
            }
            if ( T0 & (0xff << 16) )
            {
                atomicAdd(&error_count, 1);
                Sr[0] = (P0>>16) & 0xff ;
                Sr[1] = (P1>>16) & 0xff ;
                Sr[2] = (P2>>16) & 0xff ;
                Sr[3] = (P3>>16) & 0xff ;
                Sr[4] = (P4>>16) & 0xff ;
                Sr[5] = (P5>>16) & 0xff ;
                Sr[6] = (P6>>16) & 0xff ;
                Sr[7] = (P7>>16) & 0xff ;
                Sr[8] = (P8>>16) & 0xff ;
                Sr[9] = (P9>>16) & 0xff ;
                Sr[10] = (P10>>16) & 0xff ;
                Sr[11] = (P11>>16) & 0xff ;
                Sr[12] = (P12>>16) & 0xff ;
                Sr[13] = (P13>>16) & 0xff ;
                Sr[14] = (P14>>16) & 0xff ;
                Sr[15] = (P15>>16) & 0xff ;
                Sr[16] = (P16>>16) & 0xff ;
                Sr[17] = (P17>>16) & 0xff ;
                Sr[18] = (P18>>16) & 0xff ;
                Sr[19] = (P19>>16) & 0xff ;
                Sr[20] = (P20>>16) & 0xff ;
                Sr[21] = (P21>>16) & 0xff ;
                Sr[22] = (P22>>16) & 0xff ;
                Sr[23] = (P23>>16) & 0xff ;
                Sr[24] = (P24>>16) & 0xff ;
                Sr[25] = (P25>>16) & 0xff ;
                Sr[26] = (P26>>16) & 0xff ;
                Sr[27] = (P27>>16) & 0xff ;
                rs_decode_palindromic_CUDA(Sr, S, 28, k, errLocs, errMags,
                                           2+(i<<2), Lambda, B, T, Omega) ;
            }
            if ( T0 & (0xff << 24) )
            {
                atomicAdd(&error_count, 1);
                Sr[0] = (P0>>24) & 0xff ;
                Sr[1] = (P1>>24) & 0xff ;
                Sr[2] = (P2>>24) & 0xff ;
                Sr[3] = (P3>>24) & 0xff ;
                Sr[4] = (P4>>24) & 0xff ;
                Sr[5] = (P5>>24) & 0xff ;
                Sr[6] = (P6>>24) & 0xff ;
                Sr[7] = (P7>>24) & 0xff ;
                Sr[8] = (P8>>24) & 0xff ;
                Sr[9] = (P9>>24) & 0xff ;
                Sr[10] = (P10>>24) & 0xff ;
                Sr[11] = (P11>>24) & 0xff ;
                Sr[12] = (P12>>24) & 0xff ;
                Sr[13] = (P13>>24) & 0xff ;
                Sr[14] = (P14>>24) & 0xff ;
                Sr[15] = (P15>>24) & 0xff ;
                Sr[16] = (P16>>24) & 0xff ;
                Sr[17] = (P17>>24) & 0xff ;
                Sr[18] = (P18>>24) & 0xff ;
                Sr[19] = (P19>>24) & 0xff ;
                Sr[20] = (P20>>24) & 0xff ;
                Sr[21] = (P21>>24) & 0xff ;
                Sr[22] = (P22>>24) & 0xff ;
                Sr[23] = (P23>>24) & 0xff ;
                Sr[24] = (P24>>24) & 0xff ;
                Sr[25] = (P25>>24) & 0xff ;
                Sr[26] = (P26>>24) & 0xff ;
                Sr[27] = (P27>>24) & 0xff ;
                rs_decode_palindromic_CUDA(Sr, S, 28, k, errLocs, errMags,
                                           3+(i<<2), Lambda, B, T, Omega) ;
            }
        }
    }
}

// ----------------------------------------------
// Parallel LFSR Sequencer for P = 29 Codewords
// ----------------------------------------------
__global__ void ParallelLFSRSequencer29_CUDA(int k, int decoder)
{
    // Find out place in the grid of threads
    int i = blockIdx.x * blockDim.x + threadIdx.x;

    // Pointer to develop original data address
    unsigned int* loc ;

    // Create the parity values, initialize to zero
    unsigned int P0 = 0 ;
    unsigned int P1 = 0 ;
    unsigned int P2 = 0 ;
    unsigned int P3 = 0 ;
    unsigned int P4 = 0 ;
    unsigned int P5 = 0 ;
    unsigned int P6 = 0 ;
    unsigned int P7 = 0 ;
    unsigned int P8 = 0 ;
    unsigned int P9 = 0 ;
    unsigned int P10 = 0 ;
    unsigned int P11 = 0 ;
    unsigned int P12 = 0 ;
    unsigned int P13 = 0 ;
    unsigned int P14 = 0 ;
    unsigned int P15 = 0 ;
    unsigned int P16 = 0 ;
    unsigned int P17 = 0 ;
    unsigned int P18 = 0 ;
    unsigned int P19 = 0 ;
    unsigned int P20 = 0 ;
    unsigned int P21 = 0 ;
    unsigned int P22 = 0 ;
    unsigned int P23 = 0 ;
    unsigned int P24 = 0 ;
    unsigned int P25 = 0 ;
    unsigned int P26 = 0 ;
    unsigned int P27 = 0 ;
    unsigned int P28 = 0 ;

    // Temporary values for self reciprocol polynomial
    unsigned int T0;
    unsigned int T1;
    unsigned int T2;
    unsigned int T3;
    unsigned int T4;
    unsigned int T5;
    unsigned int T6;
    unsigned int T7;
    unsigned int T8;
    unsigned int T9;
    unsigned int T10;
    unsigned int T11;
    unsigned int T12;
    unsigned int T13;

    // Overflow value for LFSR and carry mask
    unsigned int OV, MSK, OF, OOV ;

    // Adjust K if we are decoding
    k += decoder ? 29 : 0 ;

    // Loop through each symbol
    for ( int sym = 0 ; sym < k ; sym ++ )
    {
        loc = CWP[sym] + i;
        OV = *loc ^ P0 ;
        OOV = OV ;

        // Process bit 0
        T2  = OV;                      // Account for bit 0 of constant 15
        T3  = OV;                      // Account for bit 0 of constant 3d
        T4  = OV;                      // Account for bit 0 of constant e7
        T5  = OV;                      // Account for bit 0 of constant a9
        T8  = OV;                      // Account for bit 0 of constant c5
        T10  = OV;                      // Account for bit 0 of constant 8b
        T11  = OV;                      // Account for bit 0 of constant 5b
        T13  = OV;                      // Account for bit 0 of constant fb
        MSK = OV & 0x80808080 ;        // Create a mask of the MSBs
        OF = (MSK >> 7) * 0x1d ;       // Account for Overflow
        OV <<= 1 ;                     // Multiply the original data by 2
        OV &= 0xfefefefe ;             // Clear the MSBs
        OV ^= OF ;                     // Add in the overflow mask

        // Process bit 1
        T0  = OV;                      // Account for bit 1 of constant 2a
        T4 ^= OV;                      // Account for bit 1 of constant e7
        T6  = OV;                      // Account for bit 1 of constant b2
        T7  = OV;                      // Account for bit 1 of constant 16
        T9  = OV;                      // Account for bit 1 of constant 42
        T10 ^= OV;                      // Account for bit 1 of constant 8b
        T11 ^= OV;                      // Account for bit 1 of constant 5b
        T12  = OV;                      // Account for bit 1 of constant d2
        T13 ^= OV;                      // Account for bit 1 of constant fb
        MSK = OV & 0x80808080 ;        // Create a mask of the MSBs
        OF = (MSK >> 7) * 0x1d ;       // Account for Overflow
        OV <<= 1 ;                     // Multiply the original data by 2
        OV &= 0xfefefefe ;             // Clear the MSBs
        OV ^= OF ;                     // Add in the overflow mask

        // Process bit 2
        T2 ^= OV;                      // Account for bit 2 of constant 15
        T3 ^= OV;                      // Account for bit 2 of constant 3d
        T4 ^= OV;                      // Account for bit 2 of constant e7
        T7 ^= OV;                      // Account for bit 2 of constant 16
        T8 ^= OV;                      // Account for bit 2 of constant c5
        MSK = OV & 0x80808080 ;        // Create a mask of the MSBs
        OF = (MSK >> 7) * 0x1d ;       // Account for Overflow
        OV <<= 1 ;                     // Multiply the original data by 2
        OV &= 0xfefefefe ;             // Clear the MSBs
        OV ^= OF ;                     // Add in the overflow mask

        // Process bit 3
        T0 ^= OV;                      // Account for bit 3 of constant 2a
        T1  = OV;                      // Account for bit 3 of constant 98
        T3 ^= OV;                      // Account for bit 3 of constant 3d
        T5 ^= OV;                      // Account for bit 3 of constant a9
        T10 ^= OV;                      // Account for bit 3 of constant 8b
        T11 ^= OV;                      // Account for bit 3 of constant 5b
        T13 ^= OV;                      // Account for bit 3 of constant fb
        MSK = OV & 0x80808080 ;        // Create a mask of the MSBs
        OF = (MSK >> 7) * 0x1d ;       // Account for Overflow
        OV <<= 1 ;                     // Multiply the original data by 2
        OV &= 0xfefefefe ;             // Clear the MSBs
        OV ^= OF ;                     // Add in the overflow mask

        // Process bit 4
        T1 ^= OV;                      // Account for bit 4 of constant 98
        T2 ^= OV;                      // Account for bit 4 of constant 15
        T3 ^= OV;                      // Account for bit 4 of constant 3d
        T6 ^= OV;                      // Account for bit 4 of constant b2
        T7 ^= OV;                      // Account for bit 4 of constant 16
        T11 ^= OV;                      // Account for bit 4 of constant 5b
        T12 ^= OV;                      // Account for bit 4 of constant d2
        T13 ^= OV;                      // Account for bit 4 of constant fb
        MSK = OV & 0x80808080 ;        // Create a mask of the MSBs
        OF = (MSK >> 7) * 0x1d ;       // Account for Overflow
        OV <<= 1 ;                     // Multiply the original data by 2
        OV &= 0xfefefefe ;             // Clear the MSBs
        OV ^= OF ;                     // Add in the overflow mask

        // Process bit 5
        T0 ^= OV;                      // Account for bit 5 of constant 2a
        T3 ^= OV;                      // Account for bit 5 of constant 3d
        T4 ^= OV;                      // Account for bit 5 of constant e7
        T5 ^= OV;                      // Account for bit 5 of constant a9
        T6 ^= OV;                      // Account for bit 5 of constant b2
        T13 ^= OV;                      // Account for bit 5 of constant fb
        MSK = OV & 0x80808080 ;        // Create a mask of the MSBs
        OF = (MSK >> 7) * 0x1d ;       // Account for Overflow
        OV <<= 1 ;                     // Multiply the original data by 2
        OV &= 0xfefefefe ;             // Clear the MSBs
        OV ^= OF ;                     // Add in the overflow mask

        // Process bit 6
        T4 ^= OV;                      // Account for bit 6 of constant e7
        T8 ^= OV;                      // Account for bit 6 of constant c5
        T9 ^= OV;                      // Account for bit 6 of constant 42
        T11 ^= OV;                      // Account for bit 6 of constant 5b
        T12 ^= OV;                      // Account for bit 6 of constant d2
        T13 ^= OV;                      // Account for bit 6 of constant fb
        MSK = OV & 0x80808080 ;        // Create a mask of the MSBs
        OF = (MSK >> 7) * 0x1d ;       // Account for Overflow
        OV <<= 1 ;                     // Multiply the original data by 2
        OV &= 0xfefefefe ;             // Clear the MSBs
        OV ^= OF ;                     // Add in the overflow mask

        // Process bit 7
        T1 ^= OV;                      // Account for bit 7 of constant 98
        T4 ^= OV;                      // Account for bit 7 of constant e7
        T5 ^= OV;                      // Account for bit 7 of constant a9
        T6 ^= OV;                      // Account for bit 7 of constant b2
        T8 ^= OV;                      // Account for bit 7 of constant c5
        T10 ^= OV;                      // Account for bit 7 of constant 8b
        T12 ^= OV;                      // Account for bit 7 of constant d2
        T13 ^= OV;                      // Account for bit 7 of constant fb

        // Now process the LFSR
        P0 = P1 ^ T0 ;
        P1 = P2 ^ T1 ;
        P2 = P3 ^ T2 ;
        P3 = P4 ^ T3 ;
        P4 = P5 ^ T4 ;
        P5 = P6 ^ T5 ;
        P6 = P7 ^ T6 ;
        P7 = P8 ^ T7 ;
        P8 = P9 ^ T8 ;
        P9 = P10 ^ T9 ;
        P10 = P11 ^ T10 ;
        P11 = P12 ^ T11 ;
        P12 = P13 ^ T12 ;
        P13 = P14 ^ T13 ;
        P14 = P15 ^ T13 ;
        P15 = P16 ^ T12 ;
        P16 = P17 ^ T11 ;
        P17 = P18 ^ T10 ;
        P18 = P19 ^ T9 ;
        P19 = P20 ^ T8 ;
        P20 = P21 ^ T7 ;
        P21 = P22 ^ T6 ;
        P22 = P23 ^ T5 ;
        P23 = P24 ^ T4 ;
        P24 = P25 ^ T3 ;
        P25 = P26 ^ T2 ;
        P26 = P27 ^ T1 ;
        P27 = P28 ^ T0 ;
        P28 = OOV ;
    }

    if ( decoder == 0 )
    {
        // Now store computed check symbols back to memory
        loc = CWP[k+0] + i;
        *loc = P0 ;
        loc = CWP[k+1] + i;
        *loc = P1 ;
        loc = CWP[k+2] + i;
        *loc = P2 ;
        loc = CWP[k+3] + i;
        *loc = P3 ;
        loc = CWP[k+4] + i;
        *loc = P4 ;
        loc = CWP[k+5] + i;
        *loc = P5 ;
        loc = CWP[k+6] + i;
        *loc = P6 ;
        loc = CWP[k+7] + i;
        *loc = P7 ;
        loc = CWP[k+8] + i;
        *loc = P8 ;
        loc = CWP[k+9] + i;
        *loc = P9 ;
        loc = CWP[k+10] + i;
        *loc = P10 ;
        loc = CWP[k+11] + i;
        *loc = P11 ;
        loc = CWP[k+12] + i;
        *loc = P12 ;
        loc = CWP[k+13] + i;
        *loc = P13 ;
        loc = CWP[k+14] + i;
        *loc = P14 ;
        loc = CWP[k+15] + i;
        *loc = P15 ;
        loc = CWP[k+16] + i;
        *loc = P16 ;
        loc = CWP[k+17] + i;
        *loc = P17 ;
        loc = CWP[k+18] + i;
        *loc = P18 ;
        loc = CWP[k+19] + i;
        *loc = P19 ;
        loc = CWP[k+20] + i;
        *loc = P20 ;
        loc = CWP[k+21] + i;
        *loc = P21 ;
        loc = CWP[k+22] + i;
        *loc = P22 ;
        loc = CWP[k+23] + i;
        *loc = P23 ;
        loc = CWP[k+24] + i;
        *loc = P24 ;
        loc = CWP[k+25] + i;
        *loc = P25 ;
        loc = CWP[k+26] + i;
        *loc = P26 ;
        loc = CWP[k+27] + i;
        *loc = P27 ;
        loc = CWP[k+28] + i;
        *loc = P28 ;
    }
    else
    {
        // Now check for parity equal to zero
        T0 = P0;
        T0 |= P1;
        T0 |= P2;
        T0 |= P3;
        T0 |= P4;
        T0 |= P5;
        T0 |= P6;
        T0 |= P7;
        T0 |= P8;
        T0 |= P9;
        T0 |= P10;
        T0 |= P11;
        T0 |= P12;
        T0 |= P13;
        T0 |= P14;
        T0 |= P15;
        T0 |= P16;
        T0 |= P17;
        T0 |= P18;
        T0 |= P19;
        T0 |= P20;
        T0 |= P21;
        T0 |= P22;
        T0 |= P23;
        T0 |= P24;
        T0 |= P25;
        T0 |= P26;
        T0 |= P27;
        T0 |= P28;
        if (T0 != 0)
        {
            unsigned char Sr[29], S[29], errLocs[29], errMags[29] ;
            unsigned char Lambda[29+1], B[29+1], T[29+1], Omega[29+1] ;
            
            if ( T0 & (0xff << 0) )
            {
                atomicAdd(&error_count, 1);
                Sr[0] = (P0>>0) & 0xff ;
                Sr[1] = (P1>>0) & 0xff ;
                Sr[2] = (P2>>0) & 0xff ;
                Sr[3] = (P3>>0) & 0xff ;
                Sr[4] = (P4>>0) & 0xff ;
                Sr[5] = (P5>>0) & 0xff ;
                Sr[6] = (P6>>0) & 0xff ;
                Sr[7] = (P7>>0) & 0xff ;
                Sr[8] = (P8>>0) & 0xff ;
                Sr[9] = (P9>>0) & 0xff ;
                Sr[10] = (P10>>0) & 0xff ;
                Sr[11] = (P11>>0) & 0xff ;
                Sr[12] = (P12>>0) & 0xff ;
                Sr[13] = (P13>>0) & 0xff ;
                Sr[14] = (P14>>0) & 0xff ;
                Sr[15] = (P15>>0) & 0xff ;
                Sr[16] = (P16>>0) & 0xff ;
                Sr[17] = (P17>>0) & 0xff ;
                Sr[18] = (P18>>0) & 0xff ;
                Sr[19] = (P19>>0) & 0xff ;
                Sr[20] = (P20>>0) & 0xff ;
                Sr[21] = (P21>>0) & 0xff ;
                Sr[22] = (P22>>0) & 0xff ;
                Sr[23] = (P23>>0) & 0xff ;
                Sr[24] = (P24>>0) & 0xff ;
                Sr[25] = (P25>>0) & 0xff ;
                Sr[26] = (P26>>0) & 0xff ;
                Sr[27] = (P27>>0) & 0xff ;
                Sr[28] = (P28>>0) & 0xff ;
                rs_decode_palindromic_CUDA(Sr, S, 29, k, errLocs, errMags,
                                           0+(i<<2), Lambda, B, T, Omega) ;
            }
            if ( T0 & (0xff << 8) )
            {
                atomicAdd(&error_count, 1);
                Sr[0] = (P0>>8) & 0xff ;
                Sr[1] = (P1>>8) & 0xff ;
                Sr[2] = (P2>>8) & 0xff ;
                Sr[3] = (P3>>8) & 0xff ;
                Sr[4] = (P4>>8) & 0xff ;
                Sr[5] = (P5>>8) & 0xff ;
                Sr[6] = (P6>>8) & 0xff ;
                Sr[7] = (P7>>8) & 0xff ;
                Sr[8] = (P8>>8) & 0xff ;
                Sr[9] = (P9>>8) & 0xff ;
                Sr[10] = (P10>>8) & 0xff ;
                Sr[11] = (P11>>8) & 0xff ;
                Sr[12] = (P12>>8) & 0xff ;
                Sr[13] = (P13>>8) & 0xff ;
                Sr[14] = (P14>>8) & 0xff ;
                Sr[15] = (P15>>8) & 0xff ;
                Sr[16] = (P16>>8) & 0xff ;
                Sr[17] = (P17>>8) & 0xff ;
                Sr[18] = (P18>>8) & 0xff ;
                Sr[19] = (P19>>8) & 0xff ;
                Sr[20] = (P20>>8) & 0xff ;
                Sr[21] = (P21>>8) & 0xff ;
                Sr[22] = (P22>>8) & 0xff ;
                Sr[23] = (P23>>8) & 0xff ;
                Sr[24] = (P24>>8) & 0xff ;
                Sr[25] = (P25>>8) & 0xff ;
                Sr[26] = (P26>>8) & 0xff ;
                Sr[27] = (P27>>8) & 0xff ;
                Sr[28] = (P28>>8) & 0xff ;
                rs_decode_palindromic_CUDA(Sr, S, 29, k, errLocs, errMags,
                                           1+(i<<2), Lambda, B, T, Omega) ;
            }
            if ( T0 & (0xff << 16) )
            {
                atomicAdd(&error_count, 1);
                Sr[0] = (P0>>16) & 0xff ;
                Sr[1] = (P1>>16) & 0xff ;
                Sr[2] = (P2>>16) & 0xff ;
                Sr[3] = (P3>>16) & 0xff ;
                Sr[4] = (P4>>16) & 0xff ;
                Sr[5] = (P5>>16) & 0xff ;
                Sr[6] = (P6>>16) & 0xff ;
                Sr[7] = (P7>>16) & 0xff ;
                Sr[8] = (P8>>16) & 0xff ;
                Sr[9] = (P9>>16) & 0xff ;
                Sr[10] = (P10>>16) & 0xff ;
                Sr[11] = (P11>>16) & 0xff ;
                Sr[12] = (P12>>16) & 0xff ;
                Sr[13] = (P13>>16) & 0xff ;
                Sr[14] = (P14>>16) & 0xff ;
                Sr[15] = (P15>>16) & 0xff ;
                Sr[16] = (P16>>16) & 0xff ;
                Sr[17] = (P17>>16) & 0xff ;
                Sr[18] = (P18>>16) & 0xff ;
                Sr[19] = (P19>>16) & 0xff ;
                Sr[20] = (P20>>16) & 0xff ;
                Sr[21] = (P21>>16) & 0xff ;
                Sr[22] = (P22>>16) & 0xff ;
                Sr[23] = (P23>>16) & 0xff ;
                Sr[24] = (P24>>16) & 0xff ;
                Sr[25] = (P25>>16) & 0xff ;
                Sr[26] = (P26>>16) & 0xff ;
                Sr[27] = (P27>>16) & 0xff ;
                Sr[28] = (P28>>16) & 0xff ;
                rs_decode_palindromic_CUDA(Sr, S, 29, k, errLocs, errMags,
                                           2+(i<<2), Lambda, B, T, Omega) ;
            }
            if ( T0 & (0xff << 24) )
            {
                atomicAdd(&error_count, 1);
                Sr[0] = (P0>>24) & 0xff ;
                Sr[1] = (P1>>24) & 0xff ;
                Sr[2] = (P2>>24) & 0xff ;
                Sr[3] = (P3>>24) & 0xff ;
                Sr[4] = (P4>>24) & 0xff ;
                Sr[5] = (P5>>24) & 0xff ;
                Sr[6] = (P6>>24) & 0xff ;
                Sr[7] = (P7>>24) & 0xff ;
                Sr[8] = (P8>>24) & 0xff ;
                Sr[9] = (P9>>24) & 0xff ;
                Sr[10] = (P10>>24) & 0xff ;
                Sr[11] = (P11>>24) & 0xff ;
                Sr[12] = (P12>>24) & 0xff ;
                Sr[13] = (P13>>24) & 0xff ;
                Sr[14] = (P14>>24) & 0xff ;
                Sr[15] = (P15>>24) & 0xff ;
                Sr[16] = (P16>>24) & 0xff ;
                Sr[17] = (P17>>24) & 0xff ;
                Sr[18] = (P18>>24) & 0xff ;
                Sr[19] = (P19>>24) & 0xff ;
                Sr[20] = (P20>>24) & 0xff ;
                Sr[21] = (P21>>24) & 0xff ;
                Sr[22] = (P22>>24) & 0xff ;
                Sr[23] = (P23>>24) & 0xff ;
                Sr[24] = (P24>>24) & 0xff ;
                Sr[25] = (P25>>24) & 0xff ;
                Sr[26] = (P26>>24) & 0xff ;
                Sr[27] = (P27>>24) & 0xff ;
                Sr[28] = (P28>>24) & 0xff ;
                rs_decode_palindromic_CUDA(Sr, S, 29, k, errLocs, errMags,
                                           3+(i<<2), Lambda, B, T, Omega) ;
            }
        }
    }
}

// ----------------------------------------------
// Parallel LFSR Sequencer for P = 30 Codewords
// ----------------------------------------------
__global__ void ParallelLFSRSequencer30_CUDA(int k, int decoder)
{
    // Find out place in the grid of threads
    int i = blockIdx.x * blockDim.x + threadIdx.x;

    // Pointer to develop original data address
    unsigned int* loc ;

    // Create the parity values, initialize to zero
    unsigned int P0 = 0 ;
    unsigned int P1 = 0 ;
    unsigned int P2 = 0 ;
    unsigned int P3 = 0 ;
    unsigned int P4 = 0 ;
    unsigned int P5 = 0 ;
    unsigned int P6 = 0 ;
    unsigned int P7 = 0 ;
    unsigned int P8 = 0 ;
    unsigned int P9 = 0 ;
    unsigned int P10 = 0 ;
    unsigned int P11 = 0 ;
    unsigned int P12 = 0 ;
    unsigned int P13 = 0 ;
    unsigned int P14 = 0 ;
    unsigned int P15 = 0 ;
    unsigned int P16 = 0 ;
    unsigned int P17 = 0 ;
    unsigned int P18 = 0 ;
    unsigned int P19 = 0 ;
    unsigned int P20 = 0 ;
    unsigned int P21 = 0 ;
    unsigned int P22 = 0 ;
    unsigned int P23 = 0 ;
    unsigned int P24 = 0 ;
    unsigned int P25 = 0 ;
    unsigned int P26 = 0 ;
    unsigned int P27 = 0 ;
    unsigned int P28 = 0 ;
    unsigned int P29 = 0 ;

    // Temporary values for self reciprocol polynomial
    unsigned int T0;
    unsigned int T1;
    unsigned int T2;
    unsigned int T3;
    unsigned int T4;
    unsigned int T5;
    unsigned int T6;
    unsigned int T7;
    unsigned int T8;
    unsigned int T9;
    unsigned int T10;
    unsigned int T11;
    unsigned int T12;
    unsigned int T13;
    unsigned int T14;

    // Overflow value for LFSR and carry mask
    unsigned int OV, MSK, OF, OOV ;

    // Adjust K if we are decoding
    k += decoder ? 30 : 0 ;

    // Loop through each symbol
    for ( int sym = 0 ; sym < k ; sym ++ )
    {
        loc = CWP[sym] + i;
        OV = *loc ^ P0 ;
        OOV = OV ;

        // Process bit 0
        T0  = OV;                      // Account for bit 0 of constant 39
        T3  = OV;                      // Account for bit 0 of constant 71
        T6  = OV;                      // Account for bit 0 of constant a7
        T7  = OV;                      // Account for bit 0 of constant 61
        T8  = OV;                      // Account for bit 0 of constant 23
        T9  = OV;                      // Account for bit 0 of constant b7
        T12  = OV;                      // Account for bit 0 of constant ab
        MSK = OV & 0x80808080 ;        // Create a mask of the MSBs
        OF = (MSK >> 7) * 0x1d ;       // Account for Overflow
        OV <<= 1 ;                     // Multiply the original data by 2
        OV &= 0xfefefefe ;             // Clear the MSBs
        OV ^= OF ;                     // Add in the overflow mask

        // Process bit 1
        T2  = OV;                      // Account for bit 1 of constant 7a
        T5  = OV;                      // Account for bit 1 of constant e
        T6 ^= OV;                      // Account for bit 1 of constant a7
        T8 ^= OV;                      // Account for bit 1 of constant 23
        T9 ^= OV;                      // Account for bit 1 of constant b7
        T11  = OV;                      // Account for bit 1 of constant 7e
        T12 ^= OV;                      // Account for bit 1 of constant ab
        T13  = OV;                      // Account for bit 1 of constant 52
        MSK = OV & 0x80808080 ;        // Create a mask of the MSBs
        OF = (MSK >> 7) * 0x1d ;       // Account for Overflow
        OV <<= 1 ;                     // Multiply the original data by 2
        OV &= 0xfefefefe ;             // Clear the MSBs
        OV ^= OF ;                     // Add in the overflow mask

        // Process bit 2
        T4  = OV;                      // Account for bit 2 of constant 4c
        T5 ^= OV;                      // Account for bit 2 of constant e
        T6 ^= OV;                      // Account for bit 2 of constant a7
        T9 ^= OV;                      // Account for bit 2 of constant b7
        T11 ^= OV;                      // Account for bit 2 of constant 7e
        MSK = OV & 0x80808080 ;        // Create a mask of the MSBs
        OF = (MSK >> 7) * 0x1d ;       // Account for Overflow
        OV <<= 1 ;                     // Multiply the original data by 2
        OV &= 0xfefefefe ;             // Clear the MSBs
        OV ^= OF ;                     // Add in the overflow mask

        // Process bit 3
        T0 ^= OV;                      // Account for bit 3 of constant 39
        T1  = OV;                      // Account for bit 3 of constant a8
        T2 ^= OV;                      // Account for bit 3 of constant 7a
        T4 ^= OV;                      // Account for bit 3 of constant 4c
        T5 ^= OV;                      // Account for bit 3 of constant e
        T10  = OV;                      // Account for bit 3 of constant 78
        T11 ^= OV;                      // Account for bit 3 of constant 7e
        T12 ^= OV;                      // Account for bit 3 of constant ab
        T14  = OV;                      // Account for bit 3 of constant b8
        MSK = OV & 0x80808080 ;        // Create a mask of the MSBs
        OF = (MSK >> 7) * 0x1d ;       // Account for Overflow
        OV <<= 1 ;                     // Multiply the original data by 2
        OV &= 0xfefefefe ;             // Clear the MSBs
        OV ^= OF ;                     // Add in the overflow mask

        // Process bit 4
        T0 ^= OV;                      // Account for bit 4 of constant 39
        T2 ^= OV;                      // Account for bit 4 of constant 7a
        T3 ^= OV;                      // Account for bit 4 of constant 71
        T9 ^= OV;                      // Account for bit 4 of constant b7
        T10 ^= OV;                      // Account for bit 4 of constant 78
        T11 ^= OV;                      // Account for bit 4 of constant 7e
        T13 ^= OV;                      // Account for bit 4 of constant 52
        T14 ^= OV;                      // Account for bit 4 of constant b8
        MSK = OV & 0x80808080 ;        // Create a mask of the MSBs
        OF = (MSK >> 7) * 0x1d ;       // Account for Overflow
        OV <<= 1 ;                     // Multiply the original data by 2
        OV &= 0xfefefefe ;             // Clear the MSBs
        OV ^= OF ;                     // Add in the overflow mask

        // Process bit 5
        T0 ^= OV;                      // Account for bit 5 of constant 39
        T1 ^= OV;                      // Account for bit 5 of constant a8
        T2 ^= OV;                      // Account for bit 5 of constant 7a
        T3 ^= OV;                      // Account for bit 5 of constant 71
        T6 ^= OV;                      // Account for bit 5 of constant a7
        T7 ^= OV;                      // Account for bit 5 of constant 61
        T8 ^= OV;                      // Account for bit 5 of constant 23
        T9 ^= OV;                      // Account for bit 5 of constant b7
        T10 ^= OV;                      // Account for bit 5 of constant 78
        T11 ^= OV;                      // Account for bit 5 of constant 7e
        T12 ^= OV;                      // Account for bit 5 of constant ab
        T14 ^= OV;                      // Account for bit 5 of constant b8
        MSK = OV & 0x80808080 ;        // Create a mask of the MSBs
        OF = (MSK >> 7) * 0x1d ;       // Account for Overflow
        OV <<= 1 ;                     // Multiply the original data by 2
        OV &= 0xfefefefe ;             // Clear the MSBs
        OV ^= OF ;                     // Add in the overflow mask

        // Process bit 6
        T2 ^= OV;                      // Account for bit 6 of constant 7a
        T3 ^= OV;                      // Account for bit 6 of constant 71
        T4 ^= OV;                      // Account for bit 6 of constant 4c
        T7 ^= OV;                      // Account for bit 6 of constant 61
        T10 ^= OV;                      // Account for bit 6 of constant 78
        T11 ^= OV;                      // Account for bit 6 of constant 7e
        T13 ^= OV;                      // Account for bit 6 of constant 52
        MSK = OV & 0x80808080 ;        // Create a mask of the MSBs
        OF = (MSK >> 7) * 0x1d ;       // Account for Overflow
        OV <<= 1 ;                     // Multiply the original data by 2
        OV &= 0xfefefefe ;             // Clear the MSBs
        OV ^= OF ;                     // Add in the overflow mask

        // Process bit 7
        T1 ^= OV;                      // Account for bit 7 of constant a8
        T6 ^= OV;                      // Account for bit 7 of constant a7
        T9 ^= OV;                      // Account for bit 7 of constant b7
        T12 ^= OV;                      // Account for bit 7 of constant ab
        T14 ^= OV;                      // Account for bit 7 of constant b8

        // Now process the LFSR
        P0 = P1 ^ T0 ;
        P1 = P2 ^ T1 ;
        P2 = P3 ^ T2 ;
        P3 = P4 ^ T3 ;
        P4 = P5 ^ T4 ;
        P5 = P6 ^ T5 ;
        P6 = P7 ^ T6 ;
        P7 = P8 ^ T7 ;
        P8 = P9 ^ T8 ;
        P9 = P10 ^ T9 ;
        P10 = P11 ^ T10 ;
        P11 = P12 ^ T11 ;
        P12 = P13 ^ T12 ;
        P13 = P14 ^ T13 ;
        P14 = P15 ^ T14 ;
        P15 = P16 ^ T13 ;
        P16 = P17 ^ T12 ;
        P17 = P18 ^ T11 ;
        P18 = P19 ^ T10 ;
        P19 = P20 ^ T9 ;
        P20 = P21 ^ T8 ;
        P21 = P22 ^ T7 ;
        P22 = P23 ^ T6 ;
        P23 = P24 ^ T5 ;
        P24 = P25 ^ T4 ;
        P25 = P26 ^ T3 ;
        P26 = P27 ^ T2 ;
        P27 = P28 ^ T1 ;
        P28 = P29 ^ T0 ;
        P29 = OOV ;
    }

    if ( decoder == 0 )
    {
        // Now store computed check symbols back to memory
        loc = CWP[k+0] + i;
        *loc = P0 ;
        loc = CWP[k+1] + i;
        *loc = P1 ;
        loc = CWP[k+2] + i;
        *loc = P2 ;
        loc = CWP[k+3] + i;
        *loc = P3 ;
        loc = CWP[k+4] + i;
        *loc = P4 ;
        loc = CWP[k+5] + i;
        *loc = P5 ;
        loc = CWP[k+6] + i;
        *loc = P6 ;
        loc = CWP[k+7] + i;
        *loc = P7 ;
        loc = CWP[k+8] + i;
        *loc = P8 ;
        loc = CWP[k+9] + i;
        *loc = P9 ;
        loc = CWP[k+10] + i;
        *loc = P10 ;
        loc = CWP[k+11] + i;
        *loc = P11 ;
        loc = CWP[k+12] + i;
        *loc = P12 ;
        loc = CWP[k+13] + i;
        *loc = P13 ;
        loc = CWP[k+14] + i;
        *loc = P14 ;
        loc = CWP[k+15] + i;
        *loc = P15 ;
        loc = CWP[k+16] + i;
        *loc = P16 ;
        loc = CWP[k+17] + i;
        *loc = P17 ;
        loc = CWP[k+18] + i;
        *loc = P18 ;
        loc = CWP[k+19] + i;
        *loc = P19 ;
        loc = CWP[k+20] + i;
        *loc = P20 ;
        loc = CWP[k+21] + i;
        *loc = P21 ;
        loc = CWP[k+22] + i;
        *loc = P22 ;
        loc = CWP[k+23] + i;
        *loc = P23 ;
        loc = CWP[k+24] + i;
        *loc = P24 ;
        loc = CWP[k+25] + i;
        *loc = P25 ;
        loc = CWP[k+26] + i;
        *loc = P26 ;
        loc = CWP[k+27] + i;
        *loc = P27 ;
        loc = CWP[k+28] + i;
        *loc = P28 ;
        loc = CWP[k+29] + i;
        *loc = P29 ;
    }
    else
    {
        // Now check for parity equal to zero
        T0 = P0;
        T0 |= P1;
        T0 |= P2;
        T0 |= P3;
        T0 |= P4;
        T0 |= P5;
        T0 |= P6;
        T0 |= P7;
        T0 |= P8;
        T0 |= P9;
        T0 |= P10;
        T0 |= P11;
        T0 |= P12;
        T0 |= P13;
        T0 |= P14;
        T0 |= P15;
        T0 |= P16;
        T0 |= P17;
        T0 |= P18;
        T0 |= P19;
        T0 |= P20;
        T0 |= P21;
        T0 |= P22;
        T0 |= P23;
        T0 |= P24;
        T0 |= P25;
        T0 |= P26;
        T0 |= P27;
        T0 |= P28;
        T0 |= P29;
        if (T0 != 0)
        {
            unsigned char Sr[30], S[30], errLocs[30], errMags[30] ;
            unsigned char Lambda[30+1], B[30+1], T[30+1], Omega[30+1] ;
            
            if ( T0 & (0xff << 0) )
            {
                atomicAdd(&error_count, 1);
                Sr[0] = (P0>>0) & 0xff ;
                Sr[1] = (P1>>0) & 0xff ;
                Sr[2] = (P2>>0) & 0xff ;
                Sr[3] = (P3>>0) & 0xff ;
                Sr[4] = (P4>>0) & 0xff ;
                Sr[5] = (P5>>0) & 0xff ;
                Sr[6] = (P6>>0) & 0xff ;
                Sr[7] = (P7>>0) & 0xff ;
                Sr[8] = (P8>>0) & 0xff ;
                Sr[9] = (P9>>0) & 0xff ;
                Sr[10] = (P10>>0) & 0xff ;
                Sr[11] = (P11>>0) & 0xff ;
                Sr[12] = (P12>>0) & 0xff ;
                Sr[13] = (P13>>0) & 0xff ;
                Sr[14] = (P14>>0) & 0xff ;
                Sr[15] = (P15>>0) & 0xff ;
                Sr[16] = (P16>>0) & 0xff ;
                Sr[17] = (P17>>0) & 0xff ;
                Sr[18] = (P18>>0) & 0xff ;
                Sr[19] = (P19>>0) & 0xff ;
                Sr[20] = (P20>>0) & 0xff ;
                Sr[21] = (P21>>0) & 0xff ;
                Sr[22] = (P22>>0) & 0xff ;
                Sr[23] = (P23>>0) & 0xff ;
                Sr[24] = (P24>>0) & 0xff ;
                Sr[25] = (P25>>0) & 0xff ;
                Sr[26] = (P26>>0) & 0xff ;
                Sr[27] = (P27>>0) & 0xff ;
                Sr[28] = (P28>>0) & 0xff ;
                Sr[29] = (P29>>0) & 0xff ;
                rs_decode_palindromic_CUDA(Sr, S, 30, k, errLocs, errMags,
                                           0+(i<<2), Lambda, B, T, Omega) ;
            }
            if ( T0 & (0xff << 8) )
            {
                atomicAdd(&error_count, 1);
                Sr[0] = (P0>>8) & 0xff ;
                Sr[1] = (P1>>8) & 0xff ;
                Sr[2] = (P2>>8) & 0xff ;
                Sr[3] = (P3>>8) & 0xff ;
                Sr[4] = (P4>>8) & 0xff ;
                Sr[5] = (P5>>8) & 0xff ;
                Sr[6] = (P6>>8) & 0xff ;
                Sr[7] = (P7>>8) & 0xff ;
                Sr[8] = (P8>>8) & 0xff ;
                Sr[9] = (P9>>8) & 0xff ;
                Sr[10] = (P10>>8) & 0xff ;
                Sr[11] = (P11>>8) & 0xff ;
                Sr[12] = (P12>>8) & 0xff ;
                Sr[13] = (P13>>8) & 0xff ;
                Sr[14] = (P14>>8) & 0xff ;
                Sr[15] = (P15>>8) & 0xff ;
                Sr[16] = (P16>>8) & 0xff ;
                Sr[17] = (P17>>8) & 0xff ;
                Sr[18] = (P18>>8) & 0xff ;
                Sr[19] = (P19>>8) & 0xff ;
                Sr[20] = (P20>>8) & 0xff ;
                Sr[21] = (P21>>8) & 0xff ;
                Sr[22] = (P22>>8) & 0xff ;
                Sr[23] = (P23>>8) & 0xff ;
                Sr[24] = (P24>>8) & 0xff ;
                Sr[25] = (P25>>8) & 0xff ;
                Sr[26] = (P26>>8) & 0xff ;
                Sr[27] = (P27>>8) & 0xff ;
                Sr[28] = (P28>>8) & 0xff ;
                Sr[29] = (P29>>8) & 0xff ;
                rs_decode_palindromic_CUDA(Sr, S, 30, k, errLocs, errMags,
                                           1+(i<<2), Lambda, B, T, Omega) ;
            }
            if ( T0 & (0xff << 16) )
            {
                atomicAdd(&error_count, 1);
                Sr[0] = (P0>>16) & 0xff ;
                Sr[1] = (P1>>16) & 0xff ;
                Sr[2] = (P2>>16) & 0xff ;
                Sr[3] = (P3>>16) & 0xff ;
                Sr[4] = (P4>>16) & 0xff ;
                Sr[5] = (P5>>16) & 0xff ;
                Sr[6] = (P6>>16) & 0xff ;
                Sr[7] = (P7>>16) & 0xff ;
                Sr[8] = (P8>>16) & 0xff ;
                Sr[9] = (P9>>16) & 0xff ;
                Sr[10] = (P10>>16) & 0xff ;
                Sr[11] = (P11>>16) & 0xff ;
                Sr[12] = (P12>>16) & 0xff ;
                Sr[13] = (P13>>16) & 0xff ;
                Sr[14] = (P14>>16) & 0xff ;
                Sr[15] = (P15>>16) & 0xff ;
                Sr[16] = (P16>>16) & 0xff ;
                Sr[17] = (P17>>16) & 0xff ;
                Sr[18] = (P18>>16) & 0xff ;
                Sr[19] = (P19>>16) & 0xff ;
                Sr[20] = (P20>>16) & 0xff ;
                Sr[21] = (P21>>16) & 0xff ;
                Sr[22] = (P22>>16) & 0xff ;
                Sr[23] = (P23>>16) & 0xff ;
                Sr[24] = (P24>>16) & 0xff ;
                Sr[25] = (P25>>16) & 0xff ;
                Sr[26] = (P26>>16) & 0xff ;
                Sr[27] = (P27>>16) & 0xff ;
                Sr[28] = (P28>>16) & 0xff ;
                Sr[29] = (P29>>16) & 0xff ;
                rs_decode_palindromic_CUDA(Sr, S, 30, k, errLocs, errMags,
                                           2+(i<<2), Lambda, B, T, Omega) ;
            }
            if ( T0 & (0xff << 24) )
            {
                atomicAdd(&error_count, 1);
                Sr[0] = (P0>>24) & 0xff ;
                Sr[1] = (P1>>24) & 0xff ;
                Sr[2] = (P2>>24) & 0xff ;
                Sr[3] = (P3>>24) & 0xff ;
                Sr[4] = (P4>>24) & 0xff ;
                Sr[5] = (P5>>24) & 0xff ;
                Sr[6] = (P6>>24) & 0xff ;
                Sr[7] = (P7>>24) & 0xff ;
                Sr[8] = (P8>>24) & 0xff ;
                Sr[9] = (P9>>24) & 0xff ;
                Sr[10] = (P10>>24) & 0xff ;
                Sr[11] = (P11>>24) & 0xff ;
                Sr[12] = (P12>>24) & 0xff ;
                Sr[13] = (P13>>24) & 0xff ;
                Sr[14] = (P14>>24) & 0xff ;
                Sr[15] = (P15>>24) & 0xff ;
                Sr[16] = (P16>>24) & 0xff ;
                Sr[17] = (P17>>24) & 0xff ;
                Sr[18] = (P18>>24) & 0xff ;
                Sr[19] = (P19>>24) & 0xff ;
                Sr[20] = (P20>>24) & 0xff ;
                Sr[21] = (P21>>24) & 0xff ;
                Sr[22] = (P22>>24) & 0xff ;
                Sr[23] = (P23>>24) & 0xff ;
                Sr[24] = (P24>>24) & 0xff ;
                Sr[25] = (P25>>24) & 0xff ;
                Sr[26] = (P26>>24) & 0xff ;
                Sr[27] = (P27>>24) & 0xff ;
                Sr[28] = (P28>>24) & 0xff ;
                Sr[29] = (P29>>24) & 0xff ;
                rs_decode_palindromic_CUDA(Sr, S, 30, k, errLocs, errMags,
                                           3+(i<<2), Lambda, B, T, Omega) ;
            }
        }
    }
}

// ----------------------------------------------
// Parallel LFSR Sequencer for P = 31 Codewords
// ----------------------------------------------
__global__ void ParallelLFSRSequencer31_CUDA(int k, int decoder)
{
    // Find out place in the grid of threads
    int i = blockIdx.x * blockDim.x + threadIdx.x;

    // Pointer to develop original data address
    unsigned int* loc ;

    // Create the parity values, initialize to zero
    unsigned int P0 = 0 ;
    unsigned int P1 = 0 ;
    unsigned int P2 = 0 ;
    unsigned int P3 = 0 ;
    unsigned int P4 = 0 ;
    unsigned int P5 = 0 ;
    unsigned int P6 = 0 ;
    unsigned int P7 = 0 ;
    unsigned int P8 = 0 ;
    unsigned int P9 = 0 ;
    unsigned int P10 = 0 ;
    unsigned int P11 = 0 ;
    unsigned int P12 = 0 ;
    unsigned int P13 = 0 ;
    unsigned int P14 = 0 ;
    unsigned int P15 = 0 ;
    unsigned int P16 = 0 ;
    unsigned int P17 = 0 ;
    unsigned int P18 = 0 ;
    unsigned int P19 = 0 ;
    unsigned int P20 = 0 ;
    unsigned int P21 = 0 ;
    unsigned int P22 = 0 ;
    unsigned int P23 = 0 ;
    unsigned int P24 = 0 ;
    unsigned int P25 = 0 ;
    unsigned int P26 = 0 ;
    unsigned int P27 = 0 ;
    unsigned int P28 = 0 ;
    unsigned int P29 = 0 ;
    unsigned int P30 = 0 ;

    // Temporary values for self reciprocol polynomial
    unsigned int T0;
    unsigned int T1;
    unsigned int T2;
    unsigned int T3;
    unsigned int T4;
    unsigned int T5;
    unsigned int T6;
    unsigned int T7;
    unsigned int T8;
    unsigned int T9;
    unsigned int T10;
    unsigned int T11;
    unsigned int T12;
    unsigned int T13;
    unsigned int T14;

    // Overflow value for LFSR and carry mask
    unsigned int OV, MSK, OF, OOV ;

    // Adjust K if we are decoding
    k += decoder ? 31 : 0 ;

    // Loop through each symbol
    for ( int sym = 0 ; sym < k ; sym ++ )
    {
        loc = CWP[sym] + i;
        OV = *loc ^ P0 ;
        OOV = OV ;

        // Process bit 0
        T3  = OV;                      // Account for bit 0 of constant 27
        T4  = OV;                      // Account for bit 0 of constant 7d
        T6  = OV;                      // Account for bit 0 of constant 3b
        T7  = OV;                      // Account for bit 0 of constant 3f
        T8  = OV;                      // Account for bit 0 of constant eb
        T9  = OV;                      // Account for bit 0 of constant a5
        T12  = OV;                      // Account for bit 0 of constant d
        T13  = OV;                      // Account for bit 0 of constant c7
        T14  = OV;                      // Account for bit 0 of constant 49
        MSK = OV & 0x80808080 ;        // Create a mask of the MSBs
        OF = (MSK >> 7) * 0x1d ;       // Account for Overflow
        OV <<= 1 ;                     // Multiply the original data by 2
        OV &= 0xfefefefe ;             // Clear the MSBs
        OV ^= OF ;                     // Add in the overflow mask

        // Process bit 1
        T2  = OV;                      // Account for bit 1 of constant a6
        T3 ^= OV;                      // Account for bit 1 of constant 27
        T6 ^= OV;                      // Account for bit 1 of constant 3b
        T7 ^= OV;                      // Account for bit 1 of constant 3f
        T8 ^= OV;                      // Account for bit 1 of constant eb
        T11  = OV;                      // Account for bit 1 of constant 3e
        T13 ^= OV;                      // Account for bit 1 of constant c7
        MSK = OV & 0x80808080 ;        // Create a mask of the MSBs
        OF = (MSK >> 7) * 0x1d ;       // Account for Overflow
        OV <<= 1 ;                     // Multiply the original data by 2
        OV &= 0xfefefefe ;             // Clear the MSBs
        OV ^= OF ;                     // Add in the overflow mask

        // Process bit 2
        T2 ^= OV;                      // Account for bit 2 of constant a6
        T3 ^= OV;                      // Account for bit 2 of constant 27
        T4 ^= OV;                      // Account for bit 2 of constant 7d
        T5  = OV;                      // Account for bit 2 of constant 2c
        T7 ^= OV;                      // Account for bit 2 of constant 3f
        T9 ^= OV;                      // Account for bit 2 of constant a5
        T11 ^= OV;                      // Account for bit 2 of constant 3e
        T12 ^= OV;                      // Account for bit 2 of constant d
        T13 ^= OV;                      // Account for bit 2 of constant c7
        MSK = OV & 0x80808080 ;        // Create a mask of the MSBs
        OF = (MSK >> 7) * 0x1d ;       // Account for Overflow
        OV <<= 1 ;                     // Multiply the original data by 2
        OV &= 0xfefefefe ;             // Clear the MSBs
        OV ^= OF ;                     // Add in the overflow mask

        // Process bit 3
        T4 ^= OV;                      // Account for bit 3 of constant 7d
        T5 ^= OV;                      // Account for bit 3 of constant 2c
        T6 ^= OV;                      // Account for bit 3 of constant 3b
        T7 ^= OV;                      // Account for bit 3 of constant 3f
        T8 ^= OV;                      // Account for bit 3 of constant eb
        T11 ^= OV;                      // Account for bit 3 of constant 3e
        T12 ^= OV;                      // Account for bit 3 of constant d
        T14 ^= OV;                      // Account for bit 3 of constant 49
        MSK = OV & 0x80808080 ;        // Create a mask of the MSBs
        OF = (MSK >> 7) * 0x1d ;       // Account for Overflow
        OV <<= 1 ;                     // Multiply the original data by 2
        OV &= 0xfefefefe ;             // Clear the MSBs
        OV ^= OF ;                     // Add in the overflow mask

        // Process bit 4
        T4 ^= OV;                      // Account for bit 4 of constant 7d
        T6 ^= OV;                      // Account for bit 4 of constant 3b
        T7 ^= OV;                      // Account for bit 4 of constant 3f
        T11 ^= OV;                      // Account for bit 4 of constant 3e
        MSK = OV & 0x80808080 ;        // Create a mask of the MSBs
        OF = (MSK >> 7) * 0x1d ;       // Account for Overflow
        OV <<= 1 ;                     // Multiply the original data by 2
        OV &= 0xfefefefe ;             // Clear the MSBs
        OV ^= OF ;                     // Add in the overflow mask

        // Process bit 5
        T0  = OV;                      // Account for bit 5 of constant 20
        T2 ^= OV;                      // Account for bit 5 of constant a6
        T3 ^= OV;                      // Account for bit 5 of constant 27
        T4 ^= OV;                      // Account for bit 5 of constant 7d
        T5 ^= OV;                      // Account for bit 5 of constant 2c
        T6 ^= OV;                      // Account for bit 5 of constant 3b
        T7 ^= OV;                      // Account for bit 5 of constant 3f
        T8 ^= OV;                      // Account for bit 5 of constant eb
        T9 ^= OV;                      // Account for bit 5 of constant a5
        T10  = OV;                      // Account for bit 5 of constant e0
        T11 ^= OV;                      // Account for bit 5 of constant 3e
        MSK = OV & 0x80808080 ;        // Create a mask of the MSBs
        OF = (MSK >> 7) * 0x1d ;       // Account for Overflow
        OV <<= 1 ;                     // Multiply the original data by 2
        OV &= 0xfefefefe ;             // Clear the MSBs
        OV ^= OF ;                     // Add in the overflow mask

        // Process bit 6
        T4 ^= OV;                      // Account for bit 6 of constant 7d
        T8 ^= OV;                      // Account for bit 6 of constant eb
        T10 ^= OV;                      // Account for bit 6 of constant e0
        T13 ^= OV;                      // Account for bit 6 of constant c7
        T14 ^= OV;                      // Account for bit 6 of constant 49
        MSK = OV & 0x80808080 ;        // Create a mask of the MSBs
        OF = (MSK >> 7) * 0x1d ;       // Account for Overflow
        OV <<= 1 ;                     // Multiply the original data by 2
        OV &= 0xfefefefe ;             // Clear the MSBs
        OV ^= OF ;                     // Add in the overflow mask

        // Process bit 7
        T1  = OV;                      // Account for bit 7 of constant 80
        T2 ^= OV;                      // Account for bit 7 of constant a6
        T8 ^= OV;                      // Account for bit 7 of constant eb
        T9 ^= OV;                      // Account for bit 7 of constant a5
        T10 ^= OV;                      // Account for bit 7 of constant e0
        T13 ^= OV;                      // Account for bit 7 of constant c7

        // Now process the LFSR
        P0 = P1 ^ T0 ;
        P1 = P2 ^ T1 ;
        P2 = P3 ^ T2 ;
        P3 = P4 ^ T3 ;
        P4 = P5 ^ T4 ;
        P5 = P6 ^ T5 ;
        P6 = P7 ^ T6 ;
        P7 = P8 ^ T7 ;
        P8 = P9 ^ T8 ;
        P9 = P10 ^ T9 ;
        P10 = P11 ^ T10 ;
        P11 = P12 ^ T11 ;
        P12 = P13 ^ T12 ;
        P13 = P14 ^ T13 ;
        P14 = P15 ^ T14 ;
        P15 = P16 ^ T14 ;
        P16 = P17 ^ T13 ;
        P17 = P18 ^ T12 ;
        P18 = P19 ^ T11 ;
        P19 = P20 ^ T10 ;
        P20 = P21 ^ T9 ;
        P21 = P22 ^ T8 ;
        P22 = P23 ^ T7 ;
        P23 = P24 ^ T6 ;
        P24 = P25 ^ T5 ;
        P25 = P26 ^ T4 ;
        P26 = P27 ^ T3 ;
        P27 = P28 ^ T2 ;
        P28 = P29 ^ T1 ;
        P29 = P30 ^ T0 ;
        P30 = OOV ;
    }

    if ( decoder == 0 )
    {
        // Now store computed check symbols back to memory
        loc = CWP[k+0] + i;
        *loc = P0 ;
        loc = CWP[k+1] + i;
        *loc = P1 ;
        loc = CWP[k+2] + i;
        *loc = P2 ;
        loc = CWP[k+3] + i;
        *loc = P3 ;
        loc = CWP[k+4] + i;
        *loc = P4 ;
        loc = CWP[k+5] + i;
        *loc = P5 ;
        loc = CWP[k+6] + i;
        *loc = P6 ;
        loc = CWP[k+7] + i;
        *loc = P7 ;
        loc = CWP[k+8] + i;
        *loc = P8 ;
        loc = CWP[k+9] + i;
        *loc = P9 ;
        loc = CWP[k+10] + i;
        *loc = P10 ;
        loc = CWP[k+11] + i;
        *loc = P11 ;
        loc = CWP[k+12] + i;
        *loc = P12 ;
        loc = CWP[k+13] + i;
        *loc = P13 ;
        loc = CWP[k+14] + i;
        *loc = P14 ;
        loc = CWP[k+15] + i;
        *loc = P15 ;
        loc = CWP[k+16] + i;
        *loc = P16 ;
        loc = CWP[k+17] + i;
        *loc = P17 ;
        loc = CWP[k+18] + i;
        *loc = P18 ;
        loc = CWP[k+19] + i;
        *loc = P19 ;
        loc = CWP[k+20] + i;
        *loc = P20 ;
        loc = CWP[k+21] + i;
        *loc = P21 ;
        loc = CWP[k+22] + i;
        *loc = P22 ;
        loc = CWP[k+23] + i;
        *loc = P23 ;
        loc = CWP[k+24] + i;
        *loc = P24 ;
        loc = CWP[k+25] + i;
        *loc = P25 ;
        loc = CWP[k+26] + i;
        *loc = P26 ;
        loc = CWP[k+27] + i;
        *loc = P27 ;
        loc = CWP[k+28] + i;
        *loc = P28 ;
        loc = CWP[k+29] + i;
        *loc = P29 ;
        loc = CWP[k+30] + i;
        *loc = P30 ;
    }
    else
    {
        // Now check for parity equal to zero
        T0 = P0;
        T0 |= P1;
        T0 |= P2;
        T0 |= P3;
        T0 |= P4;
        T0 |= P5;
        T0 |= P6;
        T0 |= P7;
        T0 |= P8;
        T0 |= P9;
        T0 |= P10;
        T0 |= P11;
        T0 |= P12;
        T0 |= P13;
        T0 |= P14;
        T0 |= P15;
        T0 |= P16;
        T0 |= P17;
        T0 |= P18;
        T0 |= P19;
        T0 |= P20;
        T0 |= P21;
        T0 |= P22;
        T0 |= P23;
        T0 |= P24;
        T0 |= P25;
        T0 |= P26;
        T0 |= P27;
        T0 |= P28;
        T0 |= P29;
        T0 |= P30;
        if (T0 != 0)
        {
            unsigned char Sr[31], S[31], errLocs[31], errMags[31] ;
            unsigned char Lambda[31+1], B[31+1], T[31+1], Omega[31+1] ;
            
            if ( T0 & (0xff << 0) )
            {
                atomicAdd(&error_count, 1);
                Sr[0] = (P0>>0) & 0xff ;
                Sr[1] = (P1>>0) & 0xff ;
                Sr[2] = (P2>>0) & 0xff ;
                Sr[3] = (P3>>0) & 0xff ;
                Sr[4] = (P4>>0) & 0xff ;
                Sr[5] = (P5>>0) & 0xff ;
                Sr[6] = (P6>>0) & 0xff ;
                Sr[7] = (P7>>0) & 0xff ;
                Sr[8] = (P8>>0) & 0xff ;
                Sr[9] = (P9>>0) & 0xff ;
                Sr[10] = (P10>>0) & 0xff ;
                Sr[11] = (P11>>0) & 0xff ;
                Sr[12] = (P12>>0) & 0xff ;
                Sr[13] = (P13>>0) & 0xff ;
                Sr[14] = (P14>>0) & 0xff ;
                Sr[15] = (P15>>0) & 0xff ;
                Sr[16] = (P16>>0) & 0xff ;
                Sr[17] = (P17>>0) & 0xff ;
                Sr[18] = (P18>>0) & 0xff ;
                Sr[19] = (P19>>0) & 0xff ;
                Sr[20] = (P20>>0) & 0xff ;
                Sr[21] = (P21>>0) & 0xff ;
                Sr[22] = (P22>>0) & 0xff ;
                Sr[23] = (P23>>0) & 0xff ;
                Sr[24] = (P24>>0) & 0xff ;
                Sr[25] = (P25>>0) & 0xff ;
                Sr[26] = (P26>>0) & 0xff ;
                Sr[27] = (P27>>0) & 0xff ;
                Sr[28] = (P28>>0) & 0xff ;
                Sr[29] = (P29>>0) & 0xff ;
                Sr[30] = (P30>>0) & 0xff ;
                rs_decode_palindromic_CUDA(Sr, S, 31, k, errLocs, errMags,
                                           0+(i<<2), Lambda, B, T, Omega) ;
            }
            if ( T0 & (0xff << 8) )
            {
                atomicAdd(&error_count, 1);
                Sr[0] = (P0>>8) & 0xff ;
                Sr[1] = (P1>>8) & 0xff ;
                Sr[2] = (P2>>8) & 0xff ;
                Sr[3] = (P3>>8) & 0xff ;
                Sr[4] = (P4>>8) & 0xff ;
                Sr[5] = (P5>>8) & 0xff ;
                Sr[6] = (P6>>8) & 0xff ;
                Sr[7] = (P7>>8) & 0xff ;
                Sr[8] = (P8>>8) & 0xff ;
                Sr[9] = (P9>>8) & 0xff ;
                Sr[10] = (P10>>8) & 0xff ;
                Sr[11] = (P11>>8) & 0xff ;
                Sr[12] = (P12>>8) & 0xff ;
                Sr[13] = (P13>>8) & 0xff ;
                Sr[14] = (P14>>8) & 0xff ;
                Sr[15] = (P15>>8) & 0xff ;
                Sr[16] = (P16>>8) & 0xff ;
                Sr[17] = (P17>>8) & 0xff ;
                Sr[18] = (P18>>8) & 0xff ;
                Sr[19] = (P19>>8) & 0xff ;
                Sr[20] = (P20>>8) & 0xff ;
                Sr[21] = (P21>>8) & 0xff ;
                Sr[22] = (P22>>8) & 0xff ;
                Sr[23] = (P23>>8) & 0xff ;
                Sr[24] = (P24>>8) & 0xff ;
                Sr[25] = (P25>>8) & 0xff ;
                Sr[26] = (P26>>8) & 0xff ;
                Sr[27] = (P27>>8) & 0xff ;
                Sr[28] = (P28>>8) & 0xff ;
                Sr[29] = (P29>>8) & 0xff ;
                Sr[30] = (P30>>8) & 0xff ;
                rs_decode_palindromic_CUDA(Sr, S, 31, k, errLocs, errMags,
                                           1+(i<<2), Lambda, B, T, Omega) ;
            }
            if ( T0 & (0xff << 16) )
            {
                atomicAdd(&error_count, 1);
                Sr[0] = (P0>>16) & 0xff ;
                Sr[1] = (P1>>16) & 0xff ;
                Sr[2] = (P2>>16) & 0xff ;
                Sr[3] = (P3>>16) & 0xff ;
                Sr[4] = (P4>>16) & 0xff ;
                Sr[5] = (P5>>16) & 0xff ;
                Sr[6] = (P6>>16) & 0xff ;
                Sr[7] = (P7>>16) & 0xff ;
                Sr[8] = (P8>>16) & 0xff ;
                Sr[9] = (P9>>16) & 0xff ;
                Sr[10] = (P10>>16) & 0xff ;
                Sr[11] = (P11>>16) & 0xff ;
                Sr[12] = (P12>>16) & 0xff ;
                Sr[13] = (P13>>16) & 0xff ;
                Sr[14] = (P14>>16) & 0xff ;
                Sr[15] = (P15>>16) & 0xff ;
                Sr[16] = (P16>>16) & 0xff ;
                Sr[17] = (P17>>16) & 0xff ;
                Sr[18] = (P18>>16) & 0xff ;
                Sr[19] = (P19>>16) & 0xff ;
                Sr[20] = (P20>>16) & 0xff ;
                Sr[21] = (P21>>16) & 0xff ;
                Sr[22] = (P22>>16) & 0xff ;
                Sr[23] = (P23>>16) & 0xff ;
                Sr[24] = (P24>>16) & 0xff ;
                Sr[25] = (P25>>16) & 0xff ;
                Sr[26] = (P26>>16) & 0xff ;
                Sr[27] = (P27>>16) & 0xff ;
                Sr[28] = (P28>>16) & 0xff ;
                Sr[29] = (P29>>16) & 0xff ;
                Sr[30] = (P30>>16) & 0xff ;
                rs_decode_palindromic_CUDA(Sr, S, 31, k, errLocs, errMags,
                                           2+(i<<2), Lambda, B, T, Omega) ;
            }
            if ( T0 & (0xff << 24) )
            {
                atomicAdd(&error_count, 1);
                Sr[0] = (P0>>24) & 0xff ;
                Sr[1] = (P1>>24) & 0xff ;
                Sr[2] = (P2>>24) & 0xff ;
                Sr[3] = (P3>>24) & 0xff ;
                Sr[4] = (P4>>24) & 0xff ;
                Sr[5] = (P5>>24) & 0xff ;
                Sr[6] = (P6>>24) & 0xff ;
                Sr[7] = (P7>>24) & 0xff ;
                Sr[8] = (P8>>24) & 0xff ;
                Sr[9] = (P9>>24) & 0xff ;
                Sr[10] = (P10>>24) & 0xff ;
                Sr[11] = (P11>>24) & 0xff ;
                Sr[12] = (P12>>24) & 0xff ;
                Sr[13] = (P13>>24) & 0xff ;
                Sr[14] = (P14>>24) & 0xff ;
                Sr[15] = (P15>>24) & 0xff ;
                Sr[16] = (P16>>24) & 0xff ;
                Sr[17] = (P17>>24) & 0xff ;
                Sr[18] = (P18>>24) & 0xff ;
                Sr[19] = (P19>>24) & 0xff ;
                Sr[20] = (P20>>24) & 0xff ;
                Sr[21] = (P21>>24) & 0xff ;
                Sr[22] = (P22>>24) & 0xff ;
                Sr[23] = (P23>>24) & 0xff ;
                Sr[24] = (P24>>24) & 0xff ;
                Sr[25] = (P25>>24) & 0xff ;
                Sr[26] = (P26>>24) & 0xff ;
                Sr[27] = (P27>>24) & 0xff ;
                Sr[28] = (P28>>24) & 0xff ;
                Sr[29] = (P29>>24) & 0xff ;
                Sr[30] = (P30>>24) & 0xff ;
                rs_decode_palindromic_CUDA(Sr, S, 31, k, errLocs, errMags,
                                           3+(i<<2), Lambda, B, T, Omega) ;
            }
        }
    }
}

// ----------------------------------------------
// Parallel LFSR Sequencer for P = 32 Codewords
// ----------------------------------------------
__global__ void ParallelLFSRSequencer32_CUDA(int k, int decoder)
{
    // Find out place in the grid of threads
    int i = blockIdx.x * blockDim.x + threadIdx.x;

    // Pointer to develop original data address
    unsigned int* loc ;

    // Create the parity values, initialize to zero
    unsigned int P0 = 0 ;
    unsigned int P1 = 0 ;
    unsigned int P2 = 0 ;
    unsigned int P3 = 0 ;
    unsigned int P4 = 0 ;
    unsigned int P5 = 0 ;
    unsigned int P6 = 0 ;
    unsigned int P7 = 0 ;
    unsigned int P8 = 0 ;
    unsigned int P9 = 0 ;
    unsigned int P10 = 0 ;
    unsigned int P11 = 0 ;
    unsigned int P12 = 0 ;
    unsigned int P13 = 0 ;
    unsigned int P14 = 0 ;
    unsigned int P15 = 0 ;
    unsigned int P16 = 0 ;
    unsigned int P17 = 0 ;
    unsigned int P18 = 0 ;
    unsigned int P19 = 0 ;
    unsigned int P20 = 0 ;
    unsigned int P21 = 0 ;
    unsigned int P22 = 0 ;
    unsigned int P23 = 0 ;
    unsigned int P24 = 0 ;
    unsigned int P25 = 0 ;
    unsigned int P26 = 0 ;
    unsigned int P27 = 0 ;
    unsigned int P28 = 0 ;
    unsigned int P29 = 0 ;
    unsigned int P30 = 0 ;
    unsigned int P31 = 0 ;

    // Temporary values for self reciprocol polynomial
    unsigned int T0;
    unsigned int T1;
    unsigned int T2;
    unsigned int T3;
    unsigned int T4;
    unsigned int T5;
    unsigned int T6;
    unsigned int T7;
    unsigned int T8;
    unsigned int T9;
    unsigned int T10;
    unsigned int T11;
    unsigned int T12;
    unsigned int T13;
    unsigned int T14;
    unsigned int T15;

    // Overflow value for LFSR and carry mask
    unsigned int OV, MSK, OF, OOV ;

    // Adjust K if we are decoding
    k += decoder ? 32 : 0 ;

    // Loop through each symbol
    for ( int sym = 0 ; sym < k ; sym ++ )
    {
        loc = CWP[sym] + i;
        OV = *loc ^ P0 ;
        OOV = OV ;

        // Process bit 0
        T3  = OV;                      // Account for bit 0 of constant 85
        T5  = OV;                      // Account for bit 0 of constant 89
        T6  = OV;                      // Account for bit 0 of constant c9
        T7  = OV;                      // Account for bit 0 of constant 7
        T8  = OV;                      // Account for bit 0 of constant 8d
        T9  = OV;                      // Account for bit 0 of constant b
        T13  = OV;                      // Account for bit 0 of constant d1
        MSK = OV & 0x80808080 ;        // Create a mask of the MSBs
        OF = (MSK >> 7) * 0x1d ;       // Account for Overflow
        OV <<= 1 ;                     // Multiply the original data by 2
        OV &= 0xfefefefe ;             // Clear the MSBs
        OV ^= OF ;                     // Add in the overflow mask

        // Process bit 1
        T4  = OV;                      // Account for bit 1 of constant ee
        T7 ^= OV;                      // Account for bit 1 of constant 7
        T9 ^= OV;                      // Account for bit 1 of constant b
        T10  = OV;                      // Account for bit 1 of constant e2
        T11  = OV;                      // Account for bit 1 of constant 22
        T14  = OV;                      // Account for bit 1 of constant 16
        T15  = OV;                      // Account for bit 1 of constant 4e
        MSK = OV & 0x80808080 ;        // Create a mask of the MSBs
        OF = (MSK >> 7) * 0x1d ;       // Account for Overflow
        OV <<= 1 ;                     // Multiply the original data by 2
        OV &= 0xfefefefe ;             // Clear the MSBs
        OV ^= OF ;                     // Add in the overflow mask

        // Process bit 2
        T0  = OV;                      // Account for bit 2 of constant ec
        T1  = OV;                      // Account for bit 2 of constant f4
        T2  = OV;                      // Account for bit 2 of constant dc
        T3 ^= OV;                      // Account for bit 2 of constant 85
        T4 ^= OV;                      // Account for bit 2 of constant ee
        T7 ^= OV;                      // Account for bit 2 of constant 7
        T8 ^= OV;                      // Account for bit 2 of constant 8d
        T12  = OV;                      // Account for bit 2 of constant fc
        T14 ^= OV;                      // Account for bit 2 of constant 16
        T15 ^= OV;                      // Account for bit 2 of constant 4e
        MSK = OV & 0x80808080 ;        // Create a mask of the MSBs
        OF = (MSK >> 7) * 0x1d ;       // Account for Overflow
        OV <<= 1 ;                     // Multiply the original data by 2
        OV &= 0xfefefefe ;             // Clear the MSBs
        OV ^= OF ;                     // Add in the overflow mask

        // Process bit 3
        T0 ^= OV;                      // Account for bit 3 of constant ec
        T2 ^= OV;                      // Account for bit 3 of constant dc
        T4 ^= OV;                      // Account for bit 3 of constant ee
        T5 ^= OV;                      // Account for bit 3 of constant 89
        T6 ^= OV;                      // Account for bit 3 of constant c9
        T8 ^= OV;                      // Account for bit 3 of constant 8d
        T9 ^= OV;                      // Account for bit 3 of constant b
        T12 ^= OV;                      // Account for bit 3 of constant fc
        T15 ^= OV;                      // Account for bit 3 of constant 4e
        MSK = OV & 0x80808080 ;        // Create a mask of the MSBs
        OF = (MSK >> 7) * 0x1d ;       // Account for Overflow
        OV <<= 1 ;                     // Multiply the original data by 2
        OV &= 0xfefefefe ;             // Clear the MSBs
        OV ^= OF ;                     // Add in the overflow mask

        // Process bit 4
        T1 ^= OV;                      // Account for bit 4 of constant f4
        T2 ^= OV;                      // Account for bit 4 of constant dc
        T12 ^= OV;                      // Account for bit 4 of constant fc
        T13 ^= OV;                      // Account for bit 4 of constant d1
        T14 ^= OV;                      // Account for bit 4 of constant 16
        MSK = OV & 0x80808080 ;        // Create a mask of the MSBs
        OF = (MSK >> 7) * 0x1d ;       // Account for Overflow
        OV <<= 1 ;                     // Multiply the original data by 2
        OV &= 0xfefefefe ;             // Clear the MSBs
        OV ^= OF ;                     // Add in the overflow mask

        // Process bit 5
        T0 ^= OV;                      // Account for bit 5 of constant ec
        T1 ^= OV;                      // Account for bit 5 of constant f4
        T4 ^= OV;                      // Account for bit 5 of constant ee
        T10 ^= OV;                      // Account for bit 5 of constant e2
        T11 ^= OV;                      // Account for bit 5 of constant 22
        T12 ^= OV;                      // Account for bit 5 of constant fc
        MSK = OV & 0x80808080 ;        // Create a mask of the MSBs
        OF = (MSK >> 7) * 0x1d ;       // Account for Overflow
        OV <<= 1 ;                     // Multiply the original data by 2
        OV &= 0xfefefefe ;             // Clear the MSBs
        OV ^= OF ;                     // Add in the overflow mask

        // Process bit 6
        T0 ^= OV;                      // Account for bit 6 of constant ec
        T1 ^= OV;                      // Account for bit 6 of constant f4
        T2 ^= OV;                      // Account for bit 6 of constant dc
        T4 ^= OV;                      // Account for bit 6 of constant ee
        T6 ^= OV;                      // Account for bit 6 of constant c9
        T10 ^= OV;                      // Account for bit 6 of constant e2
        T12 ^= OV;                      // Account for bit 6 of constant fc
        T13 ^= OV;                      // Account for bit 6 of constant d1
        T15 ^= OV;                      // Account for bit 6 of constant 4e
        MSK = OV & 0x80808080 ;        // Create a mask of the MSBs
        OF = (MSK >> 7) * 0x1d ;       // Account for Overflow
        OV <<= 1 ;                     // Multiply the original data by 2
        OV &= 0xfefefefe ;             // Clear the MSBs
        OV ^= OF ;                     // Add in the overflow mask

        // Process bit 7
        T0 ^= OV;                      // Account for bit 7 of constant ec
        T1 ^= OV;                      // Account for bit 7 of constant f4
        T2 ^= OV;                      // Account for bit 7 of constant dc
        T3 ^= OV;                      // Account for bit 7 of constant 85
        T4 ^= OV;                      // Account for bit 7 of constant ee
        T5 ^= OV;                      // Account for bit 7 of constant 89
        T6 ^= OV;                      // Account for bit 7 of constant c9
        T8 ^= OV;                      // Account for bit 7 of constant 8d
        T10 ^= OV;                      // Account for bit 7 of constant e2
        T12 ^= OV;                      // Account for bit 7 of constant fc
        T13 ^= OV;                      // Account for bit 7 of constant d1

        // Now process the LFSR
        P0 = P1 ^ T0 ;
        P1 = P2 ^ T1 ;
        P2 = P3 ^ T2 ;
        P3 = P4 ^ T3 ;
        P4 = P5 ^ T4 ;
        P5 = P6 ^ T5 ;
        P6 = P7 ^ T6 ;
        P7 = P8 ^ T7 ;
        P8 = P9 ^ T8 ;
        P9 = P10 ^ T9 ;
        P10 = P11 ^ T10 ;
        P11 = P12 ^ T11 ;
        P12 = P13 ^ T12 ;
        P13 = P14 ^ T13 ;
        P14 = P15 ^ T14 ;
        P15 = P16 ^ T15 ;
        P16 = P17 ^ T14 ;
        P17 = P18 ^ T13 ;
        P18 = P19 ^ T12 ;
        P19 = P20 ^ T11 ;
        P20 = P21 ^ T10 ;
        P21 = P22 ^ T9 ;
        P22 = P23 ^ T8 ;
        P23 = P24 ^ T7 ;
        P24 = P25 ^ T6 ;
        P25 = P26 ^ T5 ;
        P26 = P27 ^ T4 ;
        P27 = P28 ^ T3 ;
        P28 = P29 ^ T2 ;
        P29 = P30 ^ T1 ;
        P30 = P31 ^ T0 ;
        P31 = OOV ;
    }

    if ( decoder == 0 )
    {
        // Now store computed check symbols back to memory
        loc = CWP[k+0] + i;
        *loc = P0 ;
        loc = CWP[k+1] + i;
        *loc = P1 ;
        loc = CWP[k+2] + i;
        *loc = P2 ;
        loc = CWP[k+3] + i;
        *loc = P3 ;
        loc = CWP[k+4] + i;
        *loc = P4 ;
        loc = CWP[k+5] + i;
        *loc = P5 ;
        loc = CWP[k+6] + i;
        *loc = P6 ;
        loc = CWP[k+7] + i;
        *loc = P7 ;
        loc = CWP[k+8] + i;
        *loc = P8 ;
        loc = CWP[k+9] + i;
        *loc = P9 ;
        loc = CWP[k+10] + i;
        *loc = P10 ;
        loc = CWP[k+11] + i;
        *loc = P11 ;
        loc = CWP[k+12] + i;
        *loc = P12 ;
        loc = CWP[k+13] + i;
        *loc = P13 ;
        loc = CWP[k+14] + i;
        *loc = P14 ;
        loc = CWP[k+15] + i;
        *loc = P15 ;
        loc = CWP[k+16] + i;
        *loc = P16 ;
        loc = CWP[k+17] + i;
        *loc = P17 ;
        loc = CWP[k+18] + i;
        *loc = P18 ;
        loc = CWP[k+19] + i;
        *loc = P19 ;
        loc = CWP[k+20] + i;
        *loc = P20 ;
        loc = CWP[k+21] + i;
        *loc = P21 ;
        loc = CWP[k+22] + i;
        *loc = P22 ;
        loc = CWP[k+23] + i;
        *loc = P23 ;
        loc = CWP[k+24] + i;
        *loc = P24 ;
        loc = CWP[k+25] + i;
        *loc = P25 ;
        loc = CWP[k+26] + i;
        *loc = P26 ;
        loc = CWP[k+27] + i;
        *loc = P27 ;
        loc = CWP[k+28] + i;
        *loc = P28 ;
        loc = CWP[k+29] + i;
        *loc = P29 ;
        loc = CWP[k+30] + i;
        *loc = P30 ;
        loc = CWP[k+31] + i;
        *loc = P31 ;
    }
    else
    {
        // Now check for parity equal to zero
        T0 = P0;
        T0 |= P1;
        T0 |= P2;
        T0 |= P3;
        T0 |= P4;
        T0 |= P5;
        T0 |= P6;
        T0 |= P7;
        T0 |= P8;
        T0 |= P9;
        T0 |= P10;
        T0 |= P11;
        T0 |= P12;
        T0 |= P13;
        T0 |= P14;
        T0 |= P15;
        T0 |= P16;
        T0 |= P17;
        T0 |= P18;
        T0 |= P19;
        T0 |= P20;
        T0 |= P21;
        T0 |= P22;
        T0 |= P23;
        T0 |= P24;
        T0 |= P25;
        T0 |= P26;
        T0 |= P27;
        T0 |= P28;
        T0 |= P29;
        T0 |= P30;
        T0 |= P31;
        if (T0 != 0)
        {
            unsigned char Sr[32], S[32], errLocs[32], errMags[32] ;
            unsigned char Lambda[32+1], B[32+1], T[32+1], Omega[32+1] ;
            
            if ( T0 & (0xff << 0) )
            {
                atomicAdd(&error_count, 1);
                Sr[0] = (P0>>0) & 0xff ;
                Sr[1] = (P1>>0) & 0xff ;
                Sr[2] = (P2>>0) & 0xff ;
                Sr[3] = (P3>>0) & 0xff ;
                Sr[4] = (P4>>0) & 0xff ;
                Sr[5] = (P5>>0) & 0xff ;
                Sr[6] = (P6>>0) & 0xff ;
                Sr[7] = (P7>>0) & 0xff ;
                Sr[8] = (P8>>0) & 0xff ;
                Sr[9] = (P9>>0) & 0xff ;
                Sr[10] = (P10>>0) & 0xff ;
                Sr[11] = (P11>>0) & 0xff ;
                Sr[12] = (P12>>0) & 0xff ;
                Sr[13] = (P13>>0) & 0xff ;
                Sr[14] = (P14>>0) & 0xff ;
                Sr[15] = (P15>>0) & 0xff ;
                Sr[16] = (P16>>0) & 0xff ;
                Sr[17] = (P17>>0) & 0xff ;
                Sr[18] = (P18>>0) & 0xff ;
                Sr[19] = (P19>>0) & 0xff ;
                Sr[20] = (P20>>0) & 0xff ;
                Sr[21] = (P21>>0) & 0xff ;
                Sr[22] = (P22>>0) & 0xff ;
                Sr[23] = (P23>>0) & 0xff ;
                Sr[24] = (P24>>0) & 0xff ;
                Sr[25] = (P25>>0) & 0xff ;
                Sr[26] = (P26>>0) & 0xff ;
                Sr[27] = (P27>>0) & 0xff ;
                Sr[28] = (P28>>0) & 0xff ;
                Sr[29] = (P29>>0) & 0xff ;
                Sr[30] = (P30>>0) & 0xff ;
                Sr[31] = (P31>>0) & 0xff ;
                rs_decode_palindromic_CUDA(Sr, S, 32, k, errLocs, errMags,
                                           0+(i<<2), Lambda, B, T, Omega) ;
            }
            if ( T0 & (0xff << 8) )
            {
                atomicAdd(&error_count, 1);
                Sr[0] = (P0>>8) & 0xff ;
                Sr[1] = (P1>>8) & 0xff ;
                Sr[2] = (P2>>8) & 0xff ;
                Sr[3] = (P3>>8) & 0xff ;
                Sr[4] = (P4>>8) & 0xff ;
                Sr[5] = (P5>>8) & 0xff ;
                Sr[6] = (P6>>8) & 0xff ;
                Sr[7] = (P7>>8) & 0xff ;
                Sr[8] = (P8>>8) & 0xff ;
                Sr[9] = (P9>>8) & 0xff ;
                Sr[10] = (P10>>8) & 0xff ;
                Sr[11] = (P11>>8) & 0xff ;
                Sr[12] = (P12>>8) & 0xff ;
                Sr[13] = (P13>>8) & 0xff ;
                Sr[14] = (P14>>8) & 0xff ;
                Sr[15] = (P15>>8) & 0xff ;
                Sr[16] = (P16>>8) & 0xff ;
                Sr[17] = (P17>>8) & 0xff ;
                Sr[18] = (P18>>8) & 0xff ;
                Sr[19] = (P19>>8) & 0xff ;
                Sr[20] = (P20>>8) & 0xff ;
                Sr[21] = (P21>>8) & 0xff ;
                Sr[22] = (P22>>8) & 0xff ;
                Sr[23] = (P23>>8) & 0xff ;
                Sr[24] = (P24>>8) & 0xff ;
                Sr[25] = (P25>>8) & 0xff ;
                Sr[26] = (P26>>8) & 0xff ;
                Sr[27] = (P27>>8) & 0xff ;
                Sr[28] = (P28>>8) & 0xff ;
                Sr[29] = (P29>>8) & 0xff ;
                Sr[30] = (P30>>8) & 0xff ;
                Sr[31] = (P31>>8) & 0xff ;
                rs_decode_palindromic_CUDA(Sr, S, 32, k, errLocs, errMags,
                                           1+(i<<2), Lambda, B, T, Omega) ;
            }
            if ( T0 & (0xff << 16) )
            {
                atomicAdd(&error_count, 1);
                Sr[0] = (P0>>16) & 0xff ;
                Sr[1] = (P1>>16) & 0xff ;
                Sr[2] = (P2>>16) & 0xff ;
                Sr[3] = (P3>>16) & 0xff ;
                Sr[4] = (P4>>16) & 0xff ;
                Sr[5] = (P5>>16) & 0xff ;
                Sr[6] = (P6>>16) & 0xff ;
                Sr[7] = (P7>>16) & 0xff ;
                Sr[8] = (P8>>16) & 0xff ;
                Sr[9] = (P9>>16) & 0xff ;
                Sr[10] = (P10>>16) & 0xff ;
                Sr[11] = (P11>>16) & 0xff ;
                Sr[12] = (P12>>16) & 0xff ;
                Sr[13] = (P13>>16) & 0xff ;
                Sr[14] = (P14>>16) & 0xff ;
                Sr[15] = (P15>>16) & 0xff ;
                Sr[16] = (P16>>16) & 0xff ;
                Sr[17] = (P17>>16) & 0xff ;
                Sr[18] = (P18>>16) & 0xff ;
                Sr[19] = (P19>>16) & 0xff ;
                Sr[20] = (P20>>16) & 0xff ;
                Sr[21] = (P21>>16) & 0xff ;
                Sr[22] = (P22>>16) & 0xff ;
                Sr[23] = (P23>>16) & 0xff ;
                Sr[24] = (P24>>16) & 0xff ;
                Sr[25] = (P25>>16) & 0xff ;
                Sr[26] = (P26>>16) & 0xff ;
                Sr[27] = (P27>>16) & 0xff ;
                Sr[28] = (P28>>16) & 0xff ;
                Sr[29] = (P29>>16) & 0xff ;
                Sr[30] = (P30>>16) & 0xff ;
                Sr[31] = (P31>>16) & 0xff ;
                rs_decode_palindromic_CUDA(Sr, S, 32, k, errLocs, errMags,
                                           2+(i<<2), Lambda, B, T, Omega) ;
            }
            if ( T0 & (0xff << 24) )
            {
                atomicAdd(&error_count, 1);
                Sr[0] = (P0>>24) & 0xff ;
                Sr[1] = (P1>>24) & 0xff ;
                Sr[2] = (P2>>24) & 0xff ;
                Sr[3] = (P3>>24) & 0xff ;
                Sr[4] = (P4>>24) & 0xff ;
                Sr[5] = (P5>>24) & 0xff ;
                Sr[6] = (P6>>24) & 0xff ;
                Sr[7] = (P7>>24) & 0xff ;
                Sr[8] = (P8>>24) & 0xff ;
                Sr[9] = (P9>>24) & 0xff ;
                Sr[10] = (P10>>24) & 0xff ;
                Sr[11] = (P11>>24) & 0xff ;
                Sr[12] = (P12>>24) & 0xff ;
                Sr[13] = (P13>>24) & 0xff ;
                Sr[14] = (P14>>24) & 0xff ;
                Sr[15] = (P15>>24) & 0xff ;
                Sr[16] = (P16>>24) & 0xff ;
                Sr[17] = (P17>>24) & 0xff ;
                Sr[18] = (P18>>24) & 0xff ;
                Sr[19] = (P19>>24) & 0xff ;
                Sr[20] = (P20>>24) & 0xff ;
                Sr[21] = (P21>>24) & 0xff ;
                Sr[22] = (P22>>24) & 0xff ;
                Sr[23] = (P23>>24) & 0xff ;
                Sr[24] = (P24>>24) & 0xff ;
                Sr[25] = (P25>>24) & 0xff ;
                Sr[26] = (P26>>24) & 0xff ;
                Sr[27] = (P27>>24) & 0xff ;
                Sr[28] = (P28>>24) & 0xff ;
                Sr[29] = (P29>>24) & 0xff ;
                Sr[30] = (P30>>24) & 0xff ;
                Sr[31] = (P31>>24) & 0xff ;
                rs_decode_palindromic_CUDA(Sr, S, 32, k, errLocs, errMags,
                                           3+(i<<2), Lambda, B, T, Omega) ;
            }
        }
    }
}

// Single function to access each LFSR Encode device
void ParallelLFSRSequencer_CUDA(int blocks, int threads, int k, int p, int decoder)
{
    switch (p)
    {
    case 2: ParallelLFSRSequencer2_CUDA << <blocks, threads >> > ( k, decoder );
             break ;
    case 3: ParallelLFSRSequencer3_CUDA << <blocks, threads >> > ( k, decoder );
             break ;
    case 4: ParallelLFSRSequencer4_CUDA << <blocks, threads >> > ( k, decoder );
             break ;
    case 5: ParallelLFSRSequencer5_CUDA << <blocks, threads >> > ( k, decoder );
             break ;
    case 6: ParallelLFSRSequencer6_CUDA << <blocks, threads >> > ( k, decoder );
             break ;
    case 7: ParallelLFSRSequencer7_CUDA << <blocks, threads >> > ( k, decoder );
             break ;
    case 8: ParallelLFSRSequencer8_CUDA << <blocks, threads >> > ( k, decoder );
             break ;
    case 9: ParallelLFSRSequencer9_CUDA << <blocks, threads >> > ( k, decoder );
             break ;
    case 10: ParallelLFSRSequencer10_CUDA << <blocks, threads >> > ( k, decoder );
             break ;
    case 11: ParallelLFSRSequencer11_CUDA << <blocks, threads >> > ( k, decoder );
             break ;
    case 12: ParallelLFSRSequencer12_CUDA << <blocks, threads >> > ( k, decoder );
             break ;
    case 13: ParallelLFSRSequencer13_CUDA << <blocks, threads >> > ( k, decoder );
             break ;
    case 14: ParallelLFSRSequencer14_CUDA << <blocks, threads >> > ( k, decoder );
             break ;
    case 15: ParallelLFSRSequencer15_CUDA << <blocks, threads >> > ( k, decoder );
             break ;
    case 16: ParallelLFSRSequencer16_CUDA << <blocks, threads >> > ( k, decoder );
             break ;
    case 17: ParallelLFSRSequencer17_CUDA << <blocks, threads >> > ( k, decoder );
             break ;
    case 18: ParallelLFSRSequencer18_CUDA << <blocks, threads >> > ( k, decoder );
             break ;
    case 19: ParallelLFSRSequencer19_CUDA << <blocks, threads >> > ( k, decoder );
             break ;
    case 20: ParallelLFSRSequencer20_CUDA << <blocks, threads >> > ( k, decoder );
             break ;
    case 21: ParallelLFSRSequencer21_CUDA << <blocks, threads >> > ( k, decoder );
             break ;
    case 22: ParallelLFSRSequencer22_CUDA << <blocks, threads >> > ( k, decoder );
             break ;
    case 23: ParallelLFSRSequencer23_CUDA << <blocks, threads >> > ( k, decoder );
             break ;
    case 24: ParallelLFSRSequencer24_CUDA << <blocks, threads >> > ( k, decoder );
             break ;
    case 25: ParallelLFSRSequencer25_CUDA << <blocks, threads >> > ( k, decoder );
             break ;
    case 26: ParallelLFSRSequencer26_CUDA << <blocks, threads >> > ( k, decoder );
             break ;
    case 27: ParallelLFSRSequencer27_CUDA << <blocks, threads >> > ( k, decoder );
             break ;
    case 28: ParallelLFSRSequencer28_CUDA << <blocks, threads >> > ( k, decoder );
             break ;
    case 29: ParallelLFSRSequencer29_CUDA << <blocks, threads >> > ( k, decoder );
             break ;
    case 30: ParallelLFSRSequencer30_CUDA << <blocks, threads >> > ( k, decoder );
             break ;
    case 31: ParallelLFSRSequencer31_CUDA << <blocks, threads >> > ( k, decoder );
             break ;
    case 32: ParallelLFSRSequencer32_CUDA << <blocks, threads >> > ( k, decoder );
             break ;
    }
}
