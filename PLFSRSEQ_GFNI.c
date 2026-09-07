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

#include "PLFSRSEQ_GFNI.h"
uint64_t PCErrCnt = 0;                 // Host side error count for GFNI decoder errors

// Parallel LFSRD_SR Sequencer for P = 2 Codewords
void ParallelLFSRSequencer2_GFNI(int len, int k, unsigned char **data, int decoder)
{
    int curSym, curPos ;                           // Loop counters
    __m512i parity [ 2 ], taps [ 1 ] ;             // Parity registers
    __m512i data_vec, temp [ 1 ] ;
    unsigned char **sPnt ;                         // Data lookup pointers

    // Initialize the taps to the LFSR
   taps [ 0 ] = _mm512_broadcast_i32x2(*( __m128i * ) &PCAffTab[0x49]);

    // Adjust K if we are decoding
    k += decoder ? 2 : 0 ;

    // Loop through each 64 byte codeword
    for ( curPos = 0 ; curPos < len ; curPos += 64 )
    {
        sPnt = data ;
        parity [ 0 ] = _mm512_setzero_si512();
        parity [ 1 ] = _mm512_setzero_si512();

        // Loop through all the 0..k symbols
        for ( curSym = 0 ; curSym < k ; curSym ++ )
        {
            // Load next 64 bytes of Original Data
            data_vec = _mm512_stream_load_si512( *sPnt + curPos ) ;
          __builtin_prefetch ( *sPnt + curPos + 64, 0, 3 ) ;
            sPnt ++ ;

            // Add incoming data to MSB of parity, then update parities using Parallel Multiplier
            data_vec = _mm512_xor_si512( data_vec, parity [ 0 ] ) ;

            // Compute the first half of the tap values for palindromic code generator
            temp [ 0 ] = _mm512_gf2p8affine_epi64_epi8 ( data_vec, taps [ 0 ], 0 ) ;

            // Now cycle the LFSR
            parity [ 0 ] = _mm512_xor_si512 ( parity [ 1 ], temp [ 0 ] ) ;
            parity [ 1 ] = data_vec ;
        }

        if ( decoder == 0 )
        {
            // Store parity back to memory
            _mm512_stream_si512( (&data [ k+0 ][curPos]), parity [ 0 ] ) ;
            _mm512_stream_si512( (&data [ k+1 ][curPos]), parity [ 1 ] ) ;
        }
        else
        {
            // Verify Syndromes are zero
            data_vec = _mm512_or_si512 ( parity [ 0 ], parity [ 1 ] ) ;
            __mmask64 mask = _mm512_test_epi8_mask ( data_vec, data_vec ) ;

            // If syndromes are not zero then process error
            while ( !_ktestz_mask64_u8 ( mask, mask ) ) 
            {
                PCErrCnt++;
                unsigned char Sr[2], S[2], errLocs[2], errMags[2] ;
                unsigned char Lambda[2+1], B[2+1], T[2+1], Omega[2+1] ;
                unsigned char PTemp[64] ;
                uint64_t pos = _tzcnt_u64 ( mask ) ;

                _mm512_storeu_si512 ( PTemp, parity[0] ) ;
                Sr[0] = PTemp[pos];
                _mm512_storeu_si512 ( PTemp, parity[1] ) ;
                Sr[1] = PTemp[pos];
                rs_decode_palindromic_GFNI(Sr, S, 2, k, errLocs, errMags,
                             (int)pos+curPos, Lambda, B, T, Omega, data) ;
                mask = _blsr_u64(mask);
            }
        }
    }
}

// Parallel LFSRD_SR Sequencer for P = 3 Codewords
void ParallelLFSRSequencer3_GFNI(int len, int k, unsigned char **data, int decoder)
{
    int curSym, curPos ;                           // Loop counters
    __m512i parity [ 3 ], taps [ 1 ] ;             // Parity registers
    __m512i data_vec, temp [ 1 ] ;
    unsigned char **sPnt ;                         // Data lookup pointers

    // Initialize the taps to the LFSR
   taps [ 0 ] = _mm512_broadcast_i32x2(*( __m128i * ) &PCAffTab[0x8d]);

    // Adjust K if we are decoding
    k += decoder ? 3 : 0 ;

    // Loop through each 64 byte codeword
    for ( curPos = 0 ; curPos < len ; curPos += 64 )
    {
        sPnt = data ;
        parity [ 0 ] = _mm512_setzero_si512();
        parity [ 1 ] = _mm512_setzero_si512();
        parity [ 2 ] = _mm512_setzero_si512();

        // Loop through all the 0..k symbols
        for ( curSym = 0 ; curSym < k ; curSym ++ )
        {
            // Load next 64 bytes of Original Data
            data_vec = _mm512_stream_load_si512( *sPnt + curPos ) ;
          __builtin_prefetch ( *sPnt + curPos + 64, 0, 3 ) ;
            sPnt ++ ;

            // Add incoming data to MSB of parity, then update parities using Parallel Multiplier
            data_vec = _mm512_xor_si512( data_vec, parity [ 0 ] ) ;

            // Compute the first half of the tap values for palindromic code generator
            temp [ 0 ] = _mm512_gf2p8affine_epi64_epi8 ( data_vec, taps [ 0 ], 0 ) ;

            // Now cycle the LFSR
            parity [ 0 ] = _mm512_xor_si512 ( parity [ 1 ], temp [ 0 ] ) ;
            parity [ 1 ] = _mm512_xor_si512 ( parity [ 2 ], temp [ 0 ] ) ;
            parity [ 2 ] = data_vec ;
        }

        if ( decoder == 0 )
        {
            // Store parity back to memory
            _mm512_stream_si512( (&data [ k+0 ][curPos]), parity [ 0 ] ) ;
            _mm512_stream_si512( (&data [ k+1 ][curPos]), parity [ 1 ] ) ;
            _mm512_stream_si512( (&data [ k+2 ][curPos]), parity [ 2 ] ) ;
        }
        else
        {
            // Verify Syndromes are zero
            data_vec = _mm512_or_si512 ( parity [ 0 ], parity [ 1 ] ) ;
            data_vec = _mm512_ternarylogic_epi32 ( data_vec, parity [ 2 ], parity [ 2 ], 0xFE ) ;
            __mmask64 mask = _mm512_test_epi8_mask ( data_vec, data_vec ) ;

            // If syndromes are not zero then process error
            while ( !_ktestz_mask64_u8 ( mask, mask ) ) 
            {
                PCErrCnt++;
                unsigned char Sr[3], S[3], errLocs[3], errMags[3] ;
                unsigned char Lambda[3+1], B[3+1], T[3+1], Omega[3+1] ;
                unsigned char PTemp[64] ;
                uint64_t pos = _tzcnt_u64 ( mask ) ;

                _mm512_storeu_si512 ( PTemp, parity[0] ) ;
                Sr[0] = PTemp[pos];
                _mm512_storeu_si512 ( PTemp, parity[1] ) ;
                Sr[1] = PTemp[pos];
                _mm512_storeu_si512 ( PTemp, parity[2] ) ;
                Sr[2] = PTemp[pos];
                rs_decode_palindromic_GFNI(Sr, S, 3, k, errLocs, errMags,
                             (int)pos+curPos, Lambda, B, T, Omega, data) ;
                mask = _blsr_u64(mask);
            }
        }
    }
}

// Parallel LFSRD_SR Sequencer for P = 4 Codewords
void ParallelLFSRSequencer4_GFNI(int len, int k, unsigned char **data, int decoder)
{
    int curSym, curPos ;                           // Loop counters
    __m512i parity [ 4 ], taps [ 2 ] ;             // Parity registers
    __m512i data_vec, temp [ 2 ] ;
    unsigned char **sPnt ;                         // Data lookup pointers

    // Initialize the taps to the LFSR
   taps [ 0 ] = _mm512_broadcast_i32x2(*( __m128i * ) &PCAffTab[0x38]);
   taps [ 1 ] = _mm512_broadcast_i32x2(*( __m128i * ) &PCAffTab[0xcf]);

    // Adjust K if we are decoding
    k += decoder ? 4 : 0 ;

    // Loop through each 64 byte codeword
    for ( curPos = 0 ; curPos < len ; curPos += 64 )
    {
        sPnt = data ;
        parity [ 0 ] = _mm512_setzero_si512();
        parity [ 1 ] = _mm512_setzero_si512();
        parity [ 2 ] = _mm512_setzero_si512();
        parity [ 3 ] = _mm512_setzero_si512();

        // Loop through all the 0..k symbols
        for ( curSym = 0 ; curSym < k ; curSym ++ )
        {
            // Load next 64 bytes of Original Data
            data_vec = _mm512_stream_load_si512( *sPnt + curPos ) ;
          __builtin_prefetch ( *sPnt + curPos + 64, 0, 3 ) ;
            sPnt ++ ;

            // Add incoming data to MSB of parity, then update parities using Parallel Multiplier
            data_vec = _mm512_xor_si512( data_vec, parity [ 0 ] ) ;

            // Compute the first half of the tap values for palindromic code generator
            temp [ 0 ] = _mm512_gf2p8affine_epi64_epi8 ( data_vec, taps [ 0 ], 0 ) ;
            temp [ 1 ] = _mm512_gf2p8affine_epi64_epi8 ( data_vec, taps [ 1 ], 0 ) ;

            // Now cycle the LFSR
            parity [ 0 ] = _mm512_xor_si512 ( parity [ 1 ], temp [ 0 ] ) ;
            parity [ 1 ] = _mm512_xor_si512 ( parity [ 2 ], temp [ 1 ] ) ;
            parity [ 2 ] = _mm512_xor_si512 ( parity [ 3 ], temp [ 0 ] ) ;
            parity [ 3 ] = data_vec ;
        }

        if ( decoder == 0 )
        {
            // Store parity back to memory
            _mm512_stream_si512( (&data [ k+0 ][curPos]), parity [ 0 ] ) ;
            _mm512_stream_si512( (&data [ k+1 ][curPos]), parity [ 1 ] ) ;
            _mm512_stream_si512( (&data [ k+2 ][curPos]), parity [ 2 ] ) ;
            _mm512_stream_si512( (&data [ k+3 ][curPos]), parity [ 3 ] ) ;
        }
        else
        {
            // Verify Syndromes are zero
            data_vec = _mm512_or_si512 ( parity [ 0 ], parity [ 1 ] ) ;
            data_vec = _mm512_ternarylogic_epi32 ( data_vec, parity [ 2 ], parity [ 3 ], 0xFE ) ;
            __mmask64 mask = _mm512_test_epi8_mask ( data_vec, data_vec ) ;

            // If syndromes are not zero then process error
            while ( !_ktestz_mask64_u8 ( mask, mask ) ) 
            {
                PCErrCnt++;
                unsigned char Sr[4], S[4], errLocs[4], errMags[4] ;
                unsigned char Lambda[4+1], B[4+1], T[4+1], Omega[4+1] ;
                unsigned char PTemp[64] ;
                uint64_t pos = _tzcnt_u64 ( mask ) ;

                _mm512_storeu_si512 ( PTemp, parity[0] ) ;
                Sr[0] = PTemp[pos];
                _mm512_storeu_si512 ( PTemp, parity[1] ) ;
                Sr[1] = PTemp[pos];
                _mm512_storeu_si512 ( PTemp, parity[2] ) ;
                Sr[2] = PTemp[pos];
                _mm512_storeu_si512 ( PTemp, parity[3] ) ;
                Sr[3] = PTemp[pos];
                rs_decode_palindromic_GFNI(Sr, S, 4, k, errLocs, errMags,
                             (int)pos+curPos, Lambda, B, T, Omega, data) ;
                mask = _blsr_u64(mask);
            }
        }
    }
}

// Parallel LFSRD_SR Sequencer for P = 5 Codewords
void ParallelLFSRSequencer5_GFNI(int len, int k, unsigned char **data, int decoder)
{
    int curSym, curPos ;                           // Loop counters
    __m512i parity [ 5 ], taps [ 2 ] ;             // Parity registers
    __m512i data_vec, temp [ 2 ] ;
    unsigned char **sPnt ;                         // Data lookup pointers

    // Initialize the taps to the LFSR
   taps [ 0 ] = _mm512_broadcast_i32x2(*( __m128i * ) &PCAffTab[0xce]);
   taps [ 1 ] = _mm512_broadcast_i32x2(*( __m128i * ) &PCAffTab[0xe6]);

    // Adjust K if we are decoding
    k += decoder ? 5 : 0 ;

    // Loop through each 64 byte codeword
    for ( curPos = 0 ; curPos < len ; curPos += 64 )
    {
        sPnt = data ;
        parity [ 0 ] = _mm512_setzero_si512();
        parity [ 1 ] = _mm512_setzero_si512();
        parity [ 2 ] = _mm512_setzero_si512();
        parity [ 3 ] = _mm512_setzero_si512();
        parity [ 4 ] = _mm512_setzero_si512();

        // Loop through all the 0..k symbols
        for ( curSym = 0 ; curSym < k ; curSym ++ )
        {
            // Load next 64 bytes of Original Data
            data_vec = _mm512_stream_load_si512( *sPnt + curPos ) ;
          __builtin_prefetch ( *sPnt + curPos + 64, 0, 3 ) ;
            sPnt ++ ;

            // Add incoming data to MSB of parity, then update parities using Parallel Multiplier
            data_vec = _mm512_xor_si512( data_vec, parity [ 0 ] ) ;

            // Compute the first half of the tap values for palindromic code generator
            temp [ 0 ] = _mm512_gf2p8affine_epi64_epi8 ( data_vec, taps [ 0 ], 0 ) ;
            temp [ 1 ] = _mm512_gf2p8affine_epi64_epi8 ( data_vec, taps [ 1 ], 0 ) ;

            // Now cycle the LFSR
            parity [ 0 ] = _mm512_xor_si512 ( parity [ 1 ], temp [ 0 ] ) ;
            parity [ 1 ] = _mm512_xor_si512 ( parity [ 2 ], temp [ 1 ] ) ;
            parity [ 2 ] = _mm512_xor_si512 ( parity [ 3 ], temp [ 1 ] ) ;
            parity [ 3 ] = _mm512_xor_si512 ( parity [ 4 ], temp [ 0 ] ) ;
            parity [ 4 ] = data_vec ;
        }

        if ( decoder == 0 )
        {
            // Store parity back to memory
            _mm512_stream_si512( (&data [ k+0 ][curPos]), parity [ 0 ] ) ;
            _mm512_stream_si512( (&data [ k+1 ][curPos]), parity [ 1 ] ) ;
            _mm512_stream_si512( (&data [ k+2 ][curPos]), parity [ 2 ] ) ;
            _mm512_stream_si512( (&data [ k+3 ][curPos]), parity [ 3 ] ) ;
            _mm512_stream_si512( (&data [ k+4 ][curPos]), parity [ 4 ] ) ;
        }
        else
        {
            // Verify Syndromes are zero
            data_vec = _mm512_or_si512 ( parity [ 0 ], parity [ 1 ] ) ;
            data_vec = _mm512_ternarylogic_epi32 ( data_vec, parity [ 2 ], parity [ 3 ], 0xFE ) ;
            data_vec = _mm512_ternarylogic_epi32 ( data_vec, parity [ 4 ], parity [ 4 ], 0xFE ) ;
            __mmask64 mask = _mm512_test_epi8_mask ( data_vec, data_vec ) ;

            // If syndromes are not zero then process error
            while ( !_ktestz_mask64_u8 ( mask, mask ) ) 
            {
                PCErrCnt++;
                unsigned char Sr[5], S[5], errLocs[5], errMags[5] ;
                unsigned char Lambda[5+1], B[5+1], T[5+1], Omega[5+1] ;
                unsigned char PTemp[64] ;
                uint64_t pos = _tzcnt_u64 ( mask ) ;

                _mm512_storeu_si512 ( PTemp, parity[0] ) ;
                Sr[0] = PTemp[pos];
                _mm512_storeu_si512 ( PTemp, parity[1] ) ;
                Sr[1] = PTemp[pos];
                _mm512_storeu_si512 ( PTemp, parity[2] ) ;
                Sr[2] = PTemp[pos];
                _mm512_storeu_si512 ( PTemp, parity[3] ) ;
                Sr[3] = PTemp[pos];
                _mm512_storeu_si512 ( PTemp, parity[4] ) ;
                Sr[4] = PTemp[pos];
                rs_decode_palindromic_GFNI(Sr, S, 5, k, errLocs, errMags,
                             (int)pos+curPos, Lambda, B, T, Omega, data) ;
                mask = _blsr_u64(mask);
            }
        }
    }
}

// Parallel LFSRD_SR Sequencer for P = 6 Codewords
void ParallelLFSRSequencer6_GFNI(int len, int k, unsigned char **data, int decoder)
{
    int curSym, curPos ;                           // Loop counters
    __m512i parity [ 6 ], taps [ 3 ] ;             // Parity registers
    __m512i data_vec, temp [ 3 ] ;
    unsigned char **sPnt ;                         // Data lookup pointers

    // Initialize the taps to the LFSR
   taps [ 0 ] = _mm512_broadcast_i32x2(*( __m128i * ) &PCAffTab[0x25]);
   taps [ 1 ] = _mm512_broadcast_i32x2(*( __m128i * ) &PCAffTab[0x6c]);
   taps [ 2 ] = _mm512_broadcast_i32x2(*( __m128i * ) &PCAffTab[0x8e]);

    // Adjust K if we are decoding
    k += decoder ? 6 : 0 ;

    // Loop through each 64 byte codeword
    for ( curPos = 0 ; curPos < len ; curPos += 64 )
    {
        sPnt = data ;
        parity [ 0 ] = _mm512_setzero_si512();
        parity [ 1 ] = _mm512_setzero_si512();
        parity [ 2 ] = _mm512_setzero_si512();
        parity [ 3 ] = _mm512_setzero_si512();
        parity [ 4 ] = _mm512_setzero_si512();
        parity [ 5 ] = _mm512_setzero_si512();

        // Loop through all the 0..k symbols
        for ( curSym = 0 ; curSym < k ; curSym ++ )
        {
            // Load next 64 bytes of Original Data
            data_vec = _mm512_stream_load_si512( *sPnt + curPos ) ;
          __builtin_prefetch ( *sPnt + curPos + 64, 0, 3 ) ;
            sPnt ++ ;

            // Add incoming data to MSB of parity, then update parities using Parallel Multiplier
            data_vec = _mm512_xor_si512( data_vec, parity [ 0 ] ) ;

            // Compute the first half of the tap values for palindromic code generator
            temp [ 0 ] = _mm512_gf2p8affine_epi64_epi8 ( data_vec, taps [ 0 ], 0 ) ;
            temp [ 1 ] = _mm512_gf2p8affine_epi64_epi8 ( data_vec, taps [ 1 ], 0 ) ;
            temp [ 2 ] = _mm512_gf2p8affine_epi64_epi8 ( data_vec, taps [ 2 ], 0 ) ;

            // Now cycle the LFSR
            parity [ 0 ] = _mm512_xor_si512 ( parity [ 1 ], temp [ 0 ] ) ;
            parity [ 1 ] = _mm512_xor_si512 ( parity [ 2 ], temp [ 1 ] ) ;
            parity [ 2 ] = _mm512_xor_si512 ( parity [ 3 ], temp [ 2 ] ) ;
            parity [ 3 ] = _mm512_xor_si512 ( parity [ 4 ], temp [ 1 ] ) ;
            parity [ 4 ] = _mm512_xor_si512 ( parity [ 5 ], temp [ 0 ] ) ;
            parity [ 5 ] = data_vec ;
        }

        if ( decoder == 0 )
        {
            // Store parity back to memory
            _mm512_stream_si512( (&data [ k+0 ][curPos]), parity [ 0 ] ) ;
            _mm512_stream_si512( (&data [ k+1 ][curPos]), parity [ 1 ] ) ;
            _mm512_stream_si512( (&data [ k+2 ][curPos]), parity [ 2 ] ) ;
            _mm512_stream_si512( (&data [ k+3 ][curPos]), parity [ 3 ] ) ;
            _mm512_stream_si512( (&data [ k+4 ][curPos]), parity [ 4 ] ) ;
            _mm512_stream_si512( (&data [ k+5 ][curPos]), parity [ 5 ] ) ;
        }
        else
        {
            // Verify Syndromes are zero
            data_vec = _mm512_or_si512 ( parity [ 0 ], parity [ 1 ] ) ;
            data_vec = _mm512_ternarylogic_epi32 ( data_vec, parity [ 2 ], parity [ 3 ], 0xFE ) ;
            data_vec = _mm512_ternarylogic_epi32 ( data_vec, parity [ 4 ], parity [ 5 ], 0xFE ) ;
            __mmask64 mask = _mm512_test_epi8_mask ( data_vec, data_vec ) ;

            // If syndromes are not zero then process error
            while ( !_ktestz_mask64_u8 ( mask, mask ) ) 
            {
                PCErrCnt++;
                unsigned char Sr[6], S[6], errLocs[6], errMags[6] ;
                unsigned char Lambda[6+1], B[6+1], T[6+1], Omega[6+1] ;
                unsigned char PTemp[64] ;
                uint64_t pos = _tzcnt_u64 ( mask ) ;

                _mm512_storeu_si512 ( PTemp, parity[0] ) ;
                Sr[0] = PTemp[pos];
                _mm512_storeu_si512 ( PTemp, parity[1] ) ;
                Sr[1] = PTemp[pos];
                _mm512_storeu_si512 ( PTemp, parity[2] ) ;
                Sr[2] = PTemp[pos];
                _mm512_storeu_si512 ( PTemp, parity[3] ) ;
                Sr[3] = PTemp[pos];
                _mm512_storeu_si512 ( PTemp, parity[4] ) ;
                Sr[4] = PTemp[pos];
                _mm512_storeu_si512 ( PTemp, parity[5] ) ;
                Sr[5] = PTemp[pos];
                rs_decode_palindromic_GFNI(Sr, S, 6, k, errLocs, errMags,
                             (int)pos+curPos, Lambda, B, T, Omega, data) ;
                mask = _blsr_u64(mask);
            }
        }
    }
}

// Parallel LFSRD_SR Sequencer for P = 7 Codewords
void ParallelLFSRSequencer7_GFNI(int len, int k, unsigned char **data, int decoder)
{
    int curSym, curPos ;                           // Loop counters
    __m512i parity [ 7 ], taps [ 3 ] ;             // Parity registers
    __m512i data_vec, temp [ 3 ] ;
    unsigned char **sPnt ;                         // Data lookup pointers

    // Initialize the taps to the LFSR
   taps [ 0 ] = _mm512_broadcast_i32x2(*( __m128i * ) &PCAffTab[0x6b]);
   taps [ 1 ] = _mm512_broadcast_i32x2(*( __m128i * ) &PCAffTab[0x9]);
   taps [ 2 ] = _mm512_broadcast_i32x2(*( __m128i * ) &PCAffTab[0x9e]);

    // Adjust K if we are decoding
    k += decoder ? 7 : 0 ;

    // Loop through each 64 byte codeword
    for ( curPos = 0 ; curPos < len ; curPos += 64 )
    {
        sPnt = data ;
        parity [ 0 ] = _mm512_setzero_si512();
        parity [ 1 ] = _mm512_setzero_si512();
        parity [ 2 ] = _mm512_setzero_si512();
        parity [ 3 ] = _mm512_setzero_si512();
        parity [ 4 ] = _mm512_setzero_si512();
        parity [ 5 ] = _mm512_setzero_si512();
        parity [ 6 ] = _mm512_setzero_si512();

        // Loop through all the 0..k symbols
        for ( curSym = 0 ; curSym < k ; curSym ++ )
        {
            // Load next 64 bytes of Original Data
            data_vec = _mm512_stream_load_si512( *sPnt + curPos ) ;
          __builtin_prefetch ( *sPnt + curPos + 64, 0, 3 ) ;
            sPnt ++ ;

            // Add incoming data to MSB of parity, then update parities using Parallel Multiplier
            data_vec = _mm512_xor_si512( data_vec, parity [ 0 ] ) ;

            // Compute the first half of the tap values for palindromic code generator
            temp [ 0 ] = _mm512_gf2p8affine_epi64_epi8 ( data_vec, taps [ 0 ], 0 ) ;
            temp [ 1 ] = _mm512_gf2p8affine_epi64_epi8 ( data_vec, taps [ 1 ], 0 ) ;
            temp [ 2 ] = _mm512_gf2p8affine_epi64_epi8 ( data_vec, taps [ 2 ], 0 ) ;

            // Now cycle the LFSR
            parity [ 0 ] = _mm512_xor_si512 ( parity [ 1 ], temp [ 0 ] ) ;
            parity [ 1 ] = _mm512_xor_si512 ( parity [ 2 ], temp [ 1 ] ) ;
            parity [ 2 ] = _mm512_xor_si512 ( parity [ 3 ], temp [ 2 ] ) ;
            parity [ 3 ] = _mm512_xor_si512 ( parity [ 4 ], temp [ 2 ] ) ;
            parity [ 4 ] = _mm512_xor_si512 ( parity [ 5 ], temp [ 1 ] ) ;
            parity [ 5 ] = _mm512_xor_si512 ( parity [ 6 ], temp [ 0 ] ) ;
            parity [ 6 ] = data_vec ;
        }

        if ( decoder == 0 )
        {
            // Store parity back to memory
            _mm512_stream_si512( (&data [ k+0 ][curPos]), parity [ 0 ] ) ;
            _mm512_stream_si512( (&data [ k+1 ][curPos]), parity [ 1 ] ) ;
            _mm512_stream_si512( (&data [ k+2 ][curPos]), parity [ 2 ] ) ;
            _mm512_stream_si512( (&data [ k+3 ][curPos]), parity [ 3 ] ) ;
            _mm512_stream_si512( (&data [ k+4 ][curPos]), parity [ 4 ] ) ;
            _mm512_stream_si512( (&data [ k+5 ][curPos]), parity [ 5 ] ) ;
            _mm512_stream_si512( (&data [ k+6 ][curPos]), parity [ 6 ] ) ;
        }
        else
        {
            // Verify Syndromes are zero
            data_vec = _mm512_or_si512 ( parity [ 0 ], parity [ 1 ] ) ;
            data_vec = _mm512_ternarylogic_epi32 ( data_vec, parity [ 2 ], parity [ 3 ], 0xFE ) ;
            data_vec = _mm512_ternarylogic_epi32 ( data_vec, parity [ 4 ], parity [ 5 ], 0xFE ) ;
            data_vec = _mm512_ternarylogic_epi32 ( data_vec, parity [ 6 ], parity [ 6 ], 0xFE ) ;
            __mmask64 mask = _mm512_test_epi8_mask ( data_vec, data_vec ) ;

            // If syndromes are not zero then process error
            while ( !_ktestz_mask64_u8 ( mask, mask ) ) 
            {
                PCErrCnt++;
                unsigned char Sr[7], S[7], errLocs[7], errMags[7] ;
                unsigned char Lambda[7+1], B[7+1], T[7+1], Omega[7+1] ;
                unsigned char PTemp[64] ;
                uint64_t pos = _tzcnt_u64 ( mask ) ;

                _mm512_storeu_si512 ( PTemp, parity[0] ) ;
                Sr[0] = PTemp[pos];
                _mm512_storeu_si512 ( PTemp, parity[1] ) ;
                Sr[1] = PTemp[pos];
                _mm512_storeu_si512 ( PTemp, parity[2] ) ;
                Sr[2] = PTemp[pos];
                _mm512_storeu_si512 ( PTemp, parity[3] ) ;
                Sr[3] = PTemp[pos];
                _mm512_storeu_si512 ( PTemp, parity[4] ) ;
                Sr[4] = PTemp[pos];
                _mm512_storeu_si512 ( PTemp, parity[5] ) ;
                Sr[5] = PTemp[pos];
                _mm512_storeu_si512 ( PTemp, parity[6] ) ;
                Sr[6] = PTemp[pos];
                rs_decode_palindromic_GFNI(Sr, S, 7, k, errLocs, errMags,
                             (int)pos+curPos, Lambda, B, T, Omega, data) ;
                mask = _blsr_u64(mask);
            }
        }
    }
}

// Parallel LFSRD_SR Sequencer for P = 8 Codewords
void ParallelLFSRSequencer8_GFNI(int len, int k, unsigned char **data, int decoder)
{
    int curSym, curPos ;                           // Loop counters
    __m512i parity [ 8 ], taps [ 4 ] ;             // Parity registers
    __m512i data_vec, temp [ 4 ] ;
    unsigned char **sPnt ;                         // Data lookup pointers

    // Initialize the taps to the LFSR
   taps [ 0 ] = _mm512_broadcast_i32x2(*( __m128i * ) &PCAffTab[0xee]);
   taps [ 1 ] = _mm512_broadcast_i32x2(*( __m128i * ) &PCAffTab[0xf5]);
   taps [ 2 ] = _mm512_broadcast_i32x2(*( __m128i * ) &PCAffTab[0x5e]);
   taps [ 3 ] = _mm512_broadcast_i32x2(*( __m128i * ) &PCAffTab[0xeb]);

    // Adjust K if we are decoding
    k += decoder ? 8 : 0 ;

    // Loop through each 64 byte codeword
    for ( curPos = 0 ; curPos < len ; curPos += 64 )
    {
        sPnt = data ;
        parity [ 0 ] = _mm512_setzero_si512();
        parity [ 1 ] = _mm512_setzero_si512();
        parity [ 2 ] = _mm512_setzero_si512();
        parity [ 3 ] = _mm512_setzero_si512();
        parity [ 4 ] = _mm512_setzero_si512();
        parity [ 5 ] = _mm512_setzero_si512();
        parity [ 6 ] = _mm512_setzero_si512();
        parity [ 7 ] = _mm512_setzero_si512();

        // Loop through all the 0..k symbols
        for ( curSym = 0 ; curSym < k ; curSym ++ )
        {
            // Load next 64 bytes of Original Data
            data_vec = _mm512_stream_load_si512( *sPnt + curPos ) ;
          __builtin_prefetch ( *sPnt + curPos + 64, 0, 3 ) ;
            sPnt ++ ;

            // Add incoming data to MSB of parity, then update parities using Parallel Multiplier
            data_vec = _mm512_xor_si512( data_vec, parity [ 0 ] ) ;

            // Compute the first half of the tap values for palindromic code generator
            temp [ 0 ] = _mm512_gf2p8affine_epi64_epi8 ( data_vec, taps [ 0 ], 0 ) ;
            temp [ 1 ] = _mm512_gf2p8affine_epi64_epi8 ( data_vec, taps [ 1 ], 0 ) ;
            temp [ 2 ] = _mm512_gf2p8affine_epi64_epi8 ( data_vec, taps [ 2 ], 0 ) ;
            temp [ 3 ] = _mm512_gf2p8affine_epi64_epi8 ( data_vec, taps [ 3 ], 0 ) ;

            // Now cycle the LFSR
            parity [ 0 ] = _mm512_xor_si512 ( parity [ 1 ], temp [ 0 ] ) ;
            parity [ 1 ] = _mm512_xor_si512 ( parity [ 2 ], temp [ 1 ] ) ;
            parity [ 2 ] = _mm512_xor_si512 ( parity [ 3 ], temp [ 2 ] ) ;
            parity [ 3 ] = _mm512_xor_si512 ( parity [ 4 ], temp [ 3 ] ) ;
            parity [ 4 ] = _mm512_xor_si512 ( parity [ 5 ], temp [ 2 ] ) ;
            parity [ 5 ] = _mm512_xor_si512 ( parity [ 6 ], temp [ 1 ] ) ;
            parity [ 6 ] = _mm512_xor_si512 ( parity [ 7 ], temp [ 0 ] ) ;
            parity [ 7 ] = data_vec ;
        }

        if ( decoder == 0 )
        {
            // Store parity back to memory
            _mm512_stream_si512( (&data [ k+0 ][curPos]), parity [ 0 ] ) ;
            _mm512_stream_si512( (&data [ k+1 ][curPos]), parity [ 1 ] ) ;
            _mm512_stream_si512( (&data [ k+2 ][curPos]), parity [ 2 ] ) ;
            _mm512_stream_si512( (&data [ k+3 ][curPos]), parity [ 3 ] ) ;
            _mm512_stream_si512( (&data [ k+4 ][curPos]), parity [ 4 ] ) ;
            _mm512_stream_si512( (&data [ k+5 ][curPos]), parity [ 5 ] ) ;
            _mm512_stream_si512( (&data [ k+6 ][curPos]), parity [ 6 ] ) ;
            _mm512_stream_si512( (&data [ k+7 ][curPos]), parity [ 7 ] ) ;
        }
        else
        {
            // Verify Syndromes are zero
            data_vec = _mm512_or_si512 ( parity [ 0 ], parity [ 1 ] ) ;
            data_vec = _mm512_ternarylogic_epi32 ( data_vec, parity [ 2 ], parity [ 3 ], 0xFE ) ;
            data_vec = _mm512_ternarylogic_epi32 ( data_vec, parity [ 4 ], parity [ 5 ], 0xFE ) ;
            data_vec = _mm512_ternarylogic_epi32 ( data_vec, parity [ 6 ], parity [ 7 ], 0xFE ) ;
            __mmask64 mask = _mm512_test_epi8_mask ( data_vec, data_vec ) ;

            // If syndromes are not zero then process error
            while ( !_ktestz_mask64_u8 ( mask, mask ) ) 
            {
                PCErrCnt++;
                unsigned char Sr[8], S[8], errLocs[8], errMags[8] ;
                unsigned char Lambda[8+1], B[8+1], T[8+1], Omega[8+1] ;
                unsigned char PTemp[64] ;
                uint64_t pos = _tzcnt_u64 ( mask ) ;

                _mm512_storeu_si512 ( PTemp, parity[0] ) ;
                Sr[0] = PTemp[pos];
                _mm512_storeu_si512 ( PTemp, parity[1] ) ;
                Sr[1] = PTemp[pos];
                _mm512_storeu_si512 ( PTemp, parity[2] ) ;
                Sr[2] = PTemp[pos];
                _mm512_storeu_si512 ( PTemp, parity[3] ) ;
                Sr[3] = PTemp[pos];
                _mm512_storeu_si512 ( PTemp, parity[4] ) ;
                Sr[4] = PTemp[pos];
                _mm512_storeu_si512 ( PTemp, parity[5] ) ;
                Sr[5] = PTemp[pos];
                _mm512_storeu_si512 ( PTemp, parity[6] ) ;
                Sr[6] = PTemp[pos];
                _mm512_storeu_si512 ( PTemp, parity[7] ) ;
                Sr[7] = PTemp[pos];
                rs_decode_palindromic_GFNI(Sr, S, 8, k, errLocs, errMags,
                             (int)pos+curPos, Lambda, B, T, Omega, data) ;
                mask = _blsr_u64(mask);
            }
        }
    }
}

// Parallel LFSRD_SR Sequencer for P = 9 Codewords
void ParallelLFSRSequencer9_GFNI(int len, int k, unsigned char **data, int decoder)
{
    int curSym, curPos ;                           // Loop counters
    __m512i parity [ 9 ], taps [ 4 ] ;             // Parity registers
    __m512i data_vec, temp [ 4 ] ;
    unsigned char **sPnt ;                         // Data lookup pointers

    // Initialize the taps to the LFSR
   taps [ 0 ] = _mm512_broadcast_i32x2(*( __m128i * ) &PCAffTab[0xa3]);
   taps [ 1 ] = _mm512_broadcast_i32x2(*( __m128i * ) &PCAffTab[0xb]);
   taps [ 2 ] = _mm512_broadcast_i32x2(*( __m128i * ) &PCAffTab[0x33]);
   taps [ 3 ] = _mm512_broadcast_i32x2(*( __m128i * ) &PCAffTab[0xef]);

    // Adjust K if we are decoding
    k += decoder ? 9 : 0 ;

    // Loop through each 64 byte codeword
    for ( curPos = 0 ; curPos < len ; curPos += 64 )
    {
        sPnt = data ;
        parity [ 0 ] = _mm512_setzero_si512();
        parity [ 1 ] = _mm512_setzero_si512();
        parity [ 2 ] = _mm512_setzero_si512();
        parity [ 3 ] = _mm512_setzero_si512();
        parity [ 4 ] = _mm512_setzero_si512();
        parity [ 5 ] = _mm512_setzero_si512();
        parity [ 6 ] = _mm512_setzero_si512();
        parity [ 7 ] = _mm512_setzero_si512();
        parity [ 8 ] = _mm512_setzero_si512();

        // Loop through all the 0..k symbols
        for ( curSym = 0 ; curSym < k ; curSym ++ )
        {
            // Load next 64 bytes of Original Data
            data_vec = _mm512_stream_load_si512( *sPnt + curPos ) ;
          __builtin_prefetch ( *sPnt + curPos + 64, 0, 3 ) ;
            sPnt ++ ;

            // Add incoming data to MSB of parity, then update parities using Parallel Multiplier
            data_vec = _mm512_xor_si512( data_vec, parity [ 0 ] ) ;

            // Compute the first half of the tap values for palindromic code generator
            temp [ 0 ] = _mm512_gf2p8affine_epi64_epi8 ( data_vec, taps [ 0 ], 0 ) ;
            temp [ 1 ] = _mm512_gf2p8affine_epi64_epi8 ( data_vec, taps [ 1 ], 0 ) ;
            temp [ 2 ] = _mm512_gf2p8affine_epi64_epi8 ( data_vec, taps [ 2 ], 0 ) ;
            temp [ 3 ] = _mm512_gf2p8affine_epi64_epi8 ( data_vec, taps [ 3 ], 0 ) ;

            // Now cycle the LFSR
            parity [ 0 ] = _mm512_xor_si512 ( parity [ 1 ], temp [ 0 ] ) ;
            parity [ 1 ] = _mm512_xor_si512 ( parity [ 2 ], temp [ 1 ] ) ;
            parity [ 2 ] = _mm512_xor_si512 ( parity [ 3 ], temp [ 2 ] ) ;
            parity [ 3 ] = _mm512_xor_si512 ( parity [ 4 ], temp [ 3 ] ) ;
            parity [ 4 ] = _mm512_xor_si512 ( parity [ 5 ], temp [ 3 ] ) ;
            parity [ 5 ] = _mm512_xor_si512 ( parity [ 6 ], temp [ 2 ] ) ;
            parity [ 6 ] = _mm512_xor_si512 ( parity [ 7 ], temp [ 1 ] ) ;
            parity [ 7 ] = _mm512_xor_si512 ( parity [ 8 ], temp [ 0 ] ) ;
            parity [ 8 ] = data_vec ;
        }

        if ( decoder == 0 )
        {
            // Store parity back to memory
            _mm512_stream_si512( (&data [ k+0 ][curPos]), parity [ 0 ] ) ;
            _mm512_stream_si512( (&data [ k+1 ][curPos]), parity [ 1 ] ) ;
            _mm512_stream_si512( (&data [ k+2 ][curPos]), parity [ 2 ] ) ;
            _mm512_stream_si512( (&data [ k+3 ][curPos]), parity [ 3 ] ) ;
            _mm512_stream_si512( (&data [ k+4 ][curPos]), parity [ 4 ] ) ;
            _mm512_stream_si512( (&data [ k+5 ][curPos]), parity [ 5 ] ) ;
            _mm512_stream_si512( (&data [ k+6 ][curPos]), parity [ 6 ] ) ;
            _mm512_stream_si512( (&data [ k+7 ][curPos]), parity [ 7 ] ) ;
            _mm512_stream_si512( (&data [ k+8 ][curPos]), parity [ 8 ] ) ;
        }
        else
        {
            // Verify Syndromes are zero
            data_vec = _mm512_or_si512 ( parity [ 0 ], parity [ 1 ] ) ;
            data_vec = _mm512_ternarylogic_epi32 ( data_vec, parity [ 2 ], parity [ 3 ], 0xFE ) ;
            data_vec = _mm512_ternarylogic_epi32 ( data_vec, parity [ 4 ], parity [ 5 ], 0xFE ) ;
            data_vec = _mm512_ternarylogic_epi32 ( data_vec, parity [ 6 ], parity [ 7 ], 0xFE ) ;
            data_vec = _mm512_ternarylogic_epi32 ( data_vec, parity [ 8 ], parity [ 8 ], 0xFE ) ;
            __mmask64 mask = _mm512_test_epi8_mask ( data_vec, data_vec ) ;

            // If syndromes are not zero then process error
            while ( !_ktestz_mask64_u8 ( mask, mask ) ) 
            {
                PCErrCnt++;
                unsigned char Sr[9], S[9], errLocs[9], errMags[9] ;
                unsigned char Lambda[9+1], B[9+1], T[9+1], Omega[9+1] ;
                unsigned char PTemp[64] ;
                uint64_t pos = _tzcnt_u64 ( mask ) ;

                _mm512_storeu_si512 ( PTemp, parity[0] ) ;
                Sr[0] = PTemp[pos];
                _mm512_storeu_si512 ( PTemp, parity[1] ) ;
                Sr[1] = PTemp[pos];
                _mm512_storeu_si512 ( PTemp, parity[2] ) ;
                Sr[2] = PTemp[pos];
                _mm512_storeu_si512 ( PTemp, parity[3] ) ;
                Sr[3] = PTemp[pos];
                _mm512_storeu_si512 ( PTemp, parity[4] ) ;
                Sr[4] = PTemp[pos];
                _mm512_storeu_si512 ( PTemp, parity[5] ) ;
                Sr[5] = PTemp[pos];
                _mm512_storeu_si512 ( PTemp, parity[6] ) ;
                Sr[6] = PTemp[pos];
                _mm512_storeu_si512 ( PTemp, parity[7] ) ;
                Sr[7] = PTemp[pos];
                _mm512_storeu_si512 ( PTemp, parity[8] ) ;
                Sr[8] = PTemp[pos];
                rs_decode_palindromic_GFNI(Sr, S, 9, k, errLocs, errMags,
                             (int)pos+curPos, Lambda, B, T, Omega, data) ;
                mask = _blsr_u64(mask);
            }
        }
    }
}

// Parallel LFSRD_SR Sequencer for P = 10 Codewords
void ParallelLFSRSequencer10_GFNI(int len, int k, unsigned char **data, int decoder)
{
    int curSym, curPos ;                           // Loop counters
    __m512i parity [ 10 ], taps [ 5 ] ;             // Parity registers
    __m512i data_vec, temp [ 5 ] ;
    unsigned char **sPnt ;                         // Data lookup pointers

    // Initialize the taps to the LFSR
   taps [ 0 ] = _mm512_broadcast_i32x2(*( __m128i * ) &PCAffTab[0x93]);
   taps [ 1 ] = _mm512_broadcast_i32x2(*( __m128i * ) &PCAffTab[0x69]);
   taps [ 2 ] = _mm512_broadcast_i32x2(*( __m128i * ) &PCAffTab[0xe6]);
   taps [ 3 ] = _mm512_broadcast_i32x2(*( __m128i * ) &PCAffTab[0x77]);
   taps [ 4 ] = _mm512_broadcast_i32x2(*( __m128i * ) &PCAffTab[0x9]);

    // Adjust K if we are decoding
    k += decoder ? 10 : 0 ;

    // Loop through each 64 byte codeword
    for ( curPos = 0 ; curPos < len ; curPos += 64 )
    {
        sPnt = data ;
        parity [ 0 ] = _mm512_setzero_si512();
        parity [ 1 ] = _mm512_setzero_si512();
        parity [ 2 ] = _mm512_setzero_si512();
        parity [ 3 ] = _mm512_setzero_si512();
        parity [ 4 ] = _mm512_setzero_si512();
        parity [ 5 ] = _mm512_setzero_si512();
        parity [ 6 ] = _mm512_setzero_si512();
        parity [ 7 ] = _mm512_setzero_si512();
        parity [ 8 ] = _mm512_setzero_si512();
        parity [ 9 ] = _mm512_setzero_si512();

        // Loop through all the 0..k symbols
        for ( curSym = 0 ; curSym < k ; curSym ++ )
        {
            // Load next 64 bytes of Original Data
            data_vec = _mm512_stream_load_si512( *sPnt + curPos ) ;
          __builtin_prefetch ( *sPnt + curPos + 64, 0, 3 ) ;
            sPnt ++ ;

            // Add incoming data to MSB of parity, then update parities using Parallel Multiplier
            data_vec = _mm512_xor_si512( data_vec, parity [ 0 ] ) ;

            // Compute the first half of the tap values for palindromic code generator
            temp [ 0 ] = _mm512_gf2p8affine_epi64_epi8 ( data_vec, taps [ 0 ], 0 ) ;
            temp [ 1 ] = _mm512_gf2p8affine_epi64_epi8 ( data_vec, taps [ 1 ], 0 ) ;
            temp [ 2 ] = _mm512_gf2p8affine_epi64_epi8 ( data_vec, taps [ 2 ], 0 ) ;
            temp [ 3 ] = _mm512_gf2p8affine_epi64_epi8 ( data_vec, taps [ 3 ], 0 ) ;
            temp [ 4 ] = _mm512_gf2p8affine_epi64_epi8 ( data_vec, taps [ 4 ], 0 ) ;

            // Now cycle the LFSR
            parity [ 0 ] = _mm512_xor_si512 ( parity [ 1 ], temp [ 0 ] ) ;
            parity [ 1 ] = _mm512_xor_si512 ( parity [ 2 ], temp [ 1 ] ) ;
            parity [ 2 ] = _mm512_xor_si512 ( parity [ 3 ], temp [ 2 ] ) ;
            parity [ 3 ] = _mm512_xor_si512 ( parity [ 4 ], temp [ 3 ] ) ;
            parity [ 4 ] = _mm512_xor_si512 ( parity [ 5 ], temp [ 4 ] ) ;
            parity [ 5 ] = _mm512_xor_si512 ( parity [ 6 ], temp [ 3 ] ) ;
            parity [ 6 ] = _mm512_xor_si512 ( parity [ 7 ], temp [ 2 ] ) ;
            parity [ 7 ] = _mm512_xor_si512 ( parity [ 8 ], temp [ 1 ] ) ;
            parity [ 8 ] = _mm512_xor_si512 ( parity [ 9 ], temp [ 0 ] ) ;
            parity [ 9 ] = data_vec ;
        }

        if ( decoder == 0 )
        {
            // Store parity back to memory
            _mm512_stream_si512( (&data [ k+0 ][curPos]), parity [ 0 ] ) ;
            _mm512_stream_si512( (&data [ k+1 ][curPos]), parity [ 1 ] ) ;
            _mm512_stream_si512( (&data [ k+2 ][curPos]), parity [ 2 ] ) ;
            _mm512_stream_si512( (&data [ k+3 ][curPos]), parity [ 3 ] ) ;
            _mm512_stream_si512( (&data [ k+4 ][curPos]), parity [ 4 ] ) ;
            _mm512_stream_si512( (&data [ k+5 ][curPos]), parity [ 5 ] ) ;
            _mm512_stream_si512( (&data [ k+6 ][curPos]), parity [ 6 ] ) ;
            _mm512_stream_si512( (&data [ k+7 ][curPos]), parity [ 7 ] ) ;
            _mm512_stream_si512( (&data [ k+8 ][curPos]), parity [ 8 ] ) ;
            _mm512_stream_si512( (&data [ k+9 ][curPos]), parity [ 9 ] ) ;
        }
        else
        {
            // Verify Syndromes are zero
            data_vec = _mm512_or_si512 ( parity [ 0 ], parity [ 1 ] ) ;
            data_vec = _mm512_ternarylogic_epi32 ( data_vec, parity [ 2 ], parity [ 3 ], 0xFE ) ;
            data_vec = _mm512_ternarylogic_epi32 ( data_vec, parity [ 4 ], parity [ 5 ], 0xFE ) ;
            data_vec = _mm512_ternarylogic_epi32 ( data_vec, parity [ 6 ], parity [ 7 ], 0xFE ) ;
            data_vec = _mm512_ternarylogic_epi32 ( data_vec, parity [ 8 ], parity [ 9 ], 0xFE ) ;
            __mmask64 mask = _mm512_test_epi8_mask ( data_vec, data_vec ) ;

            // If syndromes are not zero then process error
            while ( !_ktestz_mask64_u8 ( mask, mask ) ) 
            {
                PCErrCnt++;
                unsigned char Sr[10], S[10], errLocs[10], errMags[10] ;
                unsigned char Lambda[10+1], B[10+1], T[10+1], Omega[10+1] ;
                unsigned char PTemp[64] ;
                uint64_t pos = _tzcnt_u64 ( mask ) ;

                _mm512_storeu_si512 ( PTemp, parity[0] ) ;
                Sr[0] = PTemp[pos];
                _mm512_storeu_si512 ( PTemp, parity[1] ) ;
                Sr[1] = PTemp[pos];
                _mm512_storeu_si512 ( PTemp, parity[2] ) ;
                Sr[2] = PTemp[pos];
                _mm512_storeu_si512 ( PTemp, parity[3] ) ;
                Sr[3] = PTemp[pos];
                _mm512_storeu_si512 ( PTemp, parity[4] ) ;
                Sr[4] = PTemp[pos];
                _mm512_storeu_si512 ( PTemp, parity[5] ) ;
                Sr[5] = PTemp[pos];
                _mm512_storeu_si512 ( PTemp, parity[6] ) ;
                Sr[6] = PTemp[pos];
                _mm512_storeu_si512 ( PTemp, parity[7] ) ;
                Sr[7] = PTemp[pos];
                _mm512_storeu_si512 ( PTemp, parity[8] ) ;
                Sr[8] = PTemp[pos];
                _mm512_storeu_si512 ( PTemp, parity[9] ) ;
                Sr[9] = PTemp[pos];
                rs_decode_palindromic_GFNI(Sr, S, 10, k, errLocs, errMags,
                             (int)pos+curPos, Lambda, B, T, Omega, data) ;
                mask = _blsr_u64(mask);
            }
        }
    }
}

// Parallel LFSRD_SR Sequencer for P = 11 Codewords
void ParallelLFSRSequencer11_GFNI(int len, int k, unsigned char **data, int decoder)
{
    int curSym, curPos ;                           // Loop counters
    __m512i parity [ 11 ], taps [ 5 ] ;             // Parity registers
    __m512i data_vec, temp [ 5 ] ;
    unsigned char **sPnt ;                         // Data lookup pointers

    // Initialize the taps to the LFSR
   taps [ 0 ] = _mm512_broadcast_i32x2(*( __m128i * ) &PCAffTab[0xef]);
   taps [ 1 ] = _mm512_broadcast_i32x2(*( __m128i * ) &PCAffTab[0x62]);
   taps [ 2 ] = _mm512_broadcast_i32x2(*( __m128i * ) &PCAffTab[0x1e]);
   taps [ 3 ] = _mm512_broadcast_i32x2(*( __m128i * ) &PCAffTab[0xf1]);
   taps [ 4 ] = _mm512_broadcast_i32x2(*( __m128i * ) &PCAffTab[0x29]);

    // Adjust K if we are decoding
    k += decoder ? 11 : 0 ;

    // Loop through each 64 byte codeword
    for ( curPos = 0 ; curPos < len ; curPos += 64 )
    {
        sPnt = data ;
        parity [ 0 ] = _mm512_setzero_si512();
        parity [ 1 ] = _mm512_setzero_si512();
        parity [ 2 ] = _mm512_setzero_si512();
        parity [ 3 ] = _mm512_setzero_si512();
        parity [ 4 ] = _mm512_setzero_si512();
        parity [ 5 ] = _mm512_setzero_si512();
        parity [ 6 ] = _mm512_setzero_si512();
        parity [ 7 ] = _mm512_setzero_si512();
        parity [ 8 ] = _mm512_setzero_si512();
        parity [ 9 ] = _mm512_setzero_si512();
        parity [ 10 ] = _mm512_setzero_si512();

        // Loop through all the 0..k symbols
        for ( curSym = 0 ; curSym < k ; curSym ++ )
        {
            // Load next 64 bytes of Original Data
            data_vec = _mm512_stream_load_si512( *sPnt + curPos ) ;
          __builtin_prefetch ( *sPnt + curPos + 64, 0, 3 ) ;
            sPnt ++ ;

            // Add incoming data to MSB of parity, then update parities using Parallel Multiplier
            data_vec = _mm512_xor_si512( data_vec, parity [ 0 ] ) ;

            // Compute the first half of the tap values for palindromic code generator
            temp [ 0 ] = _mm512_gf2p8affine_epi64_epi8 ( data_vec, taps [ 0 ], 0 ) ;
            temp [ 1 ] = _mm512_gf2p8affine_epi64_epi8 ( data_vec, taps [ 1 ], 0 ) ;
            temp [ 2 ] = _mm512_gf2p8affine_epi64_epi8 ( data_vec, taps [ 2 ], 0 ) ;
            temp [ 3 ] = _mm512_gf2p8affine_epi64_epi8 ( data_vec, taps [ 3 ], 0 ) ;
            temp [ 4 ] = _mm512_gf2p8affine_epi64_epi8 ( data_vec, taps [ 4 ], 0 ) ;

            // Now cycle the LFSR
            parity [ 0 ] = _mm512_xor_si512 ( parity [ 1 ], temp [ 0 ] ) ;
            parity [ 1 ] = _mm512_xor_si512 ( parity [ 2 ], temp [ 1 ] ) ;
            parity [ 2 ] = _mm512_xor_si512 ( parity [ 3 ], temp [ 2 ] ) ;
            parity [ 3 ] = _mm512_xor_si512 ( parity [ 4 ], temp [ 3 ] ) ;
            parity [ 4 ] = _mm512_xor_si512 ( parity [ 5 ], temp [ 4 ] ) ;
            parity [ 5 ] = _mm512_xor_si512 ( parity [ 6 ], temp [ 4 ] ) ;
            parity [ 6 ] = _mm512_xor_si512 ( parity [ 7 ], temp [ 3 ] ) ;
            parity [ 7 ] = _mm512_xor_si512 ( parity [ 8 ], temp [ 2 ] ) ;
            parity [ 8 ] = _mm512_xor_si512 ( parity [ 9 ], temp [ 1 ] ) ;
            parity [ 9 ] = _mm512_xor_si512 ( parity [ 10 ], temp [ 0 ] ) ;
            parity [ 10 ] = data_vec ;
        }

        if ( decoder == 0 )
        {
            // Store parity back to memory
            _mm512_stream_si512( (&data [ k+0 ][curPos]), parity [ 0 ] ) ;
            _mm512_stream_si512( (&data [ k+1 ][curPos]), parity [ 1 ] ) ;
            _mm512_stream_si512( (&data [ k+2 ][curPos]), parity [ 2 ] ) ;
            _mm512_stream_si512( (&data [ k+3 ][curPos]), parity [ 3 ] ) ;
            _mm512_stream_si512( (&data [ k+4 ][curPos]), parity [ 4 ] ) ;
            _mm512_stream_si512( (&data [ k+5 ][curPos]), parity [ 5 ] ) ;
            _mm512_stream_si512( (&data [ k+6 ][curPos]), parity [ 6 ] ) ;
            _mm512_stream_si512( (&data [ k+7 ][curPos]), parity [ 7 ] ) ;
            _mm512_stream_si512( (&data [ k+8 ][curPos]), parity [ 8 ] ) ;
            _mm512_stream_si512( (&data [ k+9 ][curPos]), parity [ 9 ] ) ;
            _mm512_stream_si512( (&data [ k+10 ][curPos]), parity [ 10 ] ) ;
        }
        else
        {
            // Verify Syndromes are zero
            data_vec = _mm512_or_si512 ( parity [ 0 ], parity [ 1 ] ) ;
            data_vec = _mm512_ternarylogic_epi32 ( data_vec, parity [ 2 ], parity [ 3 ], 0xFE ) ;
            data_vec = _mm512_ternarylogic_epi32 ( data_vec, parity [ 4 ], parity [ 5 ], 0xFE ) ;
            data_vec = _mm512_ternarylogic_epi32 ( data_vec, parity [ 6 ], parity [ 7 ], 0xFE ) ;
            data_vec = _mm512_ternarylogic_epi32 ( data_vec, parity [ 8 ], parity [ 9 ], 0xFE ) ;
            data_vec = _mm512_ternarylogic_epi32 ( data_vec, parity [ 10 ], parity [ 10 ], 0xFE ) ;
            __mmask64 mask = _mm512_test_epi8_mask ( data_vec, data_vec ) ;

            // If syndromes are not zero then process error
            while ( !_ktestz_mask64_u8 ( mask, mask ) ) 
            {
                PCErrCnt++;
                unsigned char Sr[11], S[11], errLocs[11], errMags[11] ;
                unsigned char Lambda[11+1], B[11+1], T[11+1], Omega[11+1] ;
                unsigned char PTemp[64] ;
                uint64_t pos = _tzcnt_u64 ( mask ) ;

                _mm512_storeu_si512 ( PTemp, parity[0] ) ;
                Sr[0] = PTemp[pos];
                _mm512_storeu_si512 ( PTemp, parity[1] ) ;
                Sr[1] = PTemp[pos];
                _mm512_storeu_si512 ( PTemp, parity[2] ) ;
                Sr[2] = PTemp[pos];
                _mm512_storeu_si512 ( PTemp, parity[3] ) ;
                Sr[3] = PTemp[pos];
                _mm512_storeu_si512 ( PTemp, parity[4] ) ;
                Sr[4] = PTemp[pos];
                _mm512_storeu_si512 ( PTemp, parity[5] ) ;
                Sr[5] = PTemp[pos];
                _mm512_storeu_si512 ( PTemp, parity[6] ) ;
                Sr[6] = PTemp[pos];
                _mm512_storeu_si512 ( PTemp, parity[7] ) ;
                Sr[7] = PTemp[pos];
                _mm512_storeu_si512 ( PTemp, parity[8] ) ;
                Sr[8] = PTemp[pos];
                _mm512_storeu_si512 ( PTemp, parity[9] ) ;
                Sr[9] = PTemp[pos];
                _mm512_storeu_si512 ( PTemp, parity[10] ) ;
                Sr[10] = PTemp[pos];
                rs_decode_palindromic_GFNI(Sr, S, 11, k, errLocs, errMags,
                             (int)pos+curPos, Lambda, B, T, Omega, data) ;
                mask = _blsr_u64(mask);
            }
        }
    }
}

// Parallel LFSRD_SR Sequencer for P = 12 Codewords
void ParallelLFSRSequencer12_GFNI(int len, int k, unsigned char **data, int decoder)
{
    int curSym, curPos ;                           // Loop counters
    __m512i parity [ 12 ], taps [ 6 ] ;             // Parity registers
    __m512i data_vec, temp [ 6 ] ;
    unsigned char **sPnt ;                         // Data lookup pointers

    // Initialize the taps to the LFSR
   taps [ 0 ] = _mm512_broadcast_i32x2(*( __m128i * ) &PCAffTab[0x12]);
   taps [ 1 ] = _mm512_broadcast_i32x2(*( __m128i * ) &PCAffTab[0x9d]);
   taps [ 2 ] = _mm512_broadcast_i32x2(*( __m128i * ) &PCAffTab[0xa2]);
   taps [ 3 ] = _mm512_broadcast_i32x2(*( __m128i * ) &PCAffTab[0x86]);
   taps [ 4 ] = _mm512_broadcast_i32x2(*( __m128i * ) &PCAffTab[0x9d]);
   taps [ 5 ] = _mm512_broadcast_i32x2(*( __m128i * ) &PCAffTab[0xfd]);

    // Adjust K if we are decoding
    k += decoder ? 12 : 0 ;

    // Loop through each 64 byte codeword
    for ( curPos = 0 ; curPos < len ; curPos += 64 )
    {
        sPnt = data ;
        parity [ 0 ] = _mm512_setzero_si512();
        parity [ 1 ] = _mm512_setzero_si512();
        parity [ 2 ] = _mm512_setzero_si512();
        parity [ 3 ] = _mm512_setzero_si512();
        parity [ 4 ] = _mm512_setzero_si512();
        parity [ 5 ] = _mm512_setzero_si512();
        parity [ 6 ] = _mm512_setzero_si512();
        parity [ 7 ] = _mm512_setzero_si512();
        parity [ 8 ] = _mm512_setzero_si512();
        parity [ 9 ] = _mm512_setzero_si512();
        parity [ 10 ] = _mm512_setzero_si512();
        parity [ 11 ] = _mm512_setzero_si512();

        // Loop through all the 0..k symbols
        for ( curSym = 0 ; curSym < k ; curSym ++ )
        {
            // Load next 64 bytes of Original Data
            data_vec = _mm512_stream_load_si512( *sPnt + curPos ) ;
          __builtin_prefetch ( *sPnt + curPos + 64, 0, 3 ) ;
            sPnt ++ ;

            // Add incoming data to MSB of parity, then update parities using Parallel Multiplier
            data_vec = _mm512_xor_si512( data_vec, parity [ 0 ] ) ;

            // Compute the first half of the tap values for palindromic code generator
            temp [ 0 ] = _mm512_gf2p8affine_epi64_epi8 ( data_vec, taps [ 0 ], 0 ) ;
            temp [ 1 ] = _mm512_gf2p8affine_epi64_epi8 ( data_vec, taps [ 1 ], 0 ) ;
            temp [ 2 ] = _mm512_gf2p8affine_epi64_epi8 ( data_vec, taps [ 2 ], 0 ) ;
            temp [ 3 ] = _mm512_gf2p8affine_epi64_epi8 ( data_vec, taps [ 3 ], 0 ) ;
            temp [ 4 ] = _mm512_gf2p8affine_epi64_epi8 ( data_vec, taps [ 4 ], 0 ) ;
            temp [ 5 ] = _mm512_gf2p8affine_epi64_epi8 ( data_vec, taps [ 5 ], 0 ) ;

            // Now cycle the LFSR
            parity [ 0 ] = _mm512_xor_si512 ( parity [ 1 ], temp [ 0 ] ) ;
            parity [ 1 ] = _mm512_xor_si512 ( parity [ 2 ], temp [ 1 ] ) ;
            parity [ 2 ] = _mm512_xor_si512 ( parity [ 3 ], temp [ 2 ] ) ;
            parity [ 3 ] = _mm512_xor_si512 ( parity [ 4 ], temp [ 3 ] ) ;
            parity [ 4 ] = _mm512_xor_si512 ( parity [ 5 ], temp [ 4 ] ) ;
            parity [ 5 ] = _mm512_xor_si512 ( parity [ 6 ], temp [ 5 ] ) ;
            parity [ 6 ] = _mm512_xor_si512 ( parity [ 7 ], temp [ 4 ] ) ;
            parity [ 7 ] = _mm512_xor_si512 ( parity [ 8 ], temp [ 3 ] ) ;
            parity [ 8 ] = _mm512_xor_si512 ( parity [ 9 ], temp [ 2 ] ) ;
            parity [ 9 ] = _mm512_xor_si512 ( parity [ 10 ], temp [ 1 ] ) ;
            parity [ 10 ] = _mm512_xor_si512 ( parity [ 11 ], temp [ 0 ] ) ;
            parity [ 11 ] = data_vec ;
        }

        if ( decoder == 0 )
        {
            // Store parity back to memory
            _mm512_stream_si512( (&data [ k+0 ][curPos]), parity [ 0 ] ) ;
            _mm512_stream_si512( (&data [ k+1 ][curPos]), parity [ 1 ] ) ;
            _mm512_stream_si512( (&data [ k+2 ][curPos]), parity [ 2 ] ) ;
            _mm512_stream_si512( (&data [ k+3 ][curPos]), parity [ 3 ] ) ;
            _mm512_stream_si512( (&data [ k+4 ][curPos]), parity [ 4 ] ) ;
            _mm512_stream_si512( (&data [ k+5 ][curPos]), parity [ 5 ] ) ;
            _mm512_stream_si512( (&data [ k+6 ][curPos]), parity [ 6 ] ) ;
            _mm512_stream_si512( (&data [ k+7 ][curPos]), parity [ 7 ] ) ;
            _mm512_stream_si512( (&data [ k+8 ][curPos]), parity [ 8 ] ) ;
            _mm512_stream_si512( (&data [ k+9 ][curPos]), parity [ 9 ] ) ;
            _mm512_stream_si512( (&data [ k+10 ][curPos]), parity [ 10 ] ) ;
            _mm512_stream_si512( (&data [ k+11 ][curPos]), parity [ 11 ] ) ;
        }
        else
        {
            // Verify Syndromes are zero
            data_vec = _mm512_or_si512 ( parity [ 0 ], parity [ 1 ] ) ;
            data_vec = _mm512_ternarylogic_epi32 ( data_vec, parity [ 2 ], parity [ 3 ], 0xFE ) ;
            data_vec = _mm512_ternarylogic_epi32 ( data_vec, parity [ 4 ], parity [ 5 ], 0xFE ) ;
            data_vec = _mm512_ternarylogic_epi32 ( data_vec, parity [ 6 ], parity [ 7 ], 0xFE ) ;
            data_vec = _mm512_ternarylogic_epi32 ( data_vec, parity [ 8 ], parity [ 9 ], 0xFE ) ;
            data_vec = _mm512_ternarylogic_epi32 ( data_vec, parity [ 10 ], parity [ 11 ], 0xFE ) ;
            __mmask64 mask = _mm512_test_epi8_mask ( data_vec, data_vec ) ;

            // If syndromes are not zero then process error
            while ( !_ktestz_mask64_u8 ( mask, mask ) ) 
            {
                PCErrCnt++;
                unsigned char Sr[12], S[12], errLocs[12], errMags[12] ;
                unsigned char Lambda[12+1], B[12+1], T[12+1], Omega[12+1] ;
                unsigned char PTemp[64] ;
                uint64_t pos = _tzcnt_u64 ( mask ) ;

                _mm512_storeu_si512 ( PTemp, parity[0] ) ;
                Sr[0] = PTemp[pos];
                _mm512_storeu_si512 ( PTemp, parity[1] ) ;
                Sr[1] = PTemp[pos];
                _mm512_storeu_si512 ( PTemp, parity[2] ) ;
                Sr[2] = PTemp[pos];
                _mm512_storeu_si512 ( PTemp, parity[3] ) ;
                Sr[3] = PTemp[pos];
                _mm512_storeu_si512 ( PTemp, parity[4] ) ;
                Sr[4] = PTemp[pos];
                _mm512_storeu_si512 ( PTemp, parity[5] ) ;
                Sr[5] = PTemp[pos];
                _mm512_storeu_si512 ( PTemp, parity[6] ) ;
                Sr[6] = PTemp[pos];
                _mm512_storeu_si512 ( PTemp, parity[7] ) ;
                Sr[7] = PTemp[pos];
                _mm512_storeu_si512 ( PTemp, parity[8] ) ;
                Sr[8] = PTemp[pos];
                _mm512_storeu_si512 ( PTemp, parity[9] ) ;
                Sr[9] = PTemp[pos];
                _mm512_storeu_si512 ( PTemp, parity[10] ) ;
                Sr[10] = PTemp[pos];
                _mm512_storeu_si512 ( PTemp, parity[11] ) ;
                Sr[11] = PTemp[pos];
                rs_decode_palindromic_GFNI(Sr, S, 12, k, errLocs, errMags,
                             (int)pos+curPos, Lambda, B, T, Omega, data) ;
                mask = _blsr_u64(mask);
            }
        }
    }
}

// Parallel LFSRD_SR Sequencer for P = 13 Codewords
void ParallelLFSRSequencer13_GFNI(int len, int k, unsigned char **data, int decoder)
{
    int curSym, curPos ;                           // Loop counters
    __m512i parity [ 13 ], taps [ 6 ] ;             // Parity registers
    __m512i data_vec, temp [ 6 ] ;
    unsigned char **sPnt ;                         // Data lookup pointers

    // Initialize the taps to the LFSR
   taps [ 0 ] = _mm512_broadcast_i32x2(*( __m128i * ) &PCAffTab[0x99]);
   taps [ 1 ] = _mm512_broadcast_i32x2(*( __m128i * ) &PCAffTab[0x84]);
   taps [ 2 ] = _mm512_broadcast_i32x2(*( __m128i * ) &PCAffTab[0xb7]);
   taps [ 3 ] = _mm512_broadcast_i32x2(*( __m128i * ) &PCAffTab[0x1e]);
   taps [ 4 ] = _mm512_broadcast_i32x2(*( __m128i * ) &PCAffTab[0x5d]);
   taps [ 5 ] = _mm512_broadcast_i32x2(*( __m128i * ) &PCAffTab[0x5f]);

    // Adjust K if we are decoding
    k += decoder ? 13 : 0 ;

    // Loop through each 64 byte codeword
    for ( curPos = 0 ; curPos < len ; curPos += 64 )
    {
        sPnt = data ;
        parity [ 0 ] = _mm512_setzero_si512();
        parity [ 1 ] = _mm512_setzero_si512();
        parity [ 2 ] = _mm512_setzero_si512();
        parity [ 3 ] = _mm512_setzero_si512();
        parity [ 4 ] = _mm512_setzero_si512();
        parity [ 5 ] = _mm512_setzero_si512();
        parity [ 6 ] = _mm512_setzero_si512();
        parity [ 7 ] = _mm512_setzero_si512();
        parity [ 8 ] = _mm512_setzero_si512();
        parity [ 9 ] = _mm512_setzero_si512();
        parity [ 10 ] = _mm512_setzero_si512();
        parity [ 11 ] = _mm512_setzero_si512();
        parity [ 12 ] = _mm512_setzero_si512();

        // Loop through all the 0..k symbols
        for ( curSym = 0 ; curSym < k ; curSym ++ )
        {
            // Load next 64 bytes of Original Data
            data_vec = _mm512_stream_load_si512( *sPnt + curPos ) ;
          __builtin_prefetch ( *sPnt + curPos + 64, 0, 3 ) ;
            sPnt ++ ;

            // Add incoming data to MSB of parity, then update parities using Parallel Multiplier
            data_vec = _mm512_xor_si512( data_vec, parity [ 0 ] ) ;

            // Compute the first half of the tap values for palindromic code generator
            temp [ 0 ] = _mm512_gf2p8affine_epi64_epi8 ( data_vec, taps [ 0 ], 0 ) ;
            temp [ 1 ] = _mm512_gf2p8affine_epi64_epi8 ( data_vec, taps [ 1 ], 0 ) ;
            temp [ 2 ] = _mm512_gf2p8affine_epi64_epi8 ( data_vec, taps [ 2 ], 0 ) ;
            temp [ 3 ] = _mm512_gf2p8affine_epi64_epi8 ( data_vec, taps [ 3 ], 0 ) ;
            temp [ 4 ] = _mm512_gf2p8affine_epi64_epi8 ( data_vec, taps [ 4 ], 0 ) ;
            temp [ 5 ] = _mm512_gf2p8affine_epi64_epi8 ( data_vec, taps [ 5 ], 0 ) ;

            // Now cycle the LFSR
            parity [ 0 ] = _mm512_xor_si512 ( parity [ 1 ], temp [ 0 ] ) ;
            parity [ 1 ] = _mm512_xor_si512 ( parity [ 2 ], temp [ 1 ] ) ;
            parity [ 2 ] = _mm512_xor_si512 ( parity [ 3 ], temp [ 2 ] ) ;
            parity [ 3 ] = _mm512_xor_si512 ( parity [ 4 ], temp [ 3 ] ) ;
            parity [ 4 ] = _mm512_xor_si512 ( parity [ 5 ], temp [ 4 ] ) ;
            parity [ 5 ] = _mm512_xor_si512 ( parity [ 6 ], temp [ 5 ] ) ;
            parity [ 6 ] = _mm512_xor_si512 ( parity [ 7 ], temp [ 5 ] ) ;
            parity [ 7 ] = _mm512_xor_si512 ( parity [ 8 ], temp [ 4 ] ) ;
            parity [ 8 ] = _mm512_xor_si512 ( parity [ 9 ], temp [ 3 ] ) ;
            parity [ 9 ] = _mm512_xor_si512 ( parity [ 10 ], temp [ 2 ] ) ;
            parity [ 10 ] = _mm512_xor_si512 ( parity [ 11 ], temp [ 1 ] ) ;
            parity [ 11 ] = _mm512_xor_si512 ( parity [ 12 ], temp [ 0 ] ) ;
            parity [ 12 ] = data_vec ;
        }

        if ( decoder == 0 )
        {
            // Store parity back to memory
            _mm512_stream_si512( (&data [ k+0 ][curPos]), parity [ 0 ] ) ;
            _mm512_stream_si512( (&data [ k+1 ][curPos]), parity [ 1 ] ) ;
            _mm512_stream_si512( (&data [ k+2 ][curPos]), parity [ 2 ] ) ;
            _mm512_stream_si512( (&data [ k+3 ][curPos]), parity [ 3 ] ) ;
            _mm512_stream_si512( (&data [ k+4 ][curPos]), parity [ 4 ] ) ;
            _mm512_stream_si512( (&data [ k+5 ][curPos]), parity [ 5 ] ) ;
            _mm512_stream_si512( (&data [ k+6 ][curPos]), parity [ 6 ] ) ;
            _mm512_stream_si512( (&data [ k+7 ][curPos]), parity [ 7 ] ) ;
            _mm512_stream_si512( (&data [ k+8 ][curPos]), parity [ 8 ] ) ;
            _mm512_stream_si512( (&data [ k+9 ][curPos]), parity [ 9 ] ) ;
            _mm512_stream_si512( (&data [ k+10 ][curPos]), parity [ 10 ] ) ;
            _mm512_stream_si512( (&data [ k+11 ][curPos]), parity [ 11 ] ) ;
            _mm512_stream_si512( (&data [ k+12 ][curPos]), parity [ 12 ] ) ;
        }
        else
        {
            // Verify Syndromes are zero
            data_vec = _mm512_or_si512 ( parity [ 0 ], parity [ 1 ] ) ;
            data_vec = _mm512_ternarylogic_epi32 ( data_vec, parity [ 2 ], parity [ 3 ], 0xFE ) ;
            data_vec = _mm512_ternarylogic_epi32 ( data_vec, parity [ 4 ], parity [ 5 ], 0xFE ) ;
            data_vec = _mm512_ternarylogic_epi32 ( data_vec, parity [ 6 ], parity [ 7 ], 0xFE ) ;
            data_vec = _mm512_ternarylogic_epi32 ( data_vec, parity [ 8 ], parity [ 9 ], 0xFE ) ;
            data_vec = _mm512_ternarylogic_epi32 ( data_vec, parity [ 10 ], parity [ 11 ], 0xFE ) ;
            data_vec = _mm512_ternarylogic_epi32 ( data_vec, parity [ 12 ], parity [ 12 ], 0xFE ) ;
            __mmask64 mask = _mm512_test_epi8_mask ( data_vec, data_vec ) ;

            // If syndromes are not zero then process error
            while ( !_ktestz_mask64_u8 ( mask, mask ) ) 
            {
                PCErrCnt++;
                unsigned char Sr[13], S[13], errLocs[13], errMags[13] ;
                unsigned char Lambda[13+1], B[13+1], T[13+1], Omega[13+1] ;
                unsigned char PTemp[64] ;
                uint64_t pos = _tzcnt_u64 ( mask ) ;

                _mm512_storeu_si512 ( PTemp, parity[0] ) ;
                Sr[0] = PTemp[pos];
                _mm512_storeu_si512 ( PTemp, parity[1] ) ;
                Sr[1] = PTemp[pos];
                _mm512_storeu_si512 ( PTemp, parity[2] ) ;
                Sr[2] = PTemp[pos];
                _mm512_storeu_si512 ( PTemp, parity[3] ) ;
                Sr[3] = PTemp[pos];
                _mm512_storeu_si512 ( PTemp, parity[4] ) ;
                Sr[4] = PTemp[pos];
                _mm512_storeu_si512 ( PTemp, parity[5] ) ;
                Sr[5] = PTemp[pos];
                _mm512_storeu_si512 ( PTemp, parity[6] ) ;
                Sr[6] = PTemp[pos];
                _mm512_storeu_si512 ( PTemp, parity[7] ) ;
                Sr[7] = PTemp[pos];
                _mm512_storeu_si512 ( PTemp, parity[8] ) ;
                Sr[8] = PTemp[pos];
                _mm512_storeu_si512 ( PTemp, parity[9] ) ;
                Sr[9] = PTemp[pos];
                _mm512_storeu_si512 ( PTemp, parity[10] ) ;
                Sr[10] = PTemp[pos];
                _mm512_storeu_si512 ( PTemp, parity[11] ) ;
                Sr[11] = PTemp[pos];
                _mm512_storeu_si512 ( PTemp, parity[12] ) ;
                Sr[12] = PTemp[pos];
                rs_decode_palindromic_GFNI(Sr, S, 13, k, errLocs, errMags,
                             (int)pos+curPos, Lambda, B, T, Omega, data) ;
                mask = _blsr_u64(mask);
            }
        }
    }
}

// Parallel LFSRD_SR Sequencer for P = 14 Codewords
void ParallelLFSRSequencer14_GFNI(int len, int k, unsigned char **data, int decoder)
{
    int curSym, curPos ;                           // Loop counters
    __m512i parity [ 14 ], taps [ 7 ] ;             // Parity registers
    __m512i data_vec, temp [ 7 ] ;
    unsigned char **sPnt ;                         // Data lookup pointers

    // Initialize the taps to the LFSR
   taps [ 0 ] = _mm512_broadcast_i32x2(*( __m128i * ) &PCAffTab[0xbe]);
   taps [ 1 ] = _mm512_broadcast_i32x2(*( __m128i * ) &PCAffTab[0xcb]);
   taps [ 2 ] = _mm512_broadcast_i32x2(*( __m128i * ) &PCAffTab[0x1d]);
   taps [ 3 ] = _mm512_broadcast_i32x2(*( __m128i * ) &PCAffTab[0xea]);
   taps [ 4 ] = _mm512_broadcast_i32x2(*( __m128i * ) &PCAffTab[0x60]);
   taps [ 5 ] = _mm512_broadcast_i32x2(*( __m128i * ) &PCAffTab[0xd6]);
   taps [ 6 ] = _mm512_broadcast_i32x2(*( __m128i * ) &PCAffTab[0xc1]);

    // Adjust K if we are decoding
    k += decoder ? 14 : 0 ;

    // Loop through each 64 byte codeword
    for ( curPos = 0 ; curPos < len ; curPos += 64 )
    {
        sPnt = data ;
        parity [ 0 ] = _mm512_setzero_si512();
        parity [ 1 ] = _mm512_setzero_si512();
        parity [ 2 ] = _mm512_setzero_si512();
        parity [ 3 ] = _mm512_setzero_si512();
        parity [ 4 ] = _mm512_setzero_si512();
        parity [ 5 ] = _mm512_setzero_si512();
        parity [ 6 ] = _mm512_setzero_si512();
        parity [ 7 ] = _mm512_setzero_si512();
        parity [ 8 ] = _mm512_setzero_si512();
        parity [ 9 ] = _mm512_setzero_si512();
        parity [ 10 ] = _mm512_setzero_si512();
        parity [ 11 ] = _mm512_setzero_si512();
        parity [ 12 ] = _mm512_setzero_si512();
        parity [ 13 ] = _mm512_setzero_si512();

        // Loop through all the 0..k symbols
        for ( curSym = 0 ; curSym < k ; curSym ++ )
        {
            // Load next 64 bytes of Original Data
            data_vec = _mm512_stream_load_si512( *sPnt + curPos ) ;
          __builtin_prefetch ( *sPnt + curPos + 64, 0, 3 ) ;
            sPnt ++ ;

            // Add incoming data to MSB of parity, then update parities using Parallel Multiplier
            data_vec = _mm512_xor_si512( data_vec, parity [ 0 ] ) ;

            // Compute the first half of the tap values for palindromic code generator
            temp [ 0 ] = _mm512_gf2p8affine_epi64_epi8 ( data_vec, taps [ 0 ], 0 ) ;
            temp [ 1 ] = _mm512_gf2p8affine_epi64_epi8 ( data_vec, taps [ 1 ], 0 ) ;
            temp [ 2 ] = _mm512_gf2p8affine_epi64_epi8 ( data_vec, taps [ 2 ], 0 ) ;
            temp [ 3 ] = _mm512_gf2p8affine_epi64_epi8 ( data_vec, taps [ 3 ], 0 ) ;
            temp [ 4 ] = _mm512_gf2p8affine_epi64_epi8 ( data_vec, taps [ 4 ], 0 ) ;
            temp [ 5 ] = _mm512_gf2p8affine_epi64_epi8 ( data_vec, taps [ 5 ], 0 ) ;
            temp [ 6 ] = _mm512_gf2p8affine_epi64_epi8 ( data_vec, taps [ 6 ], 0 ) ;

            // Now cycle the LFSR
            parity [ 0 ] = _mm512_xor_si512 ( parity [ 1 ], temp [ 0 ] ) ;
            parity [ 1 ] = _mm512_xor_si512 ( parity [ 2 ], temp [ 1 ] ) ;
            parity [ 2 ] = _mm512_xor_si512 ( parity [ 3 ], temp [ 2 ] ) ;
            parity [ 3 ] = _mm512_xor_si512 ( parity [ 4 ], temp [ 3 ] ) ;
            parity [ 4 ] = _mm512_xor_si512 ( parity [ 5 ], temp [ 4 ] ) ;
            parity [ 5 ] = _mm512_xor_si512 ( parity [ 6 ], temp [ 5 ] ) ;
            parity [ 6 ] = _mm512_xor_si512 ( parity [ 7 ], temp [ 6 ] ) ;
            parity [ 7 ] = _mm512_xor_si512 ( parity [ 8 ], temp [ 5 ] ) ;
            parity [ 8 ] = _mm512_xor_si512 ( parity [ 9 ], temp [ 4 ] ) ;
            parity [ 9 ] = _mm512_xor_si512 ( parity [ 10 ], temp [ 3 ] ) ;
            parity [ 10 ] = _mm512_xor_si512 ( parity [ 11 ], temp [ 2 ] ) ;
            parity [ 11 ] = _mm512_xor_si512 ( parity [ 12 ], temp [ 1 ] ) ;
            parity [ 12 ] = _mm512_xor_si512 ( parity [ 13 ], temp [ 0 ] ) ;
            parity [ 13 ] = data_vec ;
        }

        if ( decoder == 0 )
        {
            // Store parity back to memory
            _mm512_stream_si512( (&data [ k+0 ][curPos]), parity [ 0 ] ) ;
            _mm512_stream_si512( (&data [ k+1 ][curPos]), parity [ 1 ] ) ;
            _mm512_stream_si512( (&data [ k+2 ][curPos]), parity [ 2 ] ) ;
            _mm512_stream_si512( (&data [ k+3 ][curPos]), parity [ 3 ] ) ;
            _mm512_stream_si512( (&data [ k+4 ][curPos]), parity [ 4 ] ) ;
            _mm512_stream_si512( (&data [ k+5 ][curPos]), parity [ 5 ] ) ;
            _mm512_stream_si512( (&data [ k+6 ][curPos]), parity [ 6 ] ) ;
            _mm512_stream_si512( (&data [ k+7 ][curPos]), parity [ 7 ] ) ;
            _mm512_stream_si512( (&data [ k+8 ][curPos]), parity [ 8 ] ) ;
            _mm512_stream_si512( (&data [ k+9 ][curPos]), parity [ 9 ] ) ;
            _mm512_stream_si512( (&data [ k+10 ][curPos]), parity [ 10 ] ) ;
            _mm512_stream_si512( (&data [ k+11 ][curPos]), parity [ 11 ] ) ;
            _mm512_stream_si512( (&data [ k+12 ][curPos]), parity [ 12 ] ) ;
            _mm512_stream_si512( (&data [ k+13 ][curPos]), parity [ 13 ] ) ;
        }
        else
        {
            // Verify Syndromes are zero
            data_vec = _mm512_or_si512 ( parity [ 0 ], parity [ 1 ] ) ;
            data_vec = _mm512_ternarylogic_epi32 ( data_vec, parity [ 2 ], parity [ 3 ], 0xFE ) ;
            data_vec = _mm512_ternarylogic_epi32 ( data_vec, parity [ 4 ], parity [ 5 ], 0xFE ) ;
            data_vec = _mm512_ternarylogic_epi32 ( data_vec, parity [ 6 ], parity [ 7 ], 0xFE ) ;
            data_vec = _mm512_ternarylogic_epi32 ( data_vec, parity [ 8 ], parity [ 9 ], 0xFE ) ;
            data_vec = _mm512_ternarylogic_epi32 ( data_vec, parity [ 10 ], parity [ 11 ], 0xFE ) ;
            data_vec = _mm512_ternarylogic_epi32 ( data_vec, parity [ 12 ], parity [ 13 ], 0xFE ) ;
            __mmask64 mask = _mm512_test_epi8_mask ( data_vec, data_vec ) ;

            // If syndromes are not zero then process error
            while ( !_ktestz_mask64_u8 ( mask, mask ) ) 
            {
                PCErrCnt++;
                unsigned char Sr[14], S[14], errLocs[14], errMags[14] ;
                unsigned char Lambda[14+1], B[14+1], T[14+1], Omega[14+1] ;
                unsigned char PTemp[64] ;
                uint64_t pos = _tzcnt_u64 ( mask ) ;

                _mm512_storeu_si512 ( PTemp, parity[0] ) ;
                Sr[0] = PTemp[pos];
                _mm512_storeu_si512 ( PTemp, parity[1] ) ;
                Sr[1] = PTemp[pos];
                _mm512_storeu_si512 ( PTemp, parity[2] ) ;
                Sr[2] = PTemp[pos];
                _mm512_storeu_si512 ( PTemp, parity[3] ) ;
                Sr[3] = PTemp[pos];
                _mm512_storeu_si512 ( PTemp, parity[4] ) ;
                Sr[4] = PTemp[pos];
                _mm512_storeu_si512 ( PTemp, parity[5] ) ;
                Sr[5] = PTemp[pos];
                _mm512_storeu_si512 ( PTemp, parity[6] ) ;
                Sr[6] = PTemp[pos];
                _mm512_storeu_si512 ( PTemp, parity[7] ) ;
                Sr[7] = PTemp[pos];
                _mm512_storeu_si512 ( PTemp, parity[8] ) ;
                Sr[8] = PTemp[pos];
                _mm512_storeu_si512 ( PTemp, parity[9] ) ;
                Sr[9] = PTemp[pos];
                _mm512_storeu_si512 ( PTemp, parity[10] ) ;
                Sr[10] = PTemp[pos];
                _mm512_storeu_si512 ( PTemp, parity[11] ) ;
                Sr[11] = PTemp[pos];
                _mm512_storeu_si512 ( PTemp, parity[12] ) ;
                Sr[12] = PTemp[pos];
                _mm512_storeu_si512 ( PTemp, parity[13] ) ;
                Sr[13] = PTemp[pos];
                rs_decode_palindromic_GFNI(Sr, S, 14, k, errLocs, errMags,
                             (int)pos+curPos, Lambda, B, T, Omega, data) ;
                mask = _blsr_u64(mask);
            }
        }
    }
}

// Parallel LFSRD_SR Sequencer for P = 15 Codewords
void ParallelLFSRSequencer15_GFNI(int len, int k, unsigned char **data, int decoder)
{
    int curSym, curPos ;                           // Loop counters
    __m512i parity [ 15 ], taps [ 7 ] ;             // Parity registers
    __m512i data_vec, temp [ 7 ] ;
    unsigned char **sPnt ;                         // Data lookup pointers

    // Initialize the taps to the LFSR
   taps [ 0 ] = _mm512_broadcast_i32x2(*( __m128i * ) &PCAffTab[0x2]);
   taps [ 1 ] = _mm512_broadcast_i32x2(*( __m128i * ) &PCAffTab[0xe5]);
   taps [ 2 ] = _mm512_broadcast_i32x2(*( __m128i * ) &PCAffTab[0x6a]);
   taps [ 3 ] = _mm512_broadcast_i32x2(*( __m128i * ) &PCAffTab[0xa1]);
   taps [ 4 ] = _mm512_broadcast_i32x2(*( __m128i * ) &PCAffTab[0x7e]);
   taps [ 5 ] = _mm512_broadcast_i32x2(*( __m128i * ) &PCAffTab[0x6c]);
   taps [ 6 ] = _mm512_broadcast_i32x2(*( __m128i * ) &PCAffTab[0x4]);

    // Adjust K if we are decoding
    k += decoder ? 15 : 0 ;

    // Loop through each 64 byte codeword
    for ( curPos = 0 ; curPos < len ; curPos += 64 )
    {
        sPnt = data ;
        parity [ 0 ] = _mm512_setzero_si512();
        parity [ 1 ] = _mm512_setzero_si512();
        parity [ 2 ] = _mm512_setzero_si512();
        parity [ 3 ] = _mm512_setzero_si512();
        parity [ 4 ] = _mm512_setzero_si512();
        parity [ 5 ] = _mm512_setzero_si512();
        parity [ 6 ] = _mm512_setzero_si512();
        parity [ 7 ] = _mm512_setzero_si512();
        parity [ 8 ] = _mm512_setzero_si512();
        parity [ 9 ] = _mm512_setzero_si512();
        parity [ 10 ] = _mm512_setzero_si512();
        parity [ 11 ] = _mm512_setzero_si512();
        parity [ 12 ] = _mm512_setzero_si512();
        parity [ 13 ] = _mm512_setzero_si512();
        parity [ 14 ] = _mm512_setzero_si512();

        // Loop through all the 0..k symbols
        for ( curSym = 0 ; curSym < k ; curSym ++ )
        {
            // Load next 64 bytes of Original Data
            data_vec = _mm512_stream_load_si512( *sPnt + curPos ) ;
          __builtin_prefetch ( *sPnt + curPos + 64, 0, 3 ) ;
            sPnt ++ ;

            // Add incoming data to MSB of parity, then update parities using Parallel Multiplier
            data_vec = _mm512_xor_si512( data_vec, parity [ 0 ] ) ;

            // Compute the first half of the tap values for palindromic code generator
            temp [ 0 ] = _mm512_gf2p8affine_epi64_epi8 ( data_vec, taps [ 0 ], 0 ) ;
            temp [ 1 ] = _mm512_gf2p8affine_epi64_epi8 ( data_vec, taps [ 1 ], 0 ) ;
            temp [ 2 ] = _mm512_gf2p8affine_epi64_epi8 ( data_vec, taps [ 2 ], 0 ) ;
            temp [ 3 ] = _mm512_gf2p8affine_epi64_epi8 ( data_vec, taps [ 3 ], 0 ) ;
            temp [ 4 ] = _mm512_gf2p8affine_epi64_epi8 ( data_vec, taps [ 4 ], 0 ) ;
            temp [ 5 ] = _mm512_gf2p8affine_epi64_epi8 ( data_vec, taps [ 5 ], 0 ) ;
            temp [ 6 ] = _mm512_gf2p8affine_epi64_epi8 ( data_vec, taps [ 6 ], 0 ) ;

            // Now cycle the LFSR
            parity [ 0 ] = _mm512_xor_si512 ( parity [ 1 ], temp [ 0 ] ) ;
            parity [ 1 ] = _mm512_xor_si512 ( parity [ 2 ], temp [ 1 ] ) ;
            parity [ 2 ] = _mm512_xor_si512 ( parity [ 3 ], temp [ 2 ] ) ;
            parity [ 3 ] = _mm512_xor_si512 ( parity [ 4 ], temp [ 3 ] ) ;
            parity [ 4 ] = _mm512_xor_si512 ( parity [ 5 ], temp [ 4 ] ) ;
            parity [ 5 ] = _mm512_xor_si512 ( parity [ 6 ], temp [ 5 ] ) ;
            parity [ 6 ] = _mm512_xor_si512 ( parity [ 7 ], temp [ 6 ] ) ;
            parity [ 7 ] = _mm512_xor_si512 ( parity [ 8 ], temp [ 6 ] ) ;
            parity [ 8 ] = _mm512_xor_si512 ( parity [ 9 ], temp [ 5 ] ) ;
            parity [ 9 ] = _mm512_xor_si512 ( parity [ 10 ], temp [ 4 ] ) ;
            parity [ 10 ] = _mm512_xor_si512 ( parity [ 11 ], temp [ 3 ] ) ;
            parity [ 11 ] = _mm512_xor_si512 ( parity [ 12 ], temp [ 2 ] ) ;
            parity [ 12 ] = _mm512_xor_si512 ( parity [ 13 ], temp [ 1 ] ) ;
            parity [ 13 ] = _mm512_xor_si512 ( parity [ 14 ], temp [ 0 ] ) ;
            parity [ 14 ] = data_vec ;
        }

        if ( decoder == 0 )
        {
            // Store parity back to memory
            _mm512_stream_si512( (&data [ k+0 ][curPos]), parity [ 0 ] ) ;
            _mm512_stream_si512( (&data [ k+1 ][curPos]), parity [ 1 ] ) ;
            _mm512_stream_si512( (&data [ k+2 ][curPos]), parity [ 2 ] ) ;
            _mm512_stream_si512( (&data [ k+3 ][curPos]), parity [ 3 ] ) ;
            _mm512_stream_si512( (&data [ k+4 ][curPos]), parity [ 4 ] ) ;
            _mm512_stream_si512( (&data [ k+5 ][curPos]), parity [ 5 ] ) ;
            _mm512_stream_si512( (&data [ k+6 ][curPos]), parity [ 6 ] ) ;
            _mm512_stream_si512( (&data [ k+7 ][curPos]), parity [ 7 ] ) ;
            _mm512_stream_si512( (&data [ k+8 ][curPos]), parity [ 8 ] ) ;
            _mm512_stream_si512( (&data [ k+9 ][curPos]), parity [ 9 ] ) ;
            _mm512_stream_si512( (&data [ k+10 ][curPos]), parity [ 10 ] ) ;
            _mm512_stream_si512( (&data [ k+11 ][curPos]), parity [ 11 ] ) ;
            _mm512_stream_si512( (&data [ k+12 ][curPos]), parity [ 12 ] ) ;
            _mm512_stream_si512( (&data [ k+13 ][curPos]), parity [ 13 ] ) ;
            _mm512_stream_si512( (&data [ k+14 ][curPos]), parity [ 14 ] ) ;
        }
        else
        {
            // Verify Syndromes are zero
            data_vec = _mm512_or_si512 ( parity [ 0 ], parity [ 1 ] ) ;
            data_vec = _mm512_ternarylogic_epi32 ( data_vec, parity [ 2 ], parity [ 3 ], 0xFE ) ;
            data_vec = _mm512_ternarylogic_epi32 ( data_vec, parity [ 4 ], parity [ 5 ], 0xFE ) ;
            data_vec = _mm512_ternarylogic_epi32 ( data_vec, parity [ 6 ], parity [ 7 ], 0xFE ) ;
            data_vec = _mm512_ternarylogic_epi32 ( data_vec, parity [ 8 ], parity [ 9 ], 0xFE ) ;
            data_vec = _mm512_ternarylogic_epi32 ( data_vec, parity [ 10 ], parity [ 11 ], 0xFE ) ;
            data_vec = _mm512_ternarylogic_epi32 ( data_vec, parity [ 12 ], parity [ 13 ], 0xFE ) ;
            data_vec = _mm512_ternarylogic_epi32 ( data_vec, parity [ 14 ], parity [ 14 ], 0xFE ) ;
            __mmask64 mask = _mm512_test_epi8_mask ( data_vec, data_vec ) ;

            // If syndromes are not zero then process error
            while ( !_ktestz_mask64_u8 ( mask, mask ) ) 
            {
                PCErrCnt++;
                unsigned char Sr[15], S[15], errLocs[15], errMags[15] ;
                unsigned char Lambda[15+1], B[15+1], T[15+1], Omega[15+1] ;
                unsigned char PTemp[64] ;
                uint64_t pos = _tzcnt_u64 ( mask ) ;

                _mm512_storeu_si512 ( PTemp, parity[0] ) ;
                Sr[0] = PTemp[pos];
                _mm512_storeu_si512 ( PTemp, parity[1] ) ;
                Sr[1] = PTemp[pos];
                _mm512_storeu_si512 ( PTemp, parity[2] ) ;
                Sr[2] = PTemp[pos];
                _mm512_storeu_si512 ( PTemp, parity[3] ) ;
                Sr[3] = PTemp[pos];
                _mm512_storeu_si512 ( PTemp, parity[4] ) ;
                Sr[4] = PTemp[pos];
                _mm512_storeu_si512 ( PTemp, parity[5] ) ;
                Sr[5] = PTemp[pos];
                _mm512_storeu_si512 ( PTemp, parity[6] ) ;
                Sr[6] = PTemp[pos];
                _mm512_storeu_si512 ( PTemp, parity[7] ) ;
                Sr[7] = PTemp[pos];
                _mm512_storeu_si512 ( PTemp, parity[8] ) ;
                Sr[8] = PTemp[pos];
                _mm512_storeu_si512 ( PTemp, parity[9] ) ;
                Sr[9] = PTemp[pos];
                _mm512_storeu_si512 ( PTemp, parity[10] ) ;
                Sr[10] = PTemp[pos];
                _mm512_storeu_si512 ( PTemp, parity[11] ) ;
                Sr[11] = PTemp[pos];
                _mm512_storeu_si512 ( PTemp, parity[12] ) ;
                Sr[12] = PTemp[pos];
                _mm512_storeu_si512 ( PTemp, parity[13] ) ;
                Sr[13] = PTemp[pos];
                _mm512_storeu_si512 ( PTemp, parity[14] ) ;
                Sr[14] = PTemp[pos];
                rs_decode_palindromic_GFNI(Sr, S, 15, k, errLocs, errMags,
                             (int)pos+curPos, Lambda, B, T, Omega, data) ;
                mask = _blsr_u64(mask);
            }
        }
    }
}

// Parallel LFSRD_SR Sequencer for P = 16 Codewords
void ParallelLFSRSequencer16_GFNI(int len, int k, unsigned char **data, int decoder)
{
    int curSym, curPos ;                           // Loop counters
    __m512i parity [ 16 ], taps [ 8 ] ;             // Parity registers
    __m512i data_vec, temp [ 8 ] ;
    unsigned char **sPnt ;                         // Data lookup pointers

    // Initialize the taps to the LFSR
   taps [ 0 ] = _mm512_broadcast_i32x2(*( __m128i * ) &PCAffTab[0x2c]);
   taps [ 1 ] = _mm512_broadcast_i32x2(*( __m128i * ) &PCAffTab[0xe1]);
   taps [ 2 ] = _mm512_broadcast_i32x2(*( __m128i * ) &PCAffTab[0x79]);
   taps [ 3 ] = _mm512_broadcast_i32x2(*( __m128i * ) &PCAffTab[0xf0]);
   taps [ 4 ] = _mm512_broadcast_i32x2(*( __m128i * ) &PCAffTab[0x82]);
   taps [ 5 ] = _mm512_broadcast_i32x2(*( __m128i * ) &PCAffTab[0xf8]);
   taps [ 6 ] = _mm512_broadcast_i32x2(*( __m128i * ) &PCAffTab[0xaa]);
   taps [ 7 ] = _mm512_broadcast_i32x2(*( __m128i * ) &PCAffTab[0x7]);

    // Adjust K if we are decoding
    k += decoder ? 16 : 0 ;

    // Loop through each 64 byte codeword
    for ( curPos = 0 ; curPos < len ; curPos += 64 )
    {
        sPnt = data ;
        parity [ 0 ] = _mm512_setzero_si512();
        parity [ 1 ] = _mm512_setzero_si512();
        parity [ 2 ] = _mm512_setzero_si512();
        parity [ 3 ] = _mm512_setzero_si512();
        parity [ 4 ] = _mm512_setzero_si512();
        parity [ 5 ] = _mm512_setzero_si512();
        parity [ 6 ] = _mm512_setzero_si512();
        parity [ 7 ] = _mm512_setzero_si512();
        parity [ 8 ] = _mm512_setzero_si512();
        parity [ 9 ] = _mm512_setzero_si512();
        parity [ 10 ] = _mm512_setzero_si512();
        parity [ 11 ] = _mm512_setzero_si512();
        parity [ 12 ] = _mm512_setzero_si512();
        parity [ 13 ] = _mm512_setzero_si512();
        parity [ 14 ] = _mm512_setzero_si512();
        parity [ 15 ] = _mm512_setzero_si512();

        // Loop through all the 0..k symbols
        for ( curSym = 0 ; curSym < k ; curSym ++ )
        {
            // Load next 64 bytes of Original Data
            data_vec = _mm512_stream_load_si512( *sPnt + curPos ) ;
          __builtin_prefetch ( *sPnt + curPos + 64, 0, 3 ) ;
            sPnt ++ ;

            // Add incoming data to MSB of parity, then update parities using Parallel Multiplier
            data_vec = _mm512_xor_si512( data_vec, parity [ 0 ] ) ;

            // Compute the first half of the tap values for palindromic code generator
            temp [ 0 ] = _mm512_gf2p8affine_epi64_epi8 ( data_vec, taps [ 0 ], 0 ) ;
            temp [ 1 ] = _mm512_gf2p8affine_epi64_epi8 ( data_vec, taps [ 1 ], 0 ) ;
            temp [ 2 ] = _mm512_gf2p8affine_epi64_epi8 ( data_vec, taps [ 2 ], 0 ) ;
            temp [ 3 ] = _mm512_gf2p8affine_epi64_epi8 ( data_vec, taps [ 3 ], 0 ) ;
            temp [ 4 ] = _mm512_gf2p8affine_epi64_epi8 ( data_vec, taps [ 4 ], 0 ) ;
            temp [ 5 ] = _mm512_gf2p8affine_epi64_epi8 ( data_vec, taps [ 5 ], 0 ) ;
            temp [ 6 ] = _mm512_gf2p8affine_epi64_epi8 ( data_vec, taps [ 6 ], 0 ) ;
            temp [ 7 ] = _mm512_gf2p8affine_epi64_epi8 ( data_vec, taps [ 7 ], 0 ) ;

            // Now cycle the LFSR
            parity [ 0 ] = _mm512_xor_si512 ( parity [ 1 ], temp [ 0 ] ) ;
            parity [ 1 ] = _mm512_xor_si512 ( parity [ 2 ], temp [ 1 ] ) ;
            parity [ 2 ] = _mm512_xor_si512 ( parity [ 3 ], temp [ 2 ] ) ;
            parity [ 3 ] = _mm512_xor_si512 ( parity [ 4 ], temp [ 3 ] ) ;
            parity [ 4 ] = _mm512_xor_si512 ( parity [ 5 ], temp [ 4 ] ) ;
            parity [ 5 ] = _mm512_xor_si512 ( parity [ 6 ], temp [ 5 ] ) ;
            parity [ 6 ] = _mm512_xor_si512 ( parity [ 7 ], temp [ 6 ] ) ;
            parity [ 7 ] = _mm512_xor_si512 ( parity [ 8 ], temp [ 7 ] ) ;
            parity [ 8 ] = _mm512_xor_si512 ( parity [ 9 ], temp [ 6 ] ) ;
            parity [ 9 ] = _mm512_xor_si512 ( parity [ 10 ], temp [ 5 ] ) ;
            parity [ 10 ] = _mm512_xor_si512 ( parity [ 11 ], temp [ 4 ] ) ;
            parity [ 11 ] = _mm512_xor_si512 ( parity [ 12 ], temp [ 3 ] ) ;
            parity [ 12 ] = _mm512_xor_si512 ( parity [ 13 ], temp [ 2 ] ) ;
            parity [ 13 ] = _mm512_xor_si512 ( parity [ 14 ], temp [ 1 ] ) ;
            parity [ 14 ] = _mm512_xor_si512 ( parity [ 15 ], temp [ 0 ] ) ;
            parity [ 15 ] = data_vec ;
        }

        if ( decoder == 0 )
        {
            // Store parity back to memory
            _mm512_stream_si512( (&data [ k+0 ][curPos]), parity [ 0 ] ) ;
            _mm512_stream_si512( (&data [ k+1 ][curPos]), parity [ 1 ] ) ;
            _mm512_stream_si512( (&data [ k+2 ][curPos]), parity [ 2 ] ) ;
            _mm512_stream_si512( (&data [ k+3 ][curPos]), parity [ 3 ] ) ;
            _mm512_stream_si512( (&data [ k+4 ][curPos]), parity [ 4 ] ) ;
            _mm512_stream_si512( (&data [ k+5 ][curPos]), parity [ 5 ] ) ;
            _mm512_stream_si512( (&data [ k+6 ][curPos]), parity [ 6 ] ) ;
            _mm512_stream_si512( (&data [ k+7 ][curPos]), parity [ 7 ] ) ;
            _mm512_stream_si512( (&data [ k+8 ][curPos]), parity [ 8 ] ) ;
            _mm512_stream_si512( (&data [ k+9 ][curPos]), parity [ 9 ] ) ;
            _mm512_stream_si512( (&data [ k+10 ][curPos]), parity [ 10 ] ) ;
            _mm512_stream_si512( (&data [ k+11 ][curPos]), parity [ 11 ] ) ;
            _mm512_stream_si512( (&data [ k+12 ][curPos]), parity [ 12 ] ) ;
            _mm512_stream_si512( (&data [ k+13 ][curPos]), parity [ 13 ] ) ;
            _mm512_stream_si512( (&data [ k+14 ][curPos]), parity [ 14 ] ) ;
            _mm512_stream_si512( (&data [ k+15 ][curPos]), parity [ 15 ] ) ;
        }
        else
        {
            // Verify Syndromes are zero
            data_vec = _mm512_or_si512 ( parity [ 0 ], parity [ 1 ] ) ;
            data_vec = _mm512_ternarylogic_epi32 ( data_vec, parity [ 2 ], parity [ 3 ], 0xFE ) ;
            data_vec = _mm512_ternarylogic_epi32 ( data_vec, parity [ 4 ], parity [ 5 ], 0xFE ) ;
            data_vec = _mm512_ternarylogic_epi32 ( data_vec, parity [ 6 ], parity [ 7 ], 0xFE ) ;
            data_vec = _mm512_ternarylogic_epi32 ( data_vec, parity [ 8 ], parity [ 9 ], 0xFE ) ;
            data_vec = _mm512_ternarylogic_epi32 ( data_vec, parity [ 10 ], parity [ 11 ], 0xFE ) ;
            data_vec = _mm512_ternarylogic_epi32 ( data_vec, parity [ 12 ], parity [ 13 ], 0xFE ) ;
            data_vec = _mm512_ternarylogic_epi32 ( data_vec, parity [ 14 ], parity [ 15 ], 0xFE ) ;
            __mmask64 mask = _mm512_test_epi8_mask ( data_vec, data_vec ) ;

            // If syndromes are not zero then process error
            while ( !_ktestz_mask64_u8 ( mask, mask ) ) 
            {
                PCErrCnt++;
                unsigned char Sr[16], S[16], errLocs[16], errMags[16] ;
                unsigned char Lambda[16+1], B[16+1], T[16+1], Omega[16+1] ;
                unsigned char PTemp[64] ;
                uint64_t pos = _tzcnt_u64 ( mask ) ;

                _mm512_storeu_si512 ( PTemp, parity[0] ) ;
                Sr[0] = PTemp[pos];
                _mm512_storeu_si512 ( PTemp, parity[1] ) ;
                Sr[1] = PTemp[pos];
                _mm512_storeu_si512 ( PTemp, parity[2] ) ;
                Sr[2] = PTemp[pos];
                _mm512_storeu_si512 ( PTemp, parity[3] ) ;
                Sr[3] = PTemp[pos];
                _mm512_storeu_si512 ( PTemp, parity[4] ) ;
                Sr[4] = PTemp[pos];
                _mm512_storeu_si512 ( PTemp, parity[5] ) ;
                Sr[5] = PTemp[pos];
                _mm512_storeu_si512 ( PTemp, parity[6] ) ;
                Sr[6] = PTemp[pos];
                _mm512_storeu_si512 ( PTemp, parity[7] ) ;
                Sr[7] = PTemp[pos];
                _mm512_storeu_si512 ( PTemp, parity[8] ) ;
                Sr[8] = PTemp[pos];
                _mm512_storeu_si512 ( PTemp, parity[9] ) ;
                Sr[9] = PTemp[pos];
                _mm512_storeu_si512 ( PTemp, parity[10] ) ;
                Sr[10] = PTemp[pos];
                _mm512_storeu_si512 ( PTemp, parity[11] ) ;
                Sr[11] = PTemp[pos];
                _mm512_storeu_si512 ( PTemp, parity[12] ) ;
                Sr[12] = PTemp[pos];
                _mm512_storeu_si512 ( PTemp, parity[13] ) ;
                Sr[13] = PTemp[pos];
                _mm512_storeu_si512 ( PTemp, parity[14] ) ;
                Sr[14] = PTemp[pos];
                _mm512_storeu_si512 ( PTemp, parity[15] ) ;
                Sr[15] = PTemp[pos];
                rs_decode_palindromic_GFNI(Sr, S, 16, k, errLocs, errMags,
                             (int)pos+curPos, Lambda, B, T, Omega, data) ;
                mask = _blsr_u64(mask);
            }
        }
    }
}

// Parallel LFSRD_SR Sequencer for P = 17 Codewords
void ParallelLFSRSequencer17_GFNI(int len, int k, unsigned char **data, int decoder)
{
    int curSym, curPos ;                           // Loop counters
    __m512i parity [ 17 ], taps [ 8 ] ;             // Parity registers
    __m512i data_vec, temp [ 8 ] ;
    unsigned char **sPnt ;                         // Data lookup pointers

    // Initialize the taps to the LFSR
   taps [ 0 ] = _mm512_broadcast_i32x2(*( __m128i * ) &PCAffTab[0x9c]);
   taps [ 1 ] = _mm512_broadcast_i32x2(*( __m128i * ) &PCAffTab[0xc5]);
   taps [ 2 ] = _mm512_broadcast_i32x2(*( __m128i * ) &PCAffTab[0x62]);
   taps [ 3 ] = _mm512_broadcast_i32x2(*( __m128i * ) &PCAffTab[0x9f]);
   taps [ 4 ] = _mm512_broadcast_i32x2(*( __m128i * ) &PCAffTab[0x8]);
   taps [ 5 ] = _mm512_broadcast_i32x2(*( __m128i * ) &PCAffTab[0x41]);
   taps [ 6 ] = _mm512_broadcast_i32x2(*( __m128i * ) &PCAffTab[0xc2]);
   taps [ 7 ] = _mm512_broadcast_i32x2(*( __m128i * ) &PCAffTab[0x2a]);

    // Adjust K if we are decoding
    k += decoder ? 17 : 0 ;

    // Loop through each 64 byte codeword
    for ( curPos = 0 ; curPos < len ; curPos += 64 )
    {
        sPnt = data ;
        parity [ 0 ] = _mm512_setzero_si512();
        parity [ 1 ] = _mm512_setzero_si512();
        parity [ 2 ] = _mm512_setzero_si512();
        parity [ 3 ] = _mm512_setzero_si512();
        parity [ 4 ] = _mm512_setzero_si512();
        parity [ 5 ] = _mm512_setzero_si512();
        parity [ 6 ] = _mm512_setzero_si512();
        parity [ 7 ] = _mm512_setzero_si512();
        parity [ 8 ] = _mm512_setzero_si512();
        parity [ 9 ] = _mm512_setzero_si512();
        parity [ 10 ] = _mm512_setzero_si512();
        parity [ 11 ] = _mm512_setzero_si512();
        parity [ 12 ] = _mm512_setzero_si512();
        parity [ 13 ] = _mm512_setzero_si512();
        parity [ 14 ] = _mm512_setzero_si512();
        parity [ 15 ] = _mm512_setzero_si512();
        parity [ 16 ] = _mm512_setzero_si512();

        // Loop through all the 0..k symbols
        for ( curSym = 0 ; curSym < k ; curSym ++ )
        {
            // Load next 64 bytes of Original Data
            data_vec = _mm512_stream_load_si512( *sPnt + curPos ) ;
          __builtin_prefetch ( *sPnt + curPos + 64, 0, 3 ) ;
            sPnt ++ ;

            // Add incoming data to MSB of parity, then update parities using Parallel Multiplier
            data_vec = _mm512_xor_si512( data_vec, parity [ 0 ] ) ;

            // Compute the first half of the tap values for palindromic code generator
            temp [ 0 ] = _mm512_gf2p8affine_epi64_epi8 ( data_vec, taps [ 0 ], 0 ) ;
            temp [ 1 ] = _mm512_gf2p8affine_epi64_epi8 ( data_vec, taps [ 1 ], 0 ) ;
            temp [ 2 ] = _mm512_gf2p8affine_epi64_epi8 ( data_vec, taps [ 2 ], 0 ) ;
            temp [ 3 ] = _mm512_gf2p8affine_epi64_epi8 ( data_vec, taps [ 3 ], 0 ) ;
            temp [ 4 ] = _mm512_gf2p8affine_epi64_epi8 ( data_vec, taps [ 4 ], 0 ) ;
            temp [ 5 ] = _mm512_gf2p8affine_epi64_epi8 ( data_vec, taps [ 5 ], 0 ) ;
            temp [ 6 ] = _mm512_gf2p8affine_epi64_epi8 ( data_vec, taps [ 6 ], 0 ) ;
            temp [ 7 ] = _mm512_gf2p8affine_epi64_epi8 ( data_vec, taps [ 7 ], 0 ) ;

            // Now cycle the LFSR
            parity [ 0 ] = _mm512_xor_si512 ( parity [ 1 ], temp [ 0 ] ) ;
            parity [ 1 ] = _mm512_xor_si512 ( parity [ 2 ], temp [ 1 ] ) ;
            parity [ 2 ] = _mm512_xor_si512 ( parity [ 3 ], temp [ 2 ] ) ;
            parity [ 3 ] = _mm512_xor_si512 ( parity [ 4 ], temp [ 3 ] ) ;
            parity [ 4 ] = _mm512_xor_si512 ( parity [ 5 ], temp [ 4 ] ) ;
            parity [ 5 ] = _mm512_xor_si512 ( parity [ 6 ], temp [ 5 ] ) ;
            parity [ 6 ] = _mm512_xor_si512 ( parity [ 7 ], temp [ 6 ] ) ;
            parity [ 7 ] = _mm512_xor_si512 ( parity [ 8 ], temp [ 7 ] ) ;
            parity [ 8 ] = _mm512_xor_si512 ( parity [ 9 ], temp [ 7 ] ) ;
            parity [ 9 ] = _mm512_xor_si512 ( parity [ 10 ], temp [ 6 ] ) ;
            parity [ 10 ] = _mm512_xor_si512 ( parity [ 11 ], temp [ 5 ] ) ;
            parity [ 11 ] = _mm512_xor_si512 ( parity [ 12 ], temp [ 4 ] ) ;
            parity [ 12 ] = _mm512_xor_si512 ( parity [ 13 ], temp [ 3 ] ) ;
            parity [ 13 ] = _mm512_xor_si512 ( parity [ 14 ], temp [ 2 ] ) ;
            parity [ 14 ] = _mm512_xor_si512 ( parity [ 15 ], temp [ 1 ] ) ;
            parity [ 15 ] = _mm512_xor_si512 ( parity [ 16 ], temp [ 0 ] ) ;
            parity [ 16 ] = data_vec ;
        }

        if ( decoder == 0 )
        {
            // Store parity back to memory
            _mm512_stream_si512( (&data [ k+0 ][curPos]), parity [ 0 ] ) ;
            _mm512_stream_si512( (&data [ k+1 ][curPos]), parity [ 1 ] ) ;
            _mm512_stream_si512( (&data [ k+2 ][curPos]), parity [ 2 ] ) ;
            _mm512_stream_si512( (&data [ k+3 ][curPos]), parity [ 3 ] ) ;
            _mm512_stream_si512( (&data [ k+4 ][curPos]), parity [ 4 ] ) ;
            _mm512_stream_si512( (&data [ k+5 ][curPos]), parity [ 5 ] ) ;
            _mm512_stream_si512( (&data [ k+6 ][curPos]), parity [ 6 ] ) ;
            _mm512_stream_si512( (&data [ k+7 ][curPos]), parity [ 7 ] ) ;
            _mm512_stream_si512( (&data [ k+8 ][curPos]), parity [ 8 ] ) ;
            _mm512_stream_si512( (&data [ k+9 ][curPos]), parity [ 9 ] ) ;
            _mm512_stream_si512( (&data [ k+10 ][curPos]), parity [ 10 ] ) ;
            _mm512_stream_si512( (&data [ k+11 ][curPos]), parity [ 11 ] ) ;
            _mm512_stream_si512( (&data [ k+12 ][curPos]), parity [ 12 ] ) ;
            _mm512_stream_si512( (&data [ k+13 ][curPos]), parity [ 13 ] ) ;
            _mm512_stream_si512( (&data [ k+14 ][curPos]), parity [ 14 ] ) ;
            _mm512_stream_si512( (&data [ k+15 ][curPos]), parity [ 15 ] ) ;
            _mm512_stream_si512( (&data [ k+16 ][curPos]), parity [ 16 ] ) ;
        }
        else
        {
            // Verify Syndromes are zero
            data_vec = _mm512_or_si512 ( parity [ 0 ], parity [ 1 ] ) ;
            data_vec = _mm512_ternarylogic_epi32 ( data_vec, parity [ 2 ], parity [ 3 ], 0xFE ) ;
            data_vec = _mm512_ternarylogic_epi32 ( data_vec, parity [ 4 ], parity [ 5 ], 0xFE ) ;
            data_vec = _mm512_ternarylogic_epi32 ( data_vec, parity [ 6 ], parity [ 7 ], 0xFE ) ;
            data_vec = _mm512_ternarylogic_epi32 ( data_vec, parity [ 8 ], parity [ 9 ], 0xFE ) ;
            data_vec = _mm512_ternarylogic_epi32 ( data_vec, parity [ 10 ], parity [ 11 ], 0xFE ) ;
            data_vec = _mm512_ternarylogic_epi32 ( data_vec, parity [ 12 ], parity [ 13 ], 0xFE ) ;
            data_vec = _mm512_ternarylogic_epi32 ( data_vec, parity [ 14 ], parity [ 15 ], 0xFE ) ;
            data_vec = _mm512_ternarylogic_epi32 ( data_vec, parity [ 16 ], parity [ 16 ], 0xFE ) ;
            __mmask64 mask = _mm512_test_epi8_mask ( data_vec, data_vec ) ;

            // If syndromes are not zero then process error
            while ( !_ktestz_mask64_u8 ( mask, mask ) ) 
            {
                PCErrCnt++;
                unsigned char Sr[17], S[17], errLocs[17], errMags[17] ;
                unsigned char Lambda[17+1], B[17+1], T[17+1], Omega[17+1] ;
                unsigned char PTemp[64] ;
                uint64_t pos = _tzcnt_u64 ( mask ) ;

                _mm512_storeu_si512 ( PTemp, parity[0] ) ;
                Sr[0] = PTemp[pos];
                _mm512_storeu_si512 ( PTemp, parity[1] ) ;
                Sr[1] = PTemp[pos];
                _mm512_storeu_si512 ( PTemp, parity[2] ) ;
                Sr[2] = PTemp[pos];
                _mm512_storeu_si512 ( PTemp, parity[3] ) ;
                Sr[3] = PTemp[pos];
                _mm512_storeu_si512 ( PTemp, parity[4] ) ;
                Sr[4] = PTemp[pos];
                _mm512_storeu_si512 ( PTemp, parity[5] ) ;
                Sr[5] = PTemp[pos];
                _mm512_storeu_si512 ( PTemp, parity[6] ) ;
                Sr[6] = PTemp[pos];
                _mm512_storeu_si512 ( PTemp, parity[7] ) ;
                Sr[7] = PTemp[pos];
                _mm512_storeu_si512 ( PTemp, parity[8] ) ;
                Sr[8] = PTemp[pos];
                _mm512_storeu_si512 ( PTemp, parity[9] ) ;
                Sr[9] = PTemp[pos];
                _mm512_storeu_si512 ( PTemp, parity[10] ) ;
                Sr[10] = PTemp[pos];
                _mm512_storeu_si512 ( PTemp, parity[11] ) ;
                Sr[11] = PTemp[pos];
                _mm512_storeu_si512 ( PTemp, parity[12] ) ;
                Sr[12] = PTemp[pos];
                _mm512_storeu_si512 ( PTemp, parity[13] ) ;
                Sr[13] = PTemp[pos];
                _mm512_storeu_si512 ( PTemp, parity[14] ) ;
                Sr[14] = PTemp[pos];
                _mm512_storeu_si512 ( PTemp, parity[15] ) ;
                Sr[15] = PTemp[pos];
                _mm512_storeu_si512 ( PTemp, parity[16] ) ;
                Sr[16] = PTemp[pos];
                rs_decode_palindromic_GFNI(Sr, S, 17, k, errLocs, errMags,
                             (int)pos+curPos, Lambda, B, T, Omega, data) ;
                mask = _blsr_u64(mask);
            }
        }
    }
}

// Parallel LFSRD_SR Sequencer for P = 18 Codewords
void ParallelLFSRSequencer18_GFNI(int len, int k, unsigned char **data, int decoder)
{
    int curSym, curPos ;                           // Loop counters
    __m512i parity [ 18 ], taps [ 9 ] ;             // Parity registers
    __m512i data_vec, temp [ 9 ] ;
    unsigned char **sPnt ;                         // Data lookup pointers

    // Initialize the taps to the LFSR
   taps [ 0 ] = _mm512_broadcast_i32x2(*( __m128i * ) &PCAffTab[0xf0]);
   taps [ 1 ] = _mm512_broadcast_i32x2(*( __m128i * ) &PCAffTab[0x9b]);
   taps [ 2 ] = _mm512_broadcast_i32x2(*( __m128i * ) &PCAffTab[0x20]);
   taps [ 3 ] = _mm512_broadcast_i32x2(*( __m128i * ) &PCAffTab[0xb9]);
   taps [ 4 ] = _mm512_broadcast_i32x2(*( __m128i * ) &PCAffTab[0x13]);
   taps [ 5 ] = _mm512_broadcast_i32x2(*( __m128i * ) &PCAffTab[0x9f]);
   taps [ 6 ] = _mm512_broadcast_i32x2(*( __m128i * ) &PCAffTab[0x6e]);
   taps [ 7 ] = _mm512_broadcast_i32x2(*( __m128i * ) &PCAffTab[0x44]);
   taps [ 8 ] = _mm512_broadcast_i32x2(*( __m128i * ) &PCAffTab[0x2e]);

    // Adjust K if we are decoding
    k += decoder ? 18 : 0 ;

    // Loop through each 64 byte codeword
    for ( curPos = 0 ; curPos < len ; curPos += 64 )
    {
        sPnt = data ;
        parity [ 0 ] = _mm512_setzero_si512();
        parity [ 1 ] = _mm512_setzero_si512();
        parity [ 2 ] = _mm512_setzero_si512();
        parity [ 3 ] = _mm512_setzero_si512();
        parity [ 4 ] = _mm512_setzero_si512();
        parity [ 5 ] = _mm512_setzero_si512();
        parity [ 6 ] = _mm512_setzero_si512();
        parity [ 7 ] = _mm512_setzero_si512();
        parity [ 8 ] = _mm512_setzero_si512();
        parity [ 9 ] = _mm512_setzero_si512();
        parity [ 10 ] = _mm512_setzero_si512();
        parity [ 11 ] = _mm512_setzero_si512();
        parity [ 12 ] = _mm512_setzero_si512();
        parity [ 13 ] = _mm512_setzero_si512();
        parity [ 14 ] = _mm512_setzero_si512();
        parity [ 15 ] = _mm512_setzero_si512();
        parity [ 16 ] = _mm512_setzero_si512();
        parity [ 17 ] = _mm512_setzero_si512();

        // Loop through all the 0..k symbols
        for ( curSym = 0 ; curSym < k ; curSym ++ )
        {
            // Load next 64 bytes of Original Data
            data_vec = _mm512_stream_load_si512( *sPnt + curPos ) ;
          __builtin_prefetch ( *sPnt + curPos + 64, 0, 3 ) ;
            sPnt ++ ;

            // Add incoming data to MSB of parity, then update parities using Parallel Multiplier
            data_vec = _mm512_xor_si512( data_vec, parity [ 0 ] ) ;

            // Compute the first half of the tap values for palindromic code generator
            temp [ 0 ] = _mm512_gf2p8affine_epi64_epi8 ( data_vec, taps [ 0 ], 0 ) ;
            temp [ 1 ] = _mm512_gf2p8affine_epi64_epi8 ( data_vec, taps [ 1 ], 0 ) ;
            temp [ 2 ] = _mm512_gf2p8affine_epi64_epi8 ( data_vec, taps [ 2 ], 0 ) ;
            temp [ 3 ] = _mm512_gf2p8affine_epi64_epi8 ( data_vec, taps [ 3 ], 0 ) ;
            temp [ 4 ] = _mm512_gf2p8affine_epi64_epi8 ( data_vec, taps [ 4 ], 0 ) ;
            temp [ 5 ] = _mm512_gf2p8affine_epi64_epi8 ( data_vec, taps [ 5 ], 0 ) ;
            temp [ 6 ] = _mm512_gf2p8affine_epi64_epi8 ( data_vec, taps [ 6 ], 0 ) ;
            temp [ 7 ] = _mm512_gf2p8affine_epi64_epi8 ( data_vec, taps [ 7 ], 0 ) ;
            temp [ 8 ] = _mm512_gf2p8affine_epi64_epi8 ( data_vec, taps [ 8 ], 0 ) ;

            // Now cycle the LFSR
            parity [ 0 ] = _mm512_xor_si512 ( parity [ 1 ], temp [ 0 ] ) ;
            parity [ 1 ] = _mm512_xor_si512 ( parity [ 2 ], temp [ 1 ] ) ;
            parity [ 2 ] = _mm512_xor_si512 ( parity [ 3 ], temp [ 2 ] ) ;
            parity [ 3 ] = _mm512_xor_si512 ( parity [ 4 ], temp [ 3 ] ) ;
            parity [ 4 ] = _mm512_xor_si512 ( parity [ 5 ], temp [ 4 ] ) ;
            parity [ 5 ] = _mm512_xor_si512 ( parity [ 6 ], temp [ 5 ] ) ;
            parity [ 6 ] = _mm512_xor_si512 ( parity [ 7 ], temp [ 6 ] ) ;
            parity [ 7 ] = _mm512_xor_si512 ( parity [ 8 ], temp [ 7 ] ) ;
            parity [ 8 ] = _mm512_xor_si512 ( parity [ 9 ], temp [ 8 ] ) ;
            parity [ 9 ] = _mm512_xor_si512 ( parity [ 10 ], temp [ 7 ] ) ;
            parity [ 10 ] = _mm512_xor_si512 ( parity [ 11 ], temp [ 6 ] ) ;
            parity [ 11 ] = _mm512_xor_si512 ( parity [ 12 ], temp [ 5 ] ) ;
            parity [ 12 ] = _mm512_xor_si512 ( parity [ 13 ], temp [ 4 ] ) ;
            parity [ 13 ] = _mm512_xor_si512 ( parity [ 14 ], temp [ 3 ] ) ;
            parity [ 14 ] = _mm512_xor_si512 ( parity [ 15 ], temp [ 2 ] ) ;
            parity [ 15 ] = _mm512_xor_si512 ( parity [ 16 ], temp [ 1 ] ) ;
            parity [ 16 ] = _mm512_xor_si512 ( parity [ 17 ], temp [ 0 ] ) ;
            parity [ 17 ] = data_vec ;
        }

        if ( decoder == 0 )
        {
            // Store parity back to memory
            _mm512_stream_si512( (&data [ k+0 ][curPos]), parity [ 0 ] ) ;
            _mm512_stream_si512( (&data [ k+1 ][curPos]), parity [ 1 ] ) ;
            _mm512_stream_si512( (&data [ k+2 ][curPos]), parity [ 2 ] ) ;
            _mm512_stream_si512( (&data [ k+3 ][curPos]), parity [ 3 ] ) ;
            _mm512_stream_si512( (&data [ k+4 ][curPos]), parity [ 4 ] ) ;
            _mm512_stream_si512( (&data [ k+5 ][curPos]), parity [ 5 ] ) ;
            _mm512_stream_si512( (&data [ k+6 ][curPos]), parity [ 6 ] ) ;
            _mm512_stream_si512( (&data [ k+7 ][curPos]), parity [ 7 ] ) ;
            _mm512_stream_si512( (&data [ k+8 ][curPos]), parity [ 8 ] ) ;
            _mm512_stream_si512( (&data [ k+9 ][curPos]), parity [ 9 ] ) ;
            _mm512_stream_si512( (&data [ k+10 ][curPos]), parity [ 10 ] ) ;
            _mm512_stream_si512( (&data [ k+11 ][curPos]), parity [ 11 ] ) ;
            _mm512_stream_si512( (&data [ k+12 ][curPos]), parity [ 12 ] ) ;
            _mm512_stream_si512( (&data [ k+13 ][curPos]), parity [ 13 ] ) ;
            _mm512_stream_si512( (&data [ k+14 ][curPos]), parity [ 14 ] ) ;
            _mm512_stream_si512( (&data [ k+15 ][curPos]), parity [ 15 ] ) ;
            _mm512_stream_si512( (&data [ k+16 ][curPos]), parity [ 16 ] ) ;
            _mm512_stream_si512( (&data [ k+17 ][curPos]), parity [ 17 ] ) ;
        }
        else
        {
            // Verify Syndromes are zero
            data_vec = _mm512_or_si512 ( parity [ 0 ], parity [ 1 ] ) ;
            data_vec = _mm512_ternarylogic_epi32 ( data_vec, parity [ 2 ], parity [ 3 ], 0xFE ) ;
            data_vec = _mm512_ternarylogic_epi32 ( data_vec, parity [ 4 ], parity [ 5 ], 0xFE ) ;
            data_vec = _mm512_ternarylogic_epi32 ( data_vec, parity [ 6 ], parity [ 7 ], 0xFE ) ;
            data_vec = _mm512_ternarylogic_epi32 ( data_vec, parity [ 8 ], parity [ 9 ], 0xFE ) ;
            data_vec = _mm512_ternarylogic_epi32 ( data_vec, parity [ 10 ], parity [ 11 ], 0xFE ) ;
            data_vec = _mm512_ternarylogic_epi32 ( data_vec, parity [ 12 ], parity [ 13 ], 0xFE ) ;
            data_vec = _mm512_ternarylogic_epi32 ( data_vec, parity [ 14 ], parity [ 15 ], 0xFE ) ;
            data_vec = _mm512_ternarylogic_epi32 ( data_vec, parity [ 16 ], parity [ 17 ], 0xFE ) ;
            __mmask64 mask = _mm512_test_epi8_mask ( data_vec, data_vec ) ;

            // If syndromes are not zero then process error
            while ( !_ktestz_mask64_u8 ( mask, mask ) ) 
            {
                PCErrCnt++;
                unsigned char Sr[18], S[18], errLocs[18], errMags[18] ;
                unsigned char Lambda[18+1], B[18+1], T[18+1], Omega[18+1] ;
                unsigned char PTemp[64] ;
                uint64_t pos = _tzcnt_u64 ( mask ) ;

                _mm512_storeu_si512 ( PTemp, parity[0] ) ;
                Sr[0] = PTemp[pos];
                _mm512_storeu_si512 ( PTemp, parity[1] ) ;
                Sr[1] = PTemp[pos];
                _mm512_storeu_si512 ( PTemp, parity[2] ) ;
                Sr[2] = PTemp[pos];
                _mm512_storeu_si512 ( PTemp, parity[3] ) ;
                Sr[3] = PTemp[pos];
                _mm512_storeu_si512 ( PTemp, parity[4] ) ;
                Sr[4] = PTemp[pos];
                _mm512_storeu_si512 ( PTemp, parity[5] ) ;
                Sr[5] = PTemp[pos];
                _mm512_storeu_si512 ( PTemp, parity[6] ) ;
                Sr[6] = PTemp[pos];
                _mm512_storeu_si512 ( PTemp, parity[7] ) ;
                Sr[7] = PTemp[pos];
                _mm512_storeu_si512 ( PTemp, parity[8] ) ;
                Sr[8] = PTemp[pos];
                _mm512_storeu_si512 ( PTemp, parity[9] ) ;
                Sr[9] = PTemp[pos];
                _mm512_storeu_si512 ( PTemp, parity[10] ) ;
                Sr[10] = PTemp[pos];
                _mm512_storeu_si512 ( PTemp, parity[11] ) ;
                Sr[11] = PTemp[pos];
                _mm512_storeu_si512 ( PTemp, parity[12] ) ;
                Sr[12] = PTemp[pos];
                _mm512_storeu_si512 ( PTemp, parity[13] ) ;
                Sr[13] = PTemp[pos];
                _mm512_storeu_si512 ( PTemp, parity[14] ) ;
                Sr[14] = PTemp[pos];
                _mm512_storeu_si512 ( PTemp, parity[15] ) ;
                Sr[15] = PTemp[pos];
                _mm512_storeu_si512 ( PTemp, parity[16] ) ;
                Sr[16] = PTemp[pos];
                _mm512_storeu_si512 ( PTemp, parity[17] ) ;
                Sr[17] = PTemp[pos];
                rs_decode_palindromic_GFNI(Sr, S, 18, k, errLocs, errMags,
                             (int)pos+curPos, Lambda, B, T, Omega, data) ;
                mask = _blsr_u64(mask);
            }
        }
    }
}

// Parallel LFSRD_SR Sequencer for P = 19 Codewords
void ParallelLFSRSequencer19_GFNI(int len, int k, unsigned char **data, int decoder)
{
    int curSym, curPos ;                           // Loop counters
    __m512i parity [ 19 ], taps [ 9 ] ;             // Parity registers
    __m512i data_vec, temp [ 9 ] ;
    unsigned char **sPnt ;                         // Data lookup pointers

    // Initialize the taps to the LFSR
   taps [ 0 ] = _mm512_broadcast_i32x2(*( __m128i * ) &PCAffTab[0x69]);
   taps [ 1 ] = _mm512_broadcast_i32x2(*( __m128i * ) &PCAffTab[0x2c]);
   taps [ 2 ] = _mm512_broadcast_i32x2(*( __m128i * ) &PCAffTab[0x78]);
   taps [ 3 ] = _mm512_broadcast_i32x2(*( __m128i * ) &PCAffTab[0xed]);
   taps [ 4 ] = _mm512_broadcast_i32x2(*( __m128i * ) &PCAffTab[0x80]);
   taps [ 5 ] = _mm512_broadcast_i32x2(*( __m128i * ) &PCAffTab[0x25]);
   taps [ 6 ] = _mm512_broadcast_i32x2(*( __m128i * ) &PCAffTab[0xb4]);
   taps [ 7 ] = _mm512_broadcast_i32x2(*( __m128i * ) &PCAffTab[0x1c]);
   taps [ 8 ] = _mm512_broadcast_i32x2(*( __m128i * ) &PCAffTab[0x2f]);

    // Adjust K if we are decoding
    k += decoder ? 19 : 0 ;

    // Loop through each 64 byte codeword
    for ( curPos = 0 ; curPos < len ; curPos += 64 )
    {
        sPnt = data ;
        parity [ 0 ] = _mm512_setzero_si512();
        parity [ 1 ] = _mm512_setzero_si512();
        parity [ 2 ] = _mm512_setzero_si512();
        parity [ 3 ] = _mm512_setzero_si512();
        parity [ 4 ] = _mm512_setzero_si512();
        parity [ 5 ] = _mm512_setzero_si512();
        parity [ 6 ] = _mm512_setzero_si512();
        parity [ 7 ] = _mm512_setzero_si512();
        parity [ 8 ] = _mm512_setzero_si512();
        parity [ 9 ] = _mm512_setzero_si512();
        parity [ 10 ] = _mm512_setzero_si512();
        parity [ 11 ] = _mm512_setzero_si512();
        parity [ 12 ] = _mm512_setzero_si512();
        parity [ 13 ] = _mm512_setzero_si512();
        parity [ 14 ] = _mm512_setzero_si512();
        parity [ 15 ] = _mm512_setzero_si512();
        parity [ 16 ] = _mm512_setzero_si512();
        parity [ 17 ] = _mm512_setzero_si512();
        parity [ 18 ] = _mm512_setzero_si512();

        // Loop through all the 0..k symbols
        for ( curSym = 0 ; curSym < k ; curSym ++ )
        {
            // Load next 64 bytes of Original Data
            data_vec = _mm512_stream_load_si512( *sPnt + curPos ) ;
          __builtin_prefetch ( *sPnt + curPos + 64, 0, 3 ) ;
            sPnt ++ ;

            // Add incoming data to MSB of parity, then update parities using Parallel Multiplier
            data_vec = _mm512_xor_si512( data_vec, parity [ 0 ] ) ;

            // Compute the first half of the tap values for palindromic code generator
            temp [ 0 ] = _mm512_gf2p8affine_epi64_epi8 ( data_vec, taps [ 0 ], 0 ) ;
            temp [ 1 ] = _mm512_gf2p8affine_epi64_epi8 ( data_vec, taps [ 1 ], 0 ) ;
            temp [ 2 ] = _mm512_gf2p8affine_epi64_epi8 ( data_vec, taps [ 2 ], 0 ) ;
            temp [ 3 ] = _mm512_gf2p8affine_epi64_epi8 ( data_vec, taps [ 3 ], 0 ) ;
            temp [ 4 ] = _mm512_gf2p8affine_epi64_epi8 ( data_vec, taps [ 4 ], 0 ) ;
            temp [ 5 ] = _mm512_gf2p8affine_epi64_epi8 ( data_vec, taps [ 5 ], 0 ) ;
            temp [ 6 ] = _mm512_gf2p8affine_epi64_epi8 ( data_vec, taps [ 6 ], 0 ) ;
            temp [ 7 ] = _mm512_gf2p8affine_epi64_epi8 ( data_vec, taps [ 7 ], 0 ) ;
            temp [ 8 ] = _mm512_gf2p8affine_epi64_epi8 ( data_vec, taps [ 8 ], 0 ) ;

            // Now cycle the LFSR
            parity [ 0 ] = _mm512_xor_si512 ( parity [ 1 ], temp [ 0 ] ) ;
            parity [ 1 ] = _mm512_xor_si512 ( parity [ 2 ], temp [ 1 ] ) ;
            parity [ 2 ] = _mm512_xor_si512 ( parity [ 3 ], temp [ 2 ] ) ;
            parity [ 3 ] = _mm512_xor_si512 ( parity [ 4 ], temp [ 3 ] ) ;
            parity [ 4 ] = _mm512_xor_si512 ( parity [ 5 ], temp [ 4 ] ) ;
            parity [ 5 ] = _mm512_xor_si512 ( parity [ 6 ], temp [ 5 ] ) ;
            parity [ 6 ] = _mm512_xor_si512 ( parity [ 7 ], temp [ 6 ] ) ;
            parity [ 7 ] = _mm512_xor_si512 ( parity [ 8 ], temp [ 7 ] ) ;
            parity [ 8 ] = _mm512_xor_si512 ( parity [ 9 ], temp [ 8 ] ) ;
            parity [ 9 ] = _mm512_xor_si512 ( parity [ 10 ], temp [ 8 ] ) ;
            parity [ 10 ] = _mm512_xor_si512 ( parity [ 11 ], temp [ 7 ] ) ;
            parity [ 11 ] = _mm512_xor_si512 ( parity [ 12 ], temp [ 6 ] ) ;
            parity [ 12 ] = _mm512_xor_si512 ( parity [ 13 ], temp [ 5 ] ) ;
            parity [ 13 ] = _mm512_xor_si512 ( parity [ 14 ], temp [ 4 ] ) ;
            parity [ 14 ] = _mm512_xor_si512 ( parity [ 15 ], temp [ 3 ] ) ;
            parity [ 15 ] = _mm512_xor_si512 ( parity [ 16 ], temp [ 2 ] ) ;
            parity [ 16 ] = _mm512_xor_si512 ( parity [ 17 ], temp [ 1 ] ) ;
            parity [ 17 ] = _mm512_xor_si512 ( parity [ 18 ], temp [ 0 ] ) ;
            parity [ 18 ] = data_vec ;
        }

        if ( decoder == 0 )
        {
            // Store parity back to memory
            _mm512_stream_si512( (&data [ k+0 ][curPos]), parity [ 0 ] ) ;
            _mm512_stream_si512( (&data [ k+1 ][curPos]), parity [ 1 ] ) ;
            _mm512_stream_si512( (&data [ k+2 ][curPos]), parity [ 2 ] ) ;
            _mm512_stream_si512( (&data [ k+3 ][curPos]), parity [ 3 ] ) ;
            _mm512_stream_si512( (&data [ k+4 ][curPos]), parity [ 4 ] ) ;
            _mm512_stream_si512( (&data [ k+5 ][curPos]), parity [ 5 ] ) ;
            _mm512_stream_si512( (&data [ k+6 ][curPos]), parity [ 6 ] ) ;
            _mm512_stream_si512( (&data [ k+7 ][curPos]), parity [ 7 ] ) ;
            _mm512_stream_si512( (&data [ k+8 ][curPos]), parity [ 8 ] ) ;
            _mm512_stream_si512( (&data [ k+9 ][curPos]), parity [ 9 ] ) ;
            _mm512_stream_si512( (&data [ k+10 ][curPos]), parity [ 10 ] ) ;
            _mm512_stream_si512( (&data [ k+11 ][curPos]), parity [ 11 ] ) ;
            _mm512_stream_si512( (&data [ k+12 ][curPos]), parity [ 12 ] ) ;
            _mm512_stream_si512( (&data [ k+13 ][curPos]), parity [ 13 ] ) ;
            _mm512_stream_si512( (&data [ k+14 ][curPos]), parity [ 14 ] ) ;
            _mm512_stream_si512( (&data [ k+15 ][curPos]), parity [ 15 ] ) ;
            _mm512_stream_si512( (&data [ k+16 ][curPos]), parity [ 16 ] ) ;
            _mm512_stream_si512( (&data [ k+17 ][curPos]), parity [ 17 ] ) ;
            _mm512_stream_si512( (&data [ k+18 ][curPos]), parity [ 18 ] ) ;
        }
        else
        {
            // Verify Syndromes are zero
            data_vec = _mm512_or_si512 ( parity [ 0 ], parity [ 1 ] ) ;
            data_vec = _mm512_ternarylogic_epi32 ( data_vec, parity [ 2 ], parity [ 3 ], 0xFE ) ;
            data_vec = _mm512_ternarylogic_epi32 ( data_vec, parity [ 4 ], parity [ 5 ], 0xFE ) ;
            data_vec = _mm512_ternarylogic_epi32 ( data_vec, parity [ 6 ], parity [ 7 ], 0xFE ) ;
            data_vec = _mm512_ternarylogic_epi32 ( data_vec, parity [ 8 ], parity [ 9 ], 0xFE ) ;
            data_vec = _mm512_ternarylogic_epi32 ( data_vec, parity [ 10 ], parity [ 11 ], 0xFE ) ;
            data_vec = _mm512_ternarylogic_epi32 ( data_vec, parity [ 12 ], parity [ 13 ], 0xFE ) ;
            data_vec = _mm512_ternarylogic_epi32 ( data_vec, parity [ 14 ], parity [ 15 ], 0xFE ) ;
            data_vec = _mm512_ternarylogic_epi32 ( data_vec, parity [ 16 ], parity [ 17 ], 0xFE ) ;
            data_vec = _mm512_ternarylogic_epi32 ( data_vec, parity [ 18 ], parity [ 18 ], 0xFE ) ;
            __mmask64 mask = _mm512_test_epi8_mask ( data_vec, data_vec ) ;

            // If syndromes are not zero then process error
            while ( !_ktestz_mask64_u8 ( mask, mask ) ) 
            {
                PCErrCnt++;
                unsigned char Sr[19], S[19], errLocs[19], errMags[19] ;
                unsigned char Lambda[19+1], B[19+1], T[19+1], Omega[19+1] ;
                unsigned char PTemp[64] ;
                uint64_t pos = _tzcnt_u64 ( mask ) ;

                _mm512_storeu_si512 ( PTemp, parity[0] ) ;
                Sr[0] = PTemp[pos];
                _mm512_storeu_si512 ( PTemp, parity[1] ) ;
                Sr[1] = PTemp[pos];
                _mm512_storeu_si512 ( PTemp, parity[2] ) ;
                Sr[2] = PTemp[pos];
                _mm512_storeu_si512 ( PTemp, parity[3] ) ;
                Sr[3] = PTemp[pos];
                _mm512_storeu_si512 ( PTemp, parity[4] ) ;
                Sr[4] = PTemp[pos];
                _mm512_storeu_si512 ( PTemp, parity[5] ) ;
                Sr[5] = PTemp[pos];
                _mm512_storeu_si512 ( PTemp, parity[6] ) ;
                Sr[6] = PTemp[pos];
                _mm512_storeu_si512 ( PTemp, parity[7] ) ;
                Sr[7] = PTemp[pos];
                _mm512_storeu_si512 ( PTemp, parity[8] ) ;
                Sr[8] = PTemp[pos];
                _mm512_storeu_si512 ( PTemp, parity[9] ) ;
                Sr[9] = PTemp[pos];
                _mm512_storeu_si512 ( PTemp, parity[10] ) ;
                Sr[10] = PTemp[pos];
                _mm512_storeu_si512 ( PTemp, parity[11] ) ;
                Sr[11] = PTemp[pos];
                _mm512_storeu_si512 ( PTemp, parity[12] ) ;
                Sr[12] = PTemp[pos];
                _mm512_storeu_si512 ( PTemp, parity[13] ) ;
                Sr[13] = PTemp[pos];
                _mm512_storeu_si512 ( PTemp, parity[14] ) ;
                Sr[14] = PTemp[pos];
                _mm512_storeu_si512 ( PTemp, parity[15] ) ;
                Sr[15] = PTemp[pos];
                _mm512_storeu_si512 ( PTemp, parity[16] ) ;
                Sr[16] = PTemp[pos];
                _mm512_storeu_si512 ( PTemp, parity[17] ) ;
                Sr[17] = PTemp[pos];
                _mm512_storeu_si512 ( PTemp, parity[18] ) ;
                Sr[18] = PTemp[pos];
                rs_decode_palindromic_GFNI(Sr, S, 19, k, errLocs, errMags,
                             (int)pos+curPos, Lambda, B, T, Omega, data) ;
                mask = _blsr_u64(mask);
            }
        }
    }
}

// Parallel LFSRD_SR Sequencer for P = 20 Codewords
void ParallelLFSRSequencer20_GFNI(int len, int k, unsigned char **data, int decoder)
{
    int curSym, curPos ;                           // Loop counters
    __m512i parity [ 20 ], taps [ 10 ] ;             // Parity registers
    __m512i data_vec, temp [ 10 ] ;
    unsigned char **sPnt ;                         // Data lookup pointers

    // Initialize the taps to the LFSR
   taps [ 0 ] = _mm512_broadcast_i32x2(*( __m128i * ) &PCAffTab[0xa9]);
   taps [ 1 ] = _mm512_broadcast_i32x2(*( __m128i * ) &PCAffTab[0xd4]);
   taps [ 2 ] = _mm512_broadcast_i32x2(*( __m128i * ) &PCAffTab[0xab]);
   taps [ 3 ] = _mm512_broadcast_i32x2(*( __m128i * ) &PCAffTab[0xcd]);
   taps [ 4 ] = _mm512_broadcast_i32x2(*( __m128i * ) &PCAffTab[0x15]);
   taps [ 5 ] = _mm512_broadcast_i32x2(*( __m128i * ) &PCAffTab[0x34]);
   taps [ 6 ] = _mm512_broadcast_i32x2(*( __m128i * ) &PCAffTab[0x7f]);
   taps [ 7 ] = _mm512_broadcast_i32x2(*( __m128i * ) &PCAffTab[0xce]);
   taps [ 8 ] = _mm512_broadcast_i32x2(*( __m128i * ) &PCAffTab[0xfa]);
   taps [ 9 ] = _mm512_broadcast_i32x2(*( __m128i * ) &PCAffTab[0xd6]);

    // Adjust K if we are decoding
    k += decoder ? 20 : 0 ;

    // Loop through each 64 byte codeword
    for ( curPos = 0 ; curPos < len ; curPos += 64 )
    {
        sPnt = data ;
        parity [ 0 ] = _mm512_setzero_si512();
        parity [ 1 ] = _mm512_setzero_si512();
        parity [ 2 ] = _mm512_setzero_si512();
        parity [ 3 ] = _mm512_setzero_si512();
        parity [ 4 ] = _mm512_setzero_si512();
        parity [ 5 ] = _mm512_setzero_si512();
        parity [ 6 ] = _mm512_setzero_si512();
        parity [ 7 ] = _mm512_setzero_si512();
        parity [ 8 ] = _mm512_setzero_si512();
        parity [ 9 ] = _mm512_setzero_si512();
        parity [ 10 ] = _mm512_setzero_si512();
        parity [ 11 ] = _mm512_setzero_si512();
        parity [ 12 ] = _mm512_setzero_si512();
        parity [ 13 ] = _mm512_setzero_si512();
        parity [ 14 ] = _mm512_setzero_si512();
        parity [ 15 ] = _mm512_setzero_si512();
        parity [ 16 ] = _mm512_setzero_si512();
        parity [ 17 ] = _mm512_setzero_si512();
        parity [ 18 ] = _mm512_setzero_si512();
        parity [ 19 ] = _mm512_setzero_si512();

        // Loop through all the 0..k symbols
        for ( curSym = 0 ; curSym < k ; curSym ++ )
        {
            // Load next 64 bytes of Original Data
            data_vec = _mm512_stream_load_si512( *sPnt + curPos ) ;
          __builtin_prefetch ( *sPnt + curPos + 64, 0, 3 ) ;
            sPnt ++ ;

            // Add incoming data to MSB of parity, then update parities using Parallel Multiplier
            data_vec = _mm512_xor_si512( data_vec, parity [ 0 ] ) ;

            // Compute the first half of the tap values for palindromic code generator
            temp [ 0 ] = _mm512_gf2p8affine_epi64_epi8 ( data_vec, taps [ 0 ], 0 ) ;
            temp [ 1 ] = _mm512_gf2p8affine_epi64_epi8 ( data_vec, taps [ 1 ], 0 ) ;
            temp [ 2 ] = _mm512_gf2p8affine_epi64_epi8 ( data_vec, taps [ 2 ], 0 ) ;
            temp [ 3 ] = _mm512_gf2p8affine_epi64_epi8 ( data_vec, taps [ 3 ], 0 ) ;
            temp [ 4 ] = _mm512_gf2p8affine_epi64_epi8 ( data_vec, taps [ 4 ], 0 ) ;
            temp [ 5 ] = _mm512_gf2p8affine_epi64_epi8 ( data_vec, taps [ 5 ], 0 ) ;
            temp [ 6 ] = _mm512_gf2p8affine_epi64_epi8 ( data_vec, taps [ 6 ], 0 ) ;
            temp [ 7 ] = _mm512_gf2p8affine_epi64_epi8 ( data_vec, taps [ 7 ], 0 ) ;
            temp [ 8 ] = _mm512_gf2p8affine_epi64_epi8 ( data_vec, taps [ 8 ], 0 ) ;
            temp [ 9 ] = _mm512_gf2p8affine_epi64_epi8 ( data_vec, taps [ 9 ], 0 ) ;

            // Now cycle the LFSR
            parity [ 0 ] = _mm512_xor_si512 ( parity [ 1 ], temp [ 0 ] ) ;
            parity [ 1 ] = _mm512_xor_si512 ( parity [ 2 ], temp [ 1 ] ) ;
            parity [ 2 ] = _mm512_xor_si512 ( parity [ 3 ], temp [ 2 ] ) ;
            parity [ 3 ] = _mm512_xor_si512 ( parity [ 4 ], temp [ 3 ] ) ;
            parity [ 4 ] = _mm512_xor_si512 ( parity [ 5 ], temp [ 4 ] ) ;
            parity [ 5 ] = _mm512_xor_si512 ( parity [ 6 ], temp [ 5 ] ) ;
            parity [ 6 ] = _mm512_xor_si512 ( parity [ 7 ], temp [ 6 ] ) ;
            parity [ 7 ] = _mm512_xor_si512 ( parity [ 8 ], temp [ 7 ] ) ;
            parity [ 8 ] = _mm512_xor_si512 ( parity [ 9 ], temp [ 8 ] ) ;
            parity [ 9 ] = _mm512_xor_si512 ( parity [ 10 ], temp [ 9 ] ) ;
            parity [ 10 ] = _mm512_xor_si512 ( parity [ 11 ], temp [ 8 ] ) ;
            parity [ 11 ] = _mm512_xor_si512 ( parity [ 12 ], temp [ 7 ] ) ;
            parity [ 12 ] = _mm512_xor_si512 ( parity [ 13 ], temp [ 6 ] ) ;
            parity [ 13 ] = _mm512_xor_si512 ( parity [ 14 ], temp [ 5 ] ) ;
            parity [ 14 ] = _mm512_xor_si512 ( parity [ 15 ], temp [ 4 ] ) ;
            parity [ 15 ] = _mm512_xor_si512 ( parity [ 16 ], temp [ 3 ] ) ;
            parity [ 16 ] = _mm512_xor_si512 ( parity [ 17 ], temp [ 2 ] ) ;
            parity [ 17 ] = _mm512_xor_si512 ( parity [ 18 ], temp [ 1 ] ) ;
            parity [ 18 ] = _mm512_xor_si512 ( parity [ 19 ], temp [ 0 ] ) ;
            parity [ 19 ] = data_vec ;
        }

        if ( decoder == 0 )
        {
            // Store parity back to memory
            _mm512_stream_si512( (&data [ k+0 ][curPos]), parity [ 0 ] ) ;
            _mm512_stream_si512( (&data [ k+1 ][curPos]), parity [ 1 ] ) ;
            _mm512_stream_si512( (&data [ k+2 ][curPos]), parity [ 2 ] ) ;
            _mm512_stream_si512( (&data [ k+3 ][curPos]), parity [ 3 ] ) ;
            _mm512_stream_si512( (&data [ k+4 ][curPos]), parity [ 4 ] ) ;
            _mm512_stream_si512( (&data [ k+5 ][curPos]), parity [ 5 ] ) ;
            _mm512_stream_si512( (&data [ k+6 ][curPos]), parity [ 6 ] ) ;
            _mm512_stream_si512( (&data [ k+7 ][curPos]), parity [ 7 ] ) ;
            _mm512_stream_si512( (&data [ k+8 ][curPos]), parity [ 8 ] ) ;
            _mm512_stream_si512( (&data [ k+9 ][curPos]), parity [ 9 ] ) ;
            _mm512_stream_si512( (&data [ k+10 ][curPos]), parity [ 10 ] ) ;
            _mm512_stream_si512( (&data [ k+11 ][curPos]), parity [ 11 ] ) ;
            _mm512_stream_si512( (&data [ k+12 ][curPos]), parity [ 12 ] ) ;
            _mm512_stream_si512( (&data [ k+13 ][curPos]), parity [ 13 ] ) ;
            _mm512_stream_si512( (&data [ k+14 ][curPos]), parity [ 14 ] ) ;
            _mm512_stream_si512( (&data [ k+15 ][curPos]), parity [ 15 ] ) ;
            _mm512_stream_si512( (&data [ k+16 ][curPos]), parity [ 16 ] ) ;
            _mm512_stream_si512( (&data [ k+17 ][curPos]), parity [ 17 ] ) ;
            _mm512_stream_si512( (&data [ k+18 ][curPos]), parity [ 18 ] ) ;
            _mm512_stream_si512( (&data [ k+19 ][curPos]), parity [ 19 ] ) ;
        }
        else
        {
            // Verify Syndromes are zero
            data_vec = _mm512_or_si512 ( parity [ 0 ], parity [ 1 ] ) ;
            data_vec = _mm512_ternarylogic_epi32 ( data_vec, parity [ 2 ], parity [ 3 ], 0xFE ) ;
            data_vec = _mm512_ternarylogic_epi32 ( data_vec, parity [ 4 ], parity [ 5 ], 0xFE ) ;
            data_vec = _mm512_ternarylogic_epi32 ( data_vec, parity [ 6 ], parity [ 7 ], 0xFE ) ;
            data_vec = _mm512_ternarylogic_epi32 ( data_vec, parity [ 8 ], parity [ 9 ], 0xFE ) ;
            data_vec = _mm512_ternarylogic_epi32 ( data_vec, parity [ 10 ], parity [ 11 ], 0xFE ) ;
            data_vec = _mm512_ternarylogic_epi32 ( data_vec, parity [ 12 ], parity [ 13 ], 0xFE ) ;
            data_vec = _mm512_ternarylogic_epi32 ( data_vec, parity [ 14 ], parity [ 15 ], 0xFE ) ;
            data_vec = _mm512_ternarylogic_epi32 ( data_vec, parity [ 16 ], parity [ 17 ], 0xFE ) ;
            data_vec = _mm512_ternarylogic_epi32 ( data_vec, parity [ 18 ], parity [ 19 ], 0xFE ) ;
            __mmask64 mask = _mm512_test_epi8_mask ( data_vec, data_vec ) ;

            // If syndromes are not zero then process error
            while ( !_ktestz_mask64_u8 ( mask, mask ) ) 
            {
                PCErrCnt++;
                unsigned char Sr[20], S[20], errLocs[20], errMags[20] ;
                unsigned char Lambda[20+1], B[20+1], T[20+1], Omega[20+1] ;
                unsigned char PTemp[64] ;
                uint64_t pos = _tzcnt_u64 ( mask ) ;

                _mm512_storeu_si512 ( PTemp, parity[0] ) ;
                Sr[0] = PTemp[pos];
                _mm512_storeu_si512 ( PTemp, parity[1] ) ;
                Sr[1] = PTemp[pos];
                _mm512_storeu_si512 ( PTemp, parity[2] ) ;
                Sr[2] = PTemp[pos];
                _mm512_storeu_si512 ( PTemp, parity[3] ) ;
                Sr[3] = PTemp[pos];
                _mm512_storeu_si512 ( PTemp, parity[4] ) ;
                Sr[4] = PTemp[pos];
                _mm512_storeu_si512 ( PTemp, parity[5] ) ;
                Sr[5] = PTemp[pos];
                _mm512_storeu_si512 ( PTemp, parity[6] ) ;
                Sr[6] = PTemp[pos];
                _mm512_storeu_si512 ( PTemp, parity[7] ) ;
                Sr[7] = PTemp[pos];
                _mm512_storeu_si512 ( PTemp, parity[8] ) ;
                Sr[8] = PTemp[pos];
                _mm512_storeu_si512 ( PTemp, parity[9] ) ;
                Sr[9] = PTemp[pos];
                _mm512_storeu_si512 ( PTemp, parity[10] ) ;
                Sr[10] = PTemp[pos];
                _mm512_storeu_si512 ( PTemp, parity[11] ) ;
                Sr[11] = PTemp[pos];
                _mm512_storeu_si512 ( PTemp, parity[12] ) ;
                Sr[12] = PTemp[pos];
                _mm512_storeu_si512 ( PTemp, parity[13] ) ;
                Sr[13] = PTemp[pos];
                _mm512_storeu_si512 ( PTemp, parity[14] ) ;
                Sr[14] = PTemp[pos];
                _mm512_storeu_si512 ( PTemp, parity[15] ) ;
                Sr[15] = PTemp[pos];
                _mm512_storeu_si512 ( PTemp, parity[16] ) ;
                Sr[16] = PTemp[pos];
                _mm512_storeu_si512 ( PTemp, parity[17] ) ;
                Sr[17] = PTemp[pos];
                _mm512_storeu_si512 ( PTemp, parity[18] ) ;
                Sr[18] = PTemp[pos];
                _mm512_storeu_si512 ( PTemp, parity[19] ) ;
                Sr[19] = PTemp[pos];
                rs_decode_palindromic_GFNI(Sr, S, 20, k, errLocs, errMags,
                             (int)pos+curPos, Lambda, B, T, Omega, data) ;
                mask = _blsr_u64(mask);
            }
        }
    }
}

// Parallel LFSRD_SR Sequencer for P = 21 Codewords
void ParallelLFSRSequencer21_GFNI(int len, int k, unsigned char **data, int decoder)
{
    int curSym, curPos ;                           // Loop counters
    __m512i parity [ 21 ], taps [ 10 ] ;             // Parity registers
    __m512i data_vec, temp [ 10 ] ;
    unsigned char **sPnt ;                         // Data lookup pointers

    // Initialize the taps to the LFSR
   taps [ 0 ] = _mm512_broadcast_i32x2(*( __m128i * ) &PCAffTab[0xf4]);
   taps [ 1 ] = _mm512_broadcast_i32x2(*( __m128i * ) &PCAffTab[0xf2]);
   taps [ 2 ] = _mm512_broadcast_i32x2(*( __m128i * ) &PCAffTab[0x89]);
   taps [ 3 ] = _mm512_broadcast_i32x2(*( __m128i * ) &PCAffTab[0xa6]);
   taps [ 4 ] = _mm512_broadcast_i32x2(*( __m128i * ) &PCAffTab[0x5c]);
   taps [ 5 ] = _mm512_broadcast_i32x2(*( __m128i * ) &PCAffTab[0xfd]);
   taps [ 6 ] = _mm512_broadcast_i32x2(*( __m128i * ) &PCAffTab[0xad]);
   taps [ 7 ] = _mm512_broadcast_i32x2(*( __m128i * ) &PCAffTab[0x2d]);
   taps [ 8 ] = _mm512_broadcast_i32x2(*( __m128i * ) &PCAffTab[0x6c]);
   taps [ 9 ] = _mm512_broadcast_i32x2(*( __m128i * ) &PCAffTab[0x11]);

    // Adjust K if we are decoding
    k += decoder ? 21 : 0 ;

    // Loop through each 64 byte codeword
    for ( curPos = 0 ; curPos < len ; curPos += 64 )
    {
        sPnt = data ;
        parity [ 0 ] = _mm512_setzero_si512();
        parity [ 1 ] = _mm512_setzero_si512();
        parity [ 2 ] = _mm512_setzero_si512();
        parity [ 3 ] = _mm512_setzero_si512();
        parity [ 4 ] = _mm512_setzero_si512();
        parity [ 5 ] = _mm512_setzero_si512();
        parity [ 6 ] = _mm512_setzero_si512();
        parity [ 7 ] = _mm512_setzero_si512();
        parity [ 8 ] = _mm512_setzero_si512();
        parity [ 9 ] = _mm512_setzero_si512();
        parity [ 10 ] = _mm512_setzero_si512();
        parity [ 11 ] = _mm512_setzero_si512();
        parity [ 12 ] = _mm512_setzero_si512();
        parity [ 13 ] = _mm512_setzero_si512();
        parity [ 14 ] = _mm512_setzero_si512();
        parity [ 15 ] = _mm512_setzero_si512();
        parity [ 16 ] = _mm512_setzero_si512();
        parity [ 17 ] = _mm512_setzero_si512();
        parity [ 18 ] = _mm512_setzero_si512();
        parity [ 19 ] = _mm512_setzero_si512();
        parity [ 20 ] = _mm512_setzero_si512();

        // Loop through all the 0..k symbols
        for ( curSym = 0 ; curSym < k ; curSym ++ )
        {
            // Load next 64 bytes of Original Data
            data_vec = _mm512_stream_load_si512( *sPnt + curPos ) ;
          __builtin_prefetch ( *sPnt + curPos + 64, 0, 3 ) ;
            sPnt ++ ;

            // Add incoming data to MSB of parity, then update parities using Parallel Multiplier
            data_vec = _mm512_xor_si512( data_vec, parity [ 0 ] ) ;

            // Compute the first half of the tap values for palindromic code generator
            temp [ 0 ] = _mm512_gf2p8affine_epi64_epi8 ( data_vec, taps [ 0 ], 0 ) ;
            temp [ 1 ] = _mm512_gf2p8affine_epi64_epi8 ( data_vec, taps [ 1 ], 0 ) ;
            temp [ 2 ] = _mm512_gf2p8affine_epi64_epi8 ( data_vec, taps [ 2 ], 0 ) ;
            temp [ 3 ] = _mm512_gf2p8affine_epi64_epi8 ( data_vec, taps [ 3 ], 0 ) ;
            temp [ 4 ] = _mm512_gf2p8affine_epi64_epi8 ( data_vec, taps [ 4 ], 0 ) ;
            temp [ 5 ] = _mm512_gf2p8affine_epi64_epi8 ( data_vec, taps [ 5 ], 0 ) ;
            temp [ 6 ] = _mm512_gf2p8affine_epi64_epi8 ( data_vec, taps [ 6 ], 0 ) ;
            temp [ 7 ] = _mm512_gf2p8affine_epi64_epi8 ( data_vec, taps [ 7 ], 0 ) ;
            temp [ 8 ] = _mm512_gf2p8affine_epi64_epi8 ( data_vec, taps [ 8 ], 0 ) ;
            temp [ 9 ] = _mm512_gf2p8affine_epi64_epi8 ( data_vec, taps [ 9 ], 0 ) ;

            // Now cycle the LFSR
            parity [ 0 ] = _mm512_xor_si512 ( parity [ 1 ], temp [ 0 ] ) ;
            parity [ 1 ] = _mm512_xor_si512 ( parity [ 2 ], temp [ 1 ] ) ;
            parity [ 2 ] = _mm512_xor_si512 ( parity [ 3 ], temp [ 2 ] ) ;
            parity [ 3 ] = _mm512_xor_si512 ( parity [ 4 ], temp [ 3 ] ) ;
            parity [ 4 ] = _mm512_xor_si512 ( parity [ 5 ], temp [ 4 ] ) ;
            parity [ 5 ] = _mm512_xor_si512 ( parity [ 6 ], temp [ 5 ] ) ;
            parity [ 6 ] = _mm512_xor_si512 ( parity [ 7 ], temp [ 6 ] ) ;
            parity [ 7 ] = _mm512_xor_si512 ( parity [ 8 ], temp [ 7 ] ) ;
            parity [ 8 ] = _mm512_xor_si512 ( parity [ 9 ], temp [ 8 ] ) ;
            parity [ 9 ] = _mm512_xor_si512 ( parity [ 10 ], temp [ 9 ] ) ;
            parity [ 10 ] = _mm512_xor_si512 ( parity [ 11 ], temp [ 9 ] ) ;
            parity [ 11 ] = _mm512_xor_si512 ( parity [ 12 ], temp [ 8 ] ) ;
            parity [ 12 ] = _mm512_xor_si512 ( parity [ 13 ], temp [ 7 ] ) ;
            parity [ 13 ] = _mm512_xor_si512 ( parity [ 14 ], temp [ 6 ] ) ;
            parity [ 14 ] = _mm512_xor_si512 ( parity [ 15 ], temp [ 5 ] ) ;
            parity [ 15 ] = _mm512_xor_si512 ( parity [ 16 ], temp [ 4 ] ) ;
            parity [ 16 ] = _mm512_xor_si512 ( parity [ 17 ], temp [ 3 ] ) ;
            parity [ 17 ] = _mm512_xor_si512 ( parity [ 18 ], temp [ 2 ] ) ;
            parity [ 18 ] = _mm512_xor_si512 ( parity [ 19 ], temp [ 1 ] ) ;
            parity [ 19 ] = _mm512_xor_si512 ( parity [ 20 ], temp [ 0 ] ) ;
            parity [ 20 ] = data_vec ;
        }

        if ( decoder == 0 )
        {
            // Store parity back to memory
            _mm512_stream_si512( (&data [ k+0 ][curPos]), parity [ 0 ] ) ;
            _mm512_stream_si512( (&data [ k+1 ][curPos]), parity [ 1 ] ) ;
            _mm512_stream_si512( (&data [ k+2 ][curPos]), parity [ 2 ] ) ;
            _mm512_stream_si512( (&data [ k+3 ][curPos]), parity [ 3 ] ) ;
            _mm512_stream_si512( (&data [ k+4 ][curPos]), parity [ 4 ] ) ;
            _mm512_stream_si512( (&data [ k+5 ][curPos]), parity [ 5 ] ) ;
            _mm512_stream_si512( (&data [ k+6 ][curPos]), parity [ 6 ] ) ;
            _mm512_stream_si512( (&data [ k+7 ][curPos]), parity [ 7 ] ) ;
            _mm512_stream_si512( (&data [ k+8 ][curPos]), parity [ 8 ] ) ;
            _mm512_stream_si512( (&data [ k+9 ][curPos]), parity [ 9 ] ) ;
            _mm512_stream_si512( (&data [ k+10 ][curPos]), parity [ 10 ] ) ;
            _mm512_stream_si512( (&data [ k+11 ][curPos]), parity [ 11 ] ) ;
            _mm512_stream_si512( (&data [ k+12 ][curPos]), parity [ 12 ] ) ;
            _mm512_stream_si512( (&data [ k+13 ][curPos]), parity [ 13 ] ) ;
            _mm512_stream_si512( (&data [ k+14 ][curPos]), parity [ 14 ] ) ;
            _mm512_stream_si512( (&data [ k+15 ][curPos]), parity [ 15 ] ) ;
            _mm512_stream_si512( (&data [ k+16 ][curPos]), parity [ 16 ] ) ;
            _mm512_stream_si512( (&data [ k+17 ][curPos]), parity [ 17 ] ) ;
            _mm512_stream_si512( (&data [ k+18 ][curPos]), parity [ 18 ] ) ;
            _mm512_stream_si512( (&data [ k+19 ][curPos]), parity [ 19 ] ) ;
            _mm512_stream_si512( (&data [ k+20 ][curPos]), parity [ 20 ] ) ;
        }
        else
        {
            // Verify Syndromes are zero
            data_vec = _mm512_or_si512 ( parity [ 0 ], parity [ 1 ] ) ;
            data_vec = _mm512_ternarylogic_epi32 ( data_vec, parity [ 2 ], parity [ 3 ], 0xFE ) ;
            data_vec = _mm512_ternarylogic_epi32 ( data_vec, parity [ 4 ], parity [ 5 ], 0xFE ) ;
            data_vec = _mm512_ternarylogic_epi32 ( data_vec, parity [ 6 ], parity [ 7 ], 0xFE ) ;
            data_vec = _mm512_ternarylogic_epi32 ( data_vec, parity [ 8 ], parity [ 9 ], 0xFE ) ;
            data_vec = _mm512_ternarylogic_epi32 ( data_vec, parity [ 10 ], parity [ 11 ], 0xFE ) ;
            data_vec = _mm512_ternarylogic_epi32 ( data_vec, parity [ 12 ], parity [ 13 ], 0xFE ) ;
            data_vec = _mm512_ternarylogic_epi32 ( data_vec, parity [ 14 ], parity [ 15 ], 0xFE ) ;
            data_vec = _mm512_ternarylogic_epi32 ( data_vec, parity [ 16 ], parity [ 17 ], 0xFE ) ;
            data_vec = _mm512_ternarylogic_epi32 ( data_vec, parity [ 18 ], parity [ 19 ], 0xFE ) ;
            data_vec = _mm512_ternarylogic_epi32 ( data_vec, parity [ 20 ], parity [ 20 ], 0xFE ) ;
            __mmask64 mask = _mm512_test_epi8_mask ( data_vec, data_vec ) ;

            // If syndromes are not zero then process error
            while ( !_ktestz_mask64_u8 ( mask, mask ) ) 
            {
                PCErrCnt++;
                unsigned char Sr[21], S[21], errLocs[21], errMags[21] ;
                unsigned char Lambda[21+1], B[21+1], T[21+1], Omega[21+1] ;
                unsigned char PTemp[64] ;
                uint64_t pos = _tzcnt_u64 ( mask ) ;

                _mm512_storeu_si512 ( PTemp, parity[0] ) ;
                Sr[0] = PTemp[pos];
                _mm512_storeu_si512 ( PTemp, parity[1] ) ;
                Sr[1] = PTemp[pos];
                _mm512_storeu_si512 ( PTemp, parity[2] ) ;
                Sr[2] = PTemp[pos];
                _mm512_storeu_si512 ( PTemp, parity[3] ) ;
                Sr[3] = PTemp[pos];
                _mm512_storeu_si512 ( PTemp, parity[4] ) ;
                Sr[4] = PTemp[pos];
                _mm512_storeu_si512 ( PTemp, parity[5] ) ;
                Sr[5] = PTemp[pos];
                _mm512_storeu_si512 ( PTemp, parity[6] ) ;
                Sr[6] = PTemp[pos];
                _mm512_storeu_si512 ( PTemp, parity[7] ) ;
                Sr[7] = PTemp[pos];
                _mm512_storeu_si512 ( PTemp, parity[8] ) ;
                Sr[8] = PTemp[pos];
                _mm512_storeu_si512 ( PTemp, parity[9] ) ;
                Sr[9] = PTemp[pos];
                _mm512_storeu_si512 ( PTemp, parity[10] ) ;
                Sr[10] = PTemp[pos];
                _mm512_storeu_si512 ( PTemp, parity[11] ) ;
                Sr[11] = PTemp[pos];
                _mm512_storeu_si512 ( PTemp, parity[12] ) ;
                Sr[12] = PTemp[pos];
                _mm512_storeu_si512 ( PTemp, parity[13] ) ;
                Sr[13] = PTemp[pos];
                _mm512_storeu_si512 ( PTemp, parity[14] ) ;
                Sr[14] = PTemp[pos];
                _mm512_storeu_si512 ( PTemp, parity[15] ) ;
                Sr[15] = PTemp[pos];
                _mm512_storeu_si512 ( PTemp, parity[16] ) ;
                Sr[16] = PTemp[pos];
                _mm512_storeu_si512 ( PTemp, parity[17] ) ;
                Sr[17] = PTemp[pos];
                _mm512_storeu_si512 ( PTemp, parity[18] ) ;
                Sr[18] = PTemp[pos];
                _mm512_storeu_si512 ( PTemp, parity[19] ) ;
                Sr[19] = PTemp[pos];
                _mm512_storeu_si512 ( PTemp, parity[20] ) ;
                Sr[20] = PTemp[pos];
                rs_decode_palindromic_GFNI(Sr, S, 21, k, errLocs, errMags,
                             (int)pos+curPos, Lambda, B, T, Omega, data) ;
                mask = _blsr_u64(mask);
            }
        }
    }
}

// Parallel LFSRD_SR Sequencer for P = 22 Codewords
void ParallelLFSRSequencer22_GFNI(int len, int k, unsigned char **data, int decoder)
{
    int curSym, curPos ;                           // Loop counters
    __m512i parity [ 22 ], taps [ 11 ] ;             // Parity registers
    __m512i data_vec, temp [ 11 ] ;
    unsigned char **sPnt ;                         // Data lookup pointers

    // Initialize the taps to the LFSR
   taps [ 0 ] = _mm512_broadcast_i32x2(*( __m128i * ) &PCAffTab[0x65]);
   taps [ 1 ] = _mm512_broadcast_i32x2(*( __m128i * ) &PCAffTab[0x55]);
   taps [ 2 ] = _mm512_broadcast_i32x2(*( __m128i * ) &PCAffTab[0xfe]);
   taps [ 3 ] = _mm512_broadcast_i32x2(*( __m128i * ) &PCAffTab[0x1c]);
   taps [ 4 ] = _mm512_broadcast_i32x2(*( __m128i * ) &PCAffTab[0xfc]);
   taps [ 5 ] = _mm512_broadcast_i32x2(*( __m128i * ) &PCAffTab[0x7e]);
   taps [ 6 ] = _mm512_broadcast_i32x2(*( __m128i * ) &PCAffTab[0x99]);
   taps [ 7 ] = _mm512_broadcast_i32x2(*( __m128i * ) &PCAffTab[0x3]);
   taps [ 8 ] = _mm512_broadcast_i32x2(*( __m128i * ) &PCAffTab[0x8e]);
   taps [ 9 ] = _mm512_broadcast_i32x2(*( __m128i * ) &PCAffTab[0xe0]);
   taps [ 10 ] = _mm512_broadcast_i32x2(*( __m128i * ) &PCAffTab[0x79]);

    // Adjust K if we are decoding
    k += decoder ? 22 : 0 ;

    // Loop through each 64 byte codeword
    for ( curPos = 0 ; curPos < len ; curPos += 64 )
    {
        sPnt = data ;
        parity [ 0 ] = _mm512_setzero_si512();
        parity [ 1 ] = _mm512_setzero_si512();
        parity [ 2 ] = _mm512_setzero_si512();
        parity [ 3 ] = _mm512_setzero_si512();
        parity [ 4 ] = _mm512_setzero_si512();
        parity [ 5 ] = _mm512_setzero_si512();
        parity [ 6 ] = _mm512_setzero_si512();
        parity [ 7 ] = _mm512_setzero_si512();
        parity [ 8 ] = _mm512_setzero_si512();
        parity [ 9 ] = _mm512_setzero_si512();
        parity [ 10 ] = _mm512_setzero_si512();
        parity [ 11 ] = _mm512_setzero_si512();
        parity [ 12 ] = _mm512_setzero_si512();
        parity [ 13 ] = _mm512_setzero_si512();
        parity [ 14 ] = _mm512_setzero_si512();
        parity [ 15 ] = _mm512_setzero_si512();
        parity [ 16 ] = _mm512_setzero_si512();
        parity [ 17 ] = _mm512_setzero_si512();
        parity [ 18 ] = _mm512_setzero_si512();
        parity [ 19 ] = _mm512_setzero_si512();
        parity [ 20 ] = _mm512_setzero_si512();
        parity [ 21 ] = _mm512_setzero_si512();

        // Loop through all the 0..k symbols
        for ( curSym = 0 ; curSym < k ; curSym ++ )
        {
            // Load next 64 bytes of Original Data
            data_vec = _mm512_stream_load_si512( *sPnt + curPos ) ;
          __builtin_prefetch ( *sPnt + curPos + 64, 0, 3 ) ;
            sPnt ++ ;

            // Add incoming data to MSB of parity, then update parities using Parallel Multiplier
            data_vec = _mm512_xor_si512( data_vec, parity [ 0 ] ) ;

            // Compute the first half of the tap values for palindromic code generator
            temp [ 0 ] = _mm512_gf2p8affine_epi64_epi8 ( data_vec, taps [ 0 ], 0 ) ;
            temp [ 1 ] = _mm512_gf2p8affine_epi64_epi8 ( data_vec, taps [ 1 ], 0 ) ;
            temp [ 2 ] = _mm512_gf2p8affine_epi64_epi8 ( data_vec, taps [ 2 ], 0 ) ;
            temp [ 3 ] = _mm512_gf2p8affine_epi64_epi8 ( data_vec, taps [ 3 ], 0 ) ;
            temp [ 4 ] = _mm512_gf2p8affine_epi64_epi8 ( data_vec, taps [ 4 ], 0 ) ;
            temp [ 5 ] = _mm512_gf2p8affine_epi64_epi8 ( data_vec, taps [ 5 ], 0 ) ;
            temp [ 6 ] = _mm512_gf2p8affine_epi64_epi8 ( data_vec, taps [ 6 ], 0 ) ;
            temp [ 7 ] = _mm512_gf2p8affine_epi64_epi8 ( data_vec, taps [ 7 ], 0 ) ;
            temp [ 8 ] = _mm512_gf2p8affine_epi64_epi8 ( data_vec, taps [ 8 ], 0 ) ;
            temp [ 9 ] = _mm512_gf2p8affine_epi64_epi8 ( data_vec, taps [ 9 ], 0 ) ;
            temp [ 10 ] = _mm512_gf2p8affine_epi64_epi8 ( data_vec, taps [ 10 ], 0 ) ;

            // Now cycle the LFSR
            parity [ 0 ] = _mm512_xor_si512 ( parity [ 1 ], temp [ 0 ] ) ;
            parity [ 1 ] = _mm512_xor_si512 ( parity [ 2 ], temp [ 1 ] ) ;
            parity [ 2 ] = _mm512_xor_si512 ( parity [ 3 ], temp [ 2 ] ) ;
            parity [ 3 ] = _mm512_xor_si512 ( parity [ 4 ], temp [ 3 ] ) ;
            parity [ 4 ] = _mm512_xor_si512 ( parity [ 5 ], temp [ 4 ] ) ;
            parity [ 5 ] = _mm512_xor_si512 ( parity [ 6 ], temp [ 5 ] ) ;
            parity [ 6 ] = _mm512_xor_si512 ( parity [ 7 ], temp [ 6 ] ) ;
            parity [ 7 ] = _mm512_xor_si512 ( parity [ 8 ], temp [ 7 ] ) ;
            parity [ 8 ] = _mm512_xor_si512 ( parity [ 9 ], temp [ 8 ] ) ;
            parity [ 9 ] = _mm512_xor_si512 ( parity [ 10 ], temp [ 9 ] ) ;
            parity [ 10 ] = _mm512_xor_si512 ( parity [ 11 ], temp [ 10 ] ) ;
            parity [ 11 ] = _mm512_xor_si512 ( parity [ 12 ], temp [ 9 ] ) ;
            parity [ 12 ] = _mm512_xor_si512 ( parity [ 13 ], temp [ 8 ] ) ;
            parity [ 13 ] = _mm512_xor_si512 ( parity [ 14 ], temp [ 7 ] ) ;
            parity [ 14 ] = _mm512_xor_si512 ( parity [ 15 ], temp [ 6 ] ) ;
            parity [ 15 ] = _mm512_xor_si512 ( parity [ 16 ], temp [ 5 ] ) ;
            parity [ 16 ] = _mm512_xor_si512 ( parity [ 17 ], temp [ 4 ] ) ;
            parity [ 17 ] = _mm512_xor_si512 ( parity [ 18 ], temp [ 3 ] ) ;
            parity [ 18 ] = _mm512_xor_si512 ( parity [ 19 ], temp [ 2 ] ) ;
            parity [ 19 ] = _mm512_xor_si512 ( parity [ 20 ], temp [ 1 ] ) ;
            parity [ 20 ] = _mm512_xor_si512 ( parity [ 21 ], temp [ 0 ] ) ;
            parity [ 21 ] = data_vec ;
        }

        if ( decoder == 0 )
        {
            // Store parity back to memory
            _mm512_stream_si512( (&data [ k+0 ][curPos]), parity [ 0 ] ) ;
            _mm512_stream_si512( (&data [ k+1 ][curPos]), parity [ 1 ] ) ;
            _mm512_stream_si512( (&data [ k+2 ][curPos]), parity [ 2 ] ) ;
            _mm512_stream_si512( (&data [ k+3 ][curPos]), parity [ 3 ] ) ;
            _mm512_stream_si512( (&data [ k+4 ][curPos]), parity [ 4 ] ) ;
            _mm512_stream_si512( (&data [ k+5 ][curPos]), parity [ 5 ] ) ;
            _mm512_stream_si512( (&data [ k+6 ][curPos]), parity [ 6 ] ) ;
            _mm512_stream_si512( (&data [ k+7 ][curPos]), parity [ 7 ] ) ;
            _mm512_stream_si512( (&data [ k+8 ][curPos]), parity [ 8 ] ) ;
            _mm512_stream_si512( (&data [ k+9 ][curPos]), parity [ 9 ] ) ;
            _mm512_stream_si512( (&data [ k+10 ][curPos]), parity [ 10 ] ) ;
            _mm512_stream_si512( (&data [ k+11 ][curPos]), parity [ 11 ] ) ;
            _mm512_stream_si512( (&data [ k+12 ][curPos]), parity [ 12 ] ) ;
            _mm512_stream_si512( (&data [ k+13 ][curPos]), parity [ 13 ] ) ;
            _mm512_stream_si512( (&data [ k+14 ][curPos]), parity [ 14 ] ) ;
            _mm512_stream_si512( (&data [ k+15 ][curPos]), parity [ 15 ] ) ;
            _mm512_stream_si512( (&data [ k+16 ][curPos]), parity [ 16 ] ) ;
            _mm512_stream_si512( (&data [ k+17 ][curPos]), parity [ 17 ] ) ;
            _mm512_stream_si512( (&data [ k+18 ][curPos]), parity [ 18 ] ) ;
            _mm512_stream_si512( (&data [ k+19 ][curPos]), parity [ 19 ] ) ;
            _mm512_stream_si512( (&data [ k+20 ][curPos]), parity [ 20 ] ) ;
            _mm512_stream_si512( (&data [ k+21 ][curPos]), parity [ 21 ] ) ;
        }
        else
        {
            // Verify Syndromes are zero
            data_vec = _mm512_or_si512 ( parity [ 0 ], parity [ 1 ] ) ;
            data_vec = _mm512_ternarylogic_epi32 ( data_vec, parity [ 2 ], parity [ 3 ], 0xFE ) ;
            data_vec = _mm512_ternarylogic_epi32 ( data_vec, parity [ 4 ], parity [ 5 ], 0xFE ) ;
            data_vec = _mm512_ternarylogic_epi32 ( data_vec, parity [ 6 ], parity [ 7 ], 0xFE ) ;
            data_vec = _mm512_ternarylogic_epi32 ( data_vec, parity [ 8 ], parity [ 9 ], 0xFE ) ;
            data_vec = _mm512_ternarylogic_epi32 ( data_vec, parity [ 10 ], parity [ 11 ], 0xFE ) ;
            data_vec = _mm512_ternarylogic_epi32 ( data_vec, parity [ 12 ], parity [ 13 ], 0xFE ) ;
            data_vec = _mm512_ternarylogic_epi32 ( data_vec, parity [ 14 ], parity [ 15 ], 0xFE ) ;
            data_vec = _mm512_ternarylogic_epi32 ( data_vec, parity [ 16 ], parity [ 17 ], 0xFE ) ;
            data_vec = _mm512_ternarylogic_epi32 ( data_vec, parity [ 18 ], parity [ 19 ], 0xFE ) ;
            data_vec = _mm512_ternarylogic_epi32 ( data_vec, parity [ 20 ], parity [ 21 ], 0xFE ) ;
            __mmask64 mask = _mm512_test_epi8_mask ( data_vec, data_vec ) ;

            // If syndromes are not zero then process error
            while ( !_ktestz_mask64_u8 ( mask, mask ) ) 
            {
                PCErrCnt++;
                unsigned char Sr[22], S[22], errLocs[22], errMags[22] ;
                unsigned char Lambda[22+1], B[22+1], T[22+1], Omega[22+1] ;
                unsigned char PTemp[64] ;
                uint64_t pos = _tzcnt_u64 ( mask ) ;

                _mm512_storeu_si512 ( PTemp, parity[0] ) ;
                Sr[0] = PTemp[pos];
                _mm512_storeu_si512 ( PTemp, parity[1] ) ;
                Sr[1] = PTemp[pos];
                _mm512_storeu_si512 ( PTemp, parity[2] ) ;
                Sr[2] = PTemp[pos];
                _mm512_storeu_si512 ( PTemp, parity[3] ) ;
                Sr[3] = PTemp[pos];
                _mm512_storeu_si512 ( PTemp, parity[4] ) ;
                Sr[4] = PTemp[pos];
                _mm512_storeu_si512 ( PTemp, parity[5] ) ;
                Sr[5] = PTemp[pos];
                _mm512_storeu_si512 ( PTemp, parity[6] ) ;
                Sr[6] = PTemp[pos];
                _mm512_storeu_si512 ( PTemp, parity[7] ) ;
                Sr[7] = PTemp[pos];
                _mm512_storeu_si512 ( PTemp, parity[8] ) ;
                Sr[8] = PTemp[pos];
                _mm512_storeu_si512 ( PTemp, parity[9] ) ;
                Sr[9] = PTemp[pos];
                _mm512_storeu_si512 ( PTemp, parity[10] ) ;
                Sr[10] = PTemp[pos];
                _mm512_storeu_si512 ( PTemp, parity[11] ) ;
                Sr[11] = PTemp[pos];
                _mm512_storeu_si512 ( PTemp, parity[12] ) ;
                Sr[12] = PTemp[pos];
                _mm512_storeu_si512 ( PTemp, parity[13] ) ;
                Sr[13] = PTemp[pos];
                _mm512_storeu_si512 ( PTemp, parity[14] ) ;
                Sr[14] = PTemp[pos];
                _mm512_storeu_si512 ( PTemp, parity[15] ) ;
                Sr[15] = PTemp[pos];
                _mm512_storeu_si512 ( PTemp, parity[16] ) ;
                Sr[16] = PTemp[pos];
                _mm512_storeu_si512 ( PTemp, parity[17] ) ;
                Sr[17] = PTemp[pos];
                _mm512_storeu_si512 ( PTemp, parity[18] ) ;
                Sr[18] = PTemp[pos];
                _mm512_storeu_si512 ( PTemp, parity[19] ) ;
                Sr[19] = PTemp[pos];
                _mm512_storeu_si512 ( PTemp, parity[20] ) ;
                Sr[20] = PTemp[pos];
                _mm512_storeu_si512 ( PTemp, parity[21] ) ;
                Sr[21] = PTemp[pos];
                rs_decode_palindromic_GFNI(Sr, S, 22, k, errLocs, errMags,
                             (int)pos+curPos, Lambda, B, T, Omega, data) ;
                mask = _blsr_u64(mask);
            }
        }
    }
}

// Parallel LFSRD_SR Sequencer for P = 23 Codewords
void ParallelLFSRSequencer23_GFNI(int len, int k, unsigned char **data, int decoder)
{
    int curSym, curPos ;                           // Loop counters
    __m512i parity [ 23 ], taps [ 11 ] ;             // Parity registers
    __m512i data_vec, temp [ 11 ] ;
    unsigned char **sPnt ;                         // Data lookup pointers

    // Initialize the taps to the LFSR
   taps [ 0 ] = _mm512_broadcast_i32x2(*( __m128i * ) &PCAffTab[0xe6]);
   taps [ 1 ] = _mm512_broadcast_i32x2(*( __m128i * ) &PCAffTab[0xfd]);
   taps [ 2 ] = _mm512_broadcast_i32x2(*( __m128i * ) &PCAffTab[0x1f]);
   taps [ 3 ] = _mm512_broadcast_i32x2(*( __m128i * ) &PCAffTab[0x23]);
   taps [ 4 ] = _mm512_broadcast_i32x2(*( __m128i * ) &PCAffTab[0x36]);
   taps [ 5 ] = _mm512_broadcast_i32x2(*( __m128i * ) &PCAffTab[0x4a]);
   taps [ 6 ] = _mm512_broadcast_i32x2(*( __m128i * ) &PCAffTab[0x7d]);
   taps [ 7 ] = _mm512_broadcast_i32x2(*( __m128i * ) &PCAffTab[0x95]);
   taps [ 8 ] = _mm512_broadcast_i32x2(*( __m128i * ) &PCAffTab[0x71]);
   taps [ 9 ] = _mm512_broadcast_i32x2(*( __m128i * ) &PCAffTab[0x6a]);
   taps [ 10 ] = _mm512_broadcast_i32x2(*( __m128i * ) &PCAffTab[0x52]);

    // Adjust K if we are decoding
    k += decoder ? 23 : 0 ;

    // Loop through each 64 byte codeword
    for ( curPos = 0 ; curPos < len ; curPos += 64 )
    {
        sPnt = data ;
        parity [ 0 ] = _mm512_setzero_si512();
        parity [ 1 ] = _mm512_setzero_si512();
        parity [ 2 ] = _mm512_setzero_si512();
        parity [ 3 ] = _mm512_setzero_si512();
        parity [ 4 ] = _mm512_setzero_si512();
        parity [ 5 ] = _mm512_setzero_si512();
        parity [ 6 ] = _mm512_setzero_si512();
        parity [ 7 ] = _mm512_setzero_si512();
        parity [ 8 ] = _mm512_setzero_si512();
        parity [ 9 ] = _mm512_setzero_si512();
        parity [ 10 ] = _mm512_setzero_si512();
        parity [ 11 ] = _mm512_setzero_si512();
        parity [ 12 ] = _mm512_setzero_si512();
        parity [ 13 ] = _mm512_setzero_si512();
        parity [ 14 ] = _mm512_setzero_si512();
        parity [ 15 ] = _mm512_setzero_si512();
        parity [ 16 ] = _mm512_setzero_si512();
        parity [ 17 ] = _mm512_setzero_si512();
        parity [ 18 ] = _mm512_setzero_si512();
        parity [ 19 ] = _mm512_setzero_si512();
        parity [ 20 ] = _mm512_setzero_si512();
        parity [ 21 ] = _mm512_setzero_si512();
        parity [ 22 ] = _mm512_setzero_si512();

        // Loop through all the 0..k symbols
        for ( curSym = 0 ; curSym < k ; curSym ++ )
        {
            // Load next 64 bytes of Original Data
            data_vec = _mm512_stream_load_si512( *sPnt + curPos ) ;
          __builtin_prefetch ( *sPnt + curPos + 64, 0, 3 ) ;
            sPnt ++ ;

            // Add incoming data to MSB of parity, then update parities using Parallel Multiplier
            data_vec = _mm512_xor_si512( data_vec, parity [ 0 ] ) ;

            // Compute the first half of the tap values for palindromic code generator
            temp [ 0 ] = _mm512_gf2p8affine_epi64_epi8 ( data_vec, taps [ 0 ], 0 ) ;
            temp [ 1 ] = _mm512_gf2p8affine_epi64_epi8 ( data_vec, taps [ 1 ], 0 ) ;
            temp [ 2 ] = _mm512_gf2p8affine_epi64_epi8 ( data_vec, taps [ 2 ], 0 ) ;
            temp [ 3 ] = _mm512_gf2p8affine_epi64_epi8 ( data_vec, taps [ 3 ], 0 ) ;
            temp [ 4 ] = _mm512_gf2p8affine_epi64_epi8 ( data_vec, taps [ 4 ], 0 ) ;
            temp [ 5 ] = _mm512_gf2p8affine_epi64_epi8 ( data_vec, taps [ 5 ], 0 ) ;
            temp [ 6 ] = _mm512_gf2p8affine_epi64_epi8 ( data_vec, taps [ 6 ], 0 ) ;
            temp [ 7 ] = _mm512_gf2p8affine_epi64_epi8 ( data_vec, taps [ 7 ], 0 ) ;
            temp [ 8 ] = _mm512_gf2p8affine_epi64_epi8 ( data_vec, taps [ 8 ], 0 ) ;
            temp [ 9 ] = _mm512_gf2p8affine_epi64_epi8 ( data_vec, taps [ 9 ], 0 ) ;
            temp [ 10 ] = _mm512_gf2p8affine_epi64_epi8 ( data_vec, taps [ 10 ], 0 ) ;

            // Now cycle the LFSR
            parity [ 0 ] = _mm512_xor_si512 ( parity [ 1 ], temp [ 0 ] ) ;
            parity [ 1 ] = _mm512_xor_si512 ( parity [ 2 ], temp [ 1 ] ) ;
            parity [ 2 ] = _mm512_xor_si512 ( parity [ 3 ], temp [ 2 ] ) ;
            parity [ 3 ] = _mm512_xor_si512 ( parity [ 4 ], temp [ 3 ] ) ;
            parity [ 4 ] = _mm512_xor_si512 ( parity [ 5 ], temp [ 4 ] ) ;
            parity [ 5 ] = _mm512_xor_si512 ( parity [ 6 ], temp [ 5 ] ) ;
            parity [ 6 ] = _mm512_xor_si512 ( parity [ 7 ], temp [ 6 ] ) ;
            parity [ 7 ] = _mm512_xor_si512 ( parity [ 8 ], temp [ 7 ] ) ;
            parity [ 8 ] = _mm512_xor_si512 ( parity [ 9 ], temp [ 8 ] ) ;
            parity [ 9 ] = _mm512_xor_si512 ( parity [ 10 ], temp [ 9 ] ) ;
            parity [ 10 ] = _mm512_xor_si512 ( parity [ 11 ], temp [ 10 ] ) ;
            parity [ 11 ] = _mm512_xor_si512 ( parity [ 12 ], temp [ 10 ] ) ;
            parity [ 12 ] = _mm512_xor_si512 ( parity [ 13 ], temp [ 9 ] ) ;
            parity [ 13 ] = _mm512_xor_si512 ( parity [ 14 ], temp [ 8 ] ) ;
            parity [ 14 ] = _mm512_xor_si512 ( parity [ 15 ], temp [ 7 ] ) ;
            parity [ 15 ] = _mm512_xor_si512 ( parity [ 16 ], temp [ 6 ] ) ;
            parity [ 16 ] = _mm512_xor_si512 ( parity [ 17 ], temp [ 5 ] ) ;
            parity [ 17 ] = _mm512_xor_si512 ( parity [ 18 ], temp [ 4 ] ) ;
            parity [ 18 ] = _mm512_xor_si512 ( parity [ 19 ], temp [ 3 ] ) ;
            parity [ 19 ] = _mm512_xor_si512 ( parity [ 20 ], temp [ 2 ] ) ;
            parity [ 20 ] = _mm512_xor_si512 ( parity [ 21 ], temp [ 1 ] ) ;
            parity [ 21 ] = _mm512_xor_si512 ( parity [ 22 ], temp [ 0 ] ) ;
            parity [ 22 ] = data_vec ;
        }

        if ( decoder == 0 )
        {
            // Store parity back to memory
            _mm512_stream_si512( (&data [ k+0 ][curPos]), parity [ 0 ] ) ;
            _mm512_stream_si512( (&data [ k+1 ][curPos]), parity [ 1 ] ) ;
            _mm512_stream_si512( (&data [ k+2 ][curPos]), parity [ 2 ] ) ;
            _mm512_stream_si512( (&data [ k+3 ][curPos]), parity [ 3 ] ) ;
            _mm512_stream_si512( (&data [ k+4 ][curPos]), parity [ 4 ] ) ;
            _mm512_stream_si512( (&data [ k+5 ][curPos]), parity [ 5 ] ) ;
            _mm512_stream_si512( (&data [ k+6 ][curPos]), parity [ 6 ] ) ;
            _mm512_stream_si512( (&data [ k+7 ][curPos]), parity [ 7 ] ) ;
            _mm512_stream_si512( (&data [ k+8 ][curPos]), parity [ 8 ] ) ;
            _mm512_stream_si512( (&data [ k+9 ][curPos]), parity [ 9 ] ) ;
            _mm512_stream_si512( (&data [ k+10 ][curPos]), parity [ 10 ] ) ;
            _mm512_stream_si512( (&data [ k+11 ][curPos]), parity [ 11 ] ) ;
            _mm512_stream_si512( (&data [ k+12 ][curPos]), parity [ 12 ] ) ;
            _mm512_stream_si512( (&data [ k+13 ][curPos]), parity [ 13 ] ) ;
            _mm512_stream_si512( (&data [ k+14 ][curPos]), parity [ 14 ] ) ;
            _mm512_stream_si512( (&data [ k+15 ][curPos]), parity [ 15 ] ) ;
            _mm512_stream_si512( (&data [ k+16 ][curPos]), parity [ 16 ] ) ;
            _mm512_stream_si512( (&data [ k+17 ][curPos]), parity [ 17 ] ) ;
            _mm512_stream_si512( (&data [ k+18 ][curPos]), parity [ 18 ] ) ;
            _mm512_stream_si512( (&data [ k+19 ][curPos]), parity [ 19 ] ) ;
            _mm512_stream_si512( (&data [ k+20 ][curPos]), parity [ 20 ] ) ;
            _mm512_stream_si512( (&data [ k+21 ][curPos]), parity [ 21 ] ) ;
            _mm512_stream_si512( (&data [ k+22 ][curPos]), parity [ 22 ] ) ;
        }
        else
        {
            // Verify Syndromes are zero
            data_vec = _mm512_or_si512 ( parity [ 0 ], parity [ 1 ] ) ;
            data_vec = _mm512_ternarylogic_epi32 ( data_vec, parity [ 2 ], parity [ 3 ], 0xFE ) ;
            data_vec = _mm512_ternarylogic_epi32 ( data_vec, parity [ 4 ], parity [ 5 ], 0xFE ) ;
            data_vec = _mm512_ternarylogic_epi32 ( data_vec, parity [ 6 ], parity [ 7 ], 0xFE ) ;
            data_vec = _mm512_ternarylogic_epi32 ( data_vec, parity [ 8 ], parity [ 9 ], 0xFE ) ;
            data_vec = _mm512_ternarylogic_epi32 ( data_vec, parity [ 10 ], parity [ 11 ], 0xFE ) ;
            data_vec = _mm512_ternarylogic_epi32 ( data_vec, parity [ 12 ], parity [ 13 ], 0xFE ) ;
            data_vec = _mm512_ternarylogic_epi32 ( data_vec, parity [ 14 ], parity [ 15 ], 0xFE ) ;
            data_vec = _mm512_ternarylogic_epi32 ( data_vec, parity [ 16 ], parity [ 17 ], 0xFE ) ;
            data_vec = _mm512_ternarylogic_epi32 ( data_vec, parity [ 18 ], parity [ 19 ], 0xFE ) ;
            data_vec = _mm512_ternarylogic_epi32 ( data_vec, parity [ 20 ], parity [ 21 ], 0xFE ) ;
            data_vec = _mm512_ternarylogic_epi32 ( data_vec, parity [ 22 ], parity [ 22 ], 0xFE ) ;
            __mmask64 mask = _mm512_test_epi8_mask ( data_vec, data_vec ) ;

            // If syndromes are not zero then process error
            while ( !_ktestz_mask64_u8 ( mask, mask ) ) 
            {
                PCErrCnt++;
                unsigned char Sr[23], S[23], errLocs[23], errMags[23] ;
                unsigned char Lambda[23+1], B[23+1], T[23+1], Omega[23+1] ;
                unsigned char PTemp[64] ;
                uint64_t pos = _tzcnt_u64 ( mask ) ;

                _mm512_storeu_si512 ( PTemp, parity[0] ) ;
                Sr[0] = PTemp[pos];
                _mm512_storeu_si512 ( PTemp, parity[1] ) ;
                Sr[1] = PTemp[pos];
                _mm512_storeu_si512 ( PTemp, parity[2] ) ;
                Sr[2] = PTemp[pos];
                _mm512_storeu_si512 ( PTemp, parity[3] ) ;
                Sr[3] = PTemp[pos];
                _mm512_storeu_si512 ( PTemp, parity[4] ) ;
                Sr[4] = PTemp[pos];
                _mm512_storeu_si512 ( PTemp, parity[5] ) ;
                Sr[5] = PTemp[pos];
                _mm512_storeu_si512 ( PTemp, parity[6] ) ;
                Sr[6] = PTemp[pos];
                _mm512_storeu_si512 ( PTemp, parity[7] ) ;
                Sr[7] = PTemp[pos];
                _mm512_storeu_si512 ( PTemp, parity[8] ) ;
                Sr[8] = PTemp[pos];
                _mm512_storeu_si512 ( PTemp, parity[9] ) ;
                Sr[9] = PTemp[pos];
                _mm512_storeu_si512 ( PTemp, parity[10] ) ;
                Sr[10] = PTemp[pos];
                _mm512_storeu_si512 ( PTemp, parity[11] ) ;
                Sr[11] = PTemp[pos];
                _mm512_storeu_si512 ( PTemp, parity[12] ) ;
                Sr[12] = PTemp[pos];
                _mm512_storeu_si512 ( PTemp, parity[13] ) ;
                Sr[13] = PTemp[pos];
                _mm512_storeu_si512 ( PTemp, parity[14] ) ;
                Sr[14] = PTemp[pos];
                _mm512_storeu_si512 ( PTemp, parity[15] ) ;
                Sr[15] = PTemp[pos];
                _mm512_storeu_si512 ( PTemp, parity[16] ) ;
                Sr[16] = PTemp[pos];
                _mm512_storeu_si512 ( PTemp, parity[17] ) ;
                Sr[17] = PTemp[pos];
                _mm512_storeu_si512 ( PTemp, parity[18] ) ;
                Sr[18] = PTemp[pos];
                _mm512_storeu_si512 ( PTemp, parity[19] ) ;
                Sr[19] = PTemp[pos];
                _mm512_storeu_si512 ( PTemp, parity[20] ) ;
                Sr[20] = PTemp[pos];
                _mm512_storeu_si512 ( PTemp, parity[21] ) ;
                Sr[21] = PTemp[pos];
                _mm512_storeu_si512 ( PTemp, parity[22] ) ;
                Sr[22] = PTemp[pos];
                rs_decode_palindromic_GFNI(Sr, S, 23, k, errLocs, errMags,
                             (int)pos+curPos, Lambda, B, T, Omega, data) ;
                mask = _blsr_u64(mask);
            }
        }
    }
}

// Parallel LFSRD_SR Sequencer for P = 24 Codewords
void ParallelLFSRSequencer24_GFNI(int len, int k, unsigned char **data, int decoder)
{
    int curSym, curPos ;                           // Loop counters
    __m512i parity [ 24 ], taps [ 12 ] ;             // Parity registers
    __m512i data_vec, temp [ 12 ] ;
    unsigned char **sPnt ;                         // Data lookup pointers

    // Initialize the taps to the LFSR
   taps [ 0 ] = _mm512_broadcast_i32x2(*( __m128i * ) &PCAffTab[0xdf]);
   taps [ 1 ] = _mm512_broadcast_i32x2(*( __m128i * ) &PCAffTab[0x43]);
   taps [ 2 ] = _mm512_broadcast_i32x2(*( __m128i * ) &PCAffTab[0x3d]);
   taps [ 3 ] = _mm512_broadcast_i32x2(*( __m128i * ) &PCAffTab[0x4]);
   taps [ 4 ] = _mm512_broadcast_i32x2(*( __m128i * ) &PCAffTab[0x6]);
   taps [ 5 ] = _mm512_broadcast_i32x2(*( __m128i * ) &PCAffTab[0x46]);
   taps [ 6 ] = _mm512_broadcast_i32x2(*( __m128i * ) &PCAffTab[0x77]);
   taps [ 7 ] = _mm512_broadcast_i32x2(*( __m128i * ) &PCAffTab[0x4e]);
   taps [ 8 ] = _mm512_broadcast_i32x2(*( __m128i * ) &PCAffTab[0xc4]);
   taps [ 9 ] = _mm512_broadcast_i32x2(*( __m128i * ) &PCAffTab[0xbe]);
   taps [ 10 ] = _mm512_broadcast_i32x2(*( __m128i * ) &PCAffTab[0xd7]);
   taps [ 11 ] = _mm512_broadcast_i32x2(*( __m128i * ) &PCAffTab[0x13]);

    // Adjust K if we are decoding
    k += decoder ? 24 : 0 ;

    // Loop through each 64 byte codeword
    for ( curPos = 0 ; curPos < len ; curPos += 64 )
    {
        sPnt = data ;
        parity [ 0 ] = _mm512_setzero_si512();
        parity [ 1 ] = _mm512_setzero_si512();
        parity [ 2 ] = _mm512_setzero_si512();
        parity [ 3 ] = _mm512_setzero_si512();
        parity [ 4 ] = _mm512_setzero_si512();
        parity [ 5 ] = _mm512_setzero_si512();
        parity [ 6 ] = _mm512_setzero_si512();
        parity [ 7 ] = _mm512_setzero_si512();
        parity [ 8 ] = _mm512_setzero_si512();
        parity [ 9 ] = _mm512_setzero_si512();
        parity [ 10 ] = _mm512_setzero_si512();
        parity [ 11 ] = _mm512_setzero_si512();
        parity [ 12 ] = _mm512_setzero_si512();
        parity [ 13 ] = _mm512_setzero_si512();
        parity [ 14 ] = _mm512_setzero_si512();
        parity [ 15 ] = _mm512_setzero_si512();
        parity [ 16 ] = _mm512_setzero_si512();
        parity [ 17 ] = _mm512_setzero_si512();
        parity [ 18 ] = _mm512_setzero_si512();
        parity [ 19 ] = _mm512_setzero_si512();
        parity [ 20 ] = _mm512_setzero_si512();
        parity [ 21 ] = _mm512_setzero_si512();
        parity [ 22 ] = _mm512_setzero_si512();
        parity [ 23 ] = _mm512_setzero_si512();

        // Loop through all the 0..k symbols
        for ( curSym = 0 ; curSym < k ; curSym ++ )
        {
            // Load next 64 bytes of Original Data
            data_vec = _mm512_stream_load_si512( *sPnt + curPos ) ;
          __builtin_prefetch ( *sPnt + curPos + 64, 0, 3 ) ;
            sPnt ++ ;

            // Add incoming data to MSB of parity, then update parities using Parallel Multiplier
            data_vec = _mm512_xor_si512( data_vec, parity [ 0 ] ) ;

            // Compute the first half of the tap values for palindromic code generator
            temp [ 0 ] = _mm512_gf2p8affine_epi64_epi8 ( data_vec, taps [ 0 ], 0 ) ;
            temp [ 1 ] = _mm512_gf2p8affine_epi64_epi8 ( data_vec, taps [ 1 ], 0 ) ;
            temp [ 2 ] = _mm512_gf2p8affine_epi64_epi8 ( data_vec, taps [ 2 ], 0 ) ;
            temp [ 3 ] = _mm512_gf2p8affine_epi64_epi8 ( data_vec, taps [ 3 ], 0 ) ;
            temp [ 4 ] = _mm512_gf2p8affine_epi64_epi8 ( data_vec, taps [ 4 ], 0 ) ;
            temp [ 5 ] = _mm512_gf2p8affine_epi64_epi8 ( data_vec, taps [ 5 ], 0 ) ;
            temp [ 6 ] = _mm512_gf2p8affine_epi64_epi8 ( data_vec, taps [ 6 ], 0 ) ;
            temp [ 7 ] = _mm512_gf2p8affine_epi64_epi8 ( data_vec, taps [ 7 ], 0 ) ;
            temp [ 8 ] = _mm512_gf2p8affine_epi64_epi8 ( data_vec, taps [ 8 ], 0 ) ;
            temp [ 9 ] = _mm512_gf2p8affine_epi64_epi8 ( data_vec, taps [ 9 ], 0 ) ;
            temp [ 10 ] = _mm512_gf2p8affine_epi64_epi8 ( data_vec, taps [ 10 ], 0 ) ;
            temp [ 11 ] = _mm512_gf2p8affine_epi64_epi8 ( data_vec, taps [ 11 ], 0 ) ;

            // Now cycle the LFSR
            parity [ 0 ] = _mm512_xor_si512 ( parity [ 1 ], temp [ 0 ] ) ;
            parity [ 1 ] = _mm512_xor_si512 ( parity [ 2 ], temp [ 1 ] ) ;
            parity [ 2 ] = _mm512_xor_si512 ( parity [ 3 ], temp [ 2 ] ) ;
            parity [ 3 ] = _mm512_xor_si512 ( parity [ 4 ], temp [ 3 ] ) ;
            parity [ 4 ] = _mm512_xor_si512 ( parity [ 5 ], temp [ 4 ] ) ;
            parity [ 5 ] = _mm512_xor_si512 ( parity [ 6 ], temp [ 5 ] ) ;
            parity [ 6 ] = _mm512_xor_si512 ( parity [ 7 ], temp [ 6 ] ) ;
            parity [ 7 ] = _mm512_xor_si512 ( parity [ 8 ], temp [ 7 ] ) ;
            parity [ 8 ] = _mm512_xor_si512 ( parity [ 9 ], temp [ 8 ] ) ;
            parity [ 9 ] = _mm512_xor_si512 ( parity [ 10 ], temp [ 9 ] ) ;
            parity [ 10 ] = _mm512_xor_si512 ( parity [ 11 ], temp [ 10 ] ) ;
            parity [ 11 ] = _mm512_xor_si512 ( parity [ 12 ], temp [ 11 ] ) ;
            parity [ 12 ] = _mm512_xor_si512 ( parity [ 13 ], temp [ 10 ] ) ;
            parity [ 13 ] = _mm512_xor_si512 ( parity [ 14 ], temp [ 9 ] ) ;
            parity [ 14 ] = _mm512_xor_si512 ( parity [ 15 ], temp [ 8 ] ) ;
            parity [ 15 ] = _mm512_xor_si512 ( parity [ 16 ], temp [ 7 ] ) ;
            parity [ 16 ] = _mm512_xor_si512 ( parity [ 17 ], temp [ 6 ] ) ;
            parity [ 17 ] = _mm512_xor_si512 ( parity [ 18 ], temp [ 5 ] ) ;
            parity [ 18 ] = _mm512_xor_si512 ( parity [ 19 ], temp [ 4 ] ) ;
            parity [ 19 ] = _mm512_xor_si512 ( parity [ 20 ], temp [ 3 ] ) ;
            parity [ 20 ] = _mm512_xor_si512 ( parity [ 21 ], temp [ 2 ] ) ;
            parity [ 21 ] = _mm512_xor_si512 ( parity [ 22 ], temp [ 1 ] ) ;
            parity [ 22 ] = _mm512_xor_si512 ( parity [ 23 ], temp [ 0 ] ) ;
            parity [ 23 ] = data_vec ;
        }

        if ( decoder == 0 )
        {
            // Store parity back to memory
            _mm512_stream_si512( (&data [ k+0 ][curPos]), parity [ 0 ] ) ;
            _mm512_stream_si512( (&data [ k+1 ][curPos]), parity [ 1 ] ) ;
            _mm512_stream_si512( (&data [ k+2 ][curPos]), parity [ 2 ] ) ;
            _mm512_stream_si512( (&data [ k+3 ][curPos]), parity [ 3 ] ) ;
            _mm512_stream_si512( (&data [ k+4 ][curPos]), parity [ 4 ] ) ;
            _mm512_stream_si512( (&data [ k+5 ][curPos]), parity [ 5 ] ) ;
            _mm512_stream_si512( (&data [ k+6 ][curPos]), parity [ 6 ] ) ;
            _mm512_stream_si512( (&data [ k+7 ][curPos]), parity [ 7 ] ) ;
            _mm512_stream_si512( (&data [ k+8 ][curPos]), parity [ 8 ] ) ;
            _mm512_stream_si512( (&data [ k+9 ][curPos]), parity [ 9 ] ) ;
            _mm512_stream_si512( (&data [ k+10 ][curPos]), parity [ 10 ] ) ;
            _mm512_stream_si512( (&data [ k+11 ][curPos]), parity [ 11 ] ) ;
            _mm512_stream_si512( (&data [ k+12 ][curPos]), parity [ 12 ] ) ;
            _mm512_stream_si512( (&data [ k+13 ][curPos]), parity [ 13 ] ) ;
            _mm512_stream_si512( (&data [ k+14 ][curPos]), parity [ 14 ] ) ;
            _mm512_stream_si512( (&data [ k+15 ][curPos]), parity [ 15 ] ) ;
            _mm512_stream_si512( (&data [ k+16 ][curPos]), parity [ 16 ] ) ;
            _mm512_stream_si512( (&data [ k+17 ][curPos]), parity [ 17 ] ) ;
            _mm512_stream_si512( (&data [ k+18 ][curPos]), parity [ 18 ] ) ;
            _mm512_stream_si512( (&data [ k+19 ][curPos]), parity [ 19 ] ) ;
            _mm512_stream_si512( (&data [ k+20 ][curPos]), parity [ 20 ] ) ;
            _mm512_stream_si512( (&data [ k+21 ][curPos]), parity [ 21 ] ) ;
            _mm512_stream_si512( (&data [ k+22 ][curPos]), parity [ 22 ] ) ;
            _mm512_stream_si512( (&data [ k+23 ][curPos]), parity [ 23 ] ) ;
        }
        else
        {
            // Verify Syndromes are zero
            data_vec = _mm512_or_si512 ( parity [ 0 ], parity [ 1 ] ) ;
            data_vec = _mm512_ternarylogic_epi32 ( data_vec, parity [ 2 ], parity [ 3 ], 0xFE ) ;
            data_vec = _mm512_ternarylogic_epi32 ( data_vec, parity [ 4 ], parity [ 5 ], 0xFE ) ;
            data_vec = _mm512_ternarylogic_epi32 ( data_vec, parity [ 6 ], parity [ 7 ], 0xFE ) ;
            data_vec = _mm512_ternarylogic_epi32 ( data_vec, parity [ 8 ], parity [ 9 ], 0xFE ) ;
            data_vec = _mm512_ternarylogic_epi32 ( data_vec, parity [ 10 ], parity [ 11 ], 0xFE ) ;
            data_vec = _mm512_ternarylogic_epi32 ( data_vec, parity [ 12 ], parity [ 13 ], 0xFE ) ;
            data_vec = _mm512_ternarylogic_epi32 ( data_vec, parity [ 14 ], parity [ 15 ], 0xFE ) ;
            data_vec = _mm512_ternarylogic_epi32 ( data_vec, parity [ 16 ], parity [ 17 ], 0xFE ) ;
            data_vec = _mm512_ternarylogic_epi32 ( data_vec, parity [ 18 ], parity [ 19 ], 0xFE ) ;
            data_vec = _mm512_ternarylogic_epi32 ( data_vec, parity [ 20 ], parity [ 21 ], 0xFE ) ;
            data_vec = _mm512_ternarylogic_epi32 ( data_vec, parity [ 22 ], parity [ 23 ], 0xFE ) ;
            __mmask64 mask = _mm512_test_epi8_mask ( data_vec, data_vec ) ;

            // If syndromes are not zero then process error
            while ( !_ktestz_mask64_u8 ( mask, mask ) ) 
            {
                PCErrCnt++;
                unsigned char Sr[24], S[24], errLocs[24], errMags[24] ;
                unsigned char Lambda[24+1], B[24+1], T[24+1], Omega[24+1] ;
                unsigned char PTemp[64] ;
                uint64_t pos = _tzcnt_u64 ( mask ) ;

                _mm512_storeu_si512 ( PTemp, parity[0] ) ;
                Sr[0] = PTemp[pos];
                _mm512_storeu_si512 ( PTemp, parity[1] ) ;
                Sr[1] = PTemp[pos];
                _mm512_storeu_si512 ( PTemp, parity[2] ) ;
                Sr[2] = PTemp[pos];
                _mm512_storeu_si512 ( PTemp, parity[3] ) ;
                Sr[3] = PTemp[pos];
                _mm512_storeu_si512 ( PTemp, parity[4] ) ;
                Sr[4] = PTemp[pos];
                _mm512_storeu_si512 ( PTemp, parity[5] ) ;
                Sr[5] = PTemp[pos];
                _mm512_storeu_si512 ( PTemp, parity[6] ) ;
                Sr[6] = PTemp[pos];
                _mm512_storeu_si512 ( PTemp, parity[7] ) ;
                Sr[7] = PTemp[pos];
                _mm512_storeu_si512 ( PTemp, parity[8] ) ;
                Sr[8] = PTemp[pos];
                _mm512_storeu_si512 ( PTemp, parity[9] ) ;
                Sr[9] = PTemp[pos];
                _mm512_storeu_si512 ( PTemp, parity[10] ) ;
                Sr[10] = PTemp[pos];
                _mm512_storeu_si512 ( PTemp, parity[11] ) ;
                Sr[11] = PTemp[pos];
                _mm512_storeu_si512 ( PTemp, parity[12] ) ;
                Sr[12] = PTemp[pos];
                _mm512_storeu_si512 ( PTemp, parity[13] ) ;
                Sr[13] = PTemp[pos];
                _mm512_storeu_si512 ( PTemp, parity[14] ) ;
                Sr[14] = PTemp[pos];
                _mm512_storeu_si512 ( PTemp, parity[15] ) ;
                Sr[15] = PTemp[pos];
                _mm512_storeu_si512 ( PTemp, parity[16] ) ;
                Sr[16] = PTemp[pos];
                _mm512_storeu_si512 ( PTemp, parity[17] ) ;
                Sr[17] = PTemp[pos];
                _mm512_storeu_si512 ( PTemp, parity[18] ) ;
                Sr[18] = PTemp[pos];
                _mm512_storeu_si512 ( PTemp, parity[19] ) ;
                Sr[19] = PTemp[pos];
                _mm512_storeu_si512 ( PTemp, parity[20] ) ;
                Sr[20] = PTemp[pos];
                _mm512_storeu_si512 ( PTemp, parity[21] ) ;
                Sr[21] = PTemp[pos];
                _mm512_storeu_si512 ( PTemp, parity[22] ) ;
                Sr[22] = PTemp[pos];
                _mm512_storeu_si512 ( PTemp, parity[23] ) ;
                Sr[23] = PTemp[pos];
                rs_decode_palindromic_GFNI(Sr, S, 24, k, errLocs, errMags,
                             (int)pos+curPos, Lambda, B, T, Omega, data) ;
                mask = _blsr_u64(mask);
            }
        }
    }
}

// Parallel LFSRD_SR Sequencer for P = 25 Codewords
void ParallelLFSRSequencer25_GFNI(int len, int k, unsigned char **data, int decoder)
{
    int curSym, curPos ;                           // Loop counters
    __m512i parity [ 25 ], taps [ 12 ] ;             // Parity registers
    __m512i data_vec, temp [ 12 ] ;
    unsigned char **sPnt ;                         // Data lookup pointers

    // Initialize the taps to the LFSR
   taps [ 0 ] = _mm512_broadcast_i32x2(*( __m128i * ) &PCAffTab[0x56]);
   taps [ 1 ] = _mm512_broadcast_i32x2(*( __m128i * ) &PCAffTab[0xd5]);
   taps [ 2 ] = _mm512_broadcast_i32x2(*( __m128i * ) &PCAffTab[0x3b]);
   taps [ 3 ] = _mm512_broadcast_i32x2(*( __m128i * ) &PCAffTab[0xcf]);
   taps [ 4 ] = _mm512_broadcast_i32x2(*( __m128i * ) &PCAffTab[0x67]);
   taps [ 5 ] = _mm512_broadcast_i32x2(*( __m128i * ) &PCAffTab[0xa2]);
   taps [ 6 ] = _mm512_broadcast_i32x2(*( __m128i * ) &PCAffTab[0xc4]);
   taps [ 7 ] = _mm512_broadcast_i32x2(*( __m128i * ) &PCAffTab[0x2b]);
   taps [ 8 ] = _mm512_broadcast_i32x2(*( __m128i * ) &PCAffTab[0xbf]);
   taps [ 9 ] = _mm512_broadcast_i32x2(*( __m128i * ) &PCAffTab[0x18]);
   taps [ 10 ] = _mm512_broadcast_i32x2(*( __m128i * ) &PCAffTab[0x2f]);
   taps [ 11 ] = _mm512_broadcast_i32x2(*( __m128i * ) &PCAffTab[0x91]);

    // Adjust K if we are decoding
    k += decoder ? 25 : 0 ;

    // Loop through each 64 byte codeword
    for ( curPos = 0 ; curPos < len ; curPos += 64 )
    {
        sPnt = data ;
        parity [ 0 ] = _mm512_setzero_si512();
        parity [ 1 ] = _mm512_setzero_si512();
        parity [ 2 ] = _mm512_setzero_si512();
        parity [ 3 ] = _mm512_setzero_si512();
        parity [ 4 ] = _mm512_setzero_si512();
        parity [ 5 ] = _mm512_setzero_si512();
        parity [ 6 ] = _mm512_setzero_si512();
        parity [ 7 ] = _mm512_setzero_si512();
        parity [ 8 ] = _mm512_setzero_si512();
        parity [ 9 ] = _mm512_setzero_si512();
        parity [ 10 ] = _mm512_setzero_si512();
        parity [ 11 ] = _mm512_setzero_si512();
        parity [ 12 ] = _mm512_setzero_si512();
        parity [ 13 ] = _mm512_setzero_si512();
        parity [ 14 ] = _mm512_setzero_si512();
        parity [ 15 ] = _mm512_setzero_si512();
        parity [ 16 ] = _mm512_setzero_si512();
        parity [ 17 ] = _mm512_setzero_si512();
        parity [ 18 ] = _mm512_setzero_si512();
        parity [ 19 ] = _mm512_setzero_si512();
        parity [ 20 ] = _mm512_setzero_si512();
        parity [ 21 ] = _mm512_setzero_si512();
        parity [ 22 ] = _mm512_setzero_si512();
        parity [ 23 ] = _mm512_setzero_si512();
        parity [ 24 ] = _mm512_setzero_si512();

        // Loop through all the 0..k symbols
        for ( curSym = 0 ; curSym < k ; curSym ++ )
        {
            // Load next 64 bytes of Original Data
            data_vec = _mm512_stream_load_si512( *sPnt + curPos ) ;
          __builtin_prefetch ( *sPnt + curPos + 64, 0, 3 ) ;
            sPnt ++ ;

            // Add incoming data to MSB of parity, then update parities using Parallel Multiplier
            data_vec = _mm512_xor_si512( data_vec, parity [ 0 ] ) ;

            // Compute the first half of the tap values for palindromic code generator
            temp [ 0 ] = _mm512_gf2p8affine_epi64_epi8 ( data_vec, taps [ 0 ], 0 ) ;
            temp [ 1 ] = _mm512_gf2p8affine_epi64_epi8 ( data_vec, taps [ 1 ], 0 ) ;
            temp [ 2 ] = _mm512_gf2p8affine_epi64_epi8 ( data_vec, taps [ 2 ], 0 ) ;
            temp [ 3 ] = _mm512_gf2p8affine_epi64_epi8 ( data_vec, taps [ 3 ], 0 ) ;
            temp [ 4 ] = _mm512_gf2p8affine_epi64_epi8 ( data_vec, taps [ 4 ], 0 ) ;
            temp [ 5 ] = _mm512_gf2p8affine_epi64_epi8 ( data_vec, taps [ 5 ], 0 ) ;
            temp [ 6 ] = _mm512_gf2p8affine_epi64_epi8 ( data_vec, taps [ 6 ], 0 ) ;
            temp [ 7 ] = _mm512_gf2p8affine_epi64_epi8 ( data_vec, taps [ 7 ], 0 ) ;
            temp [ 8 ] = _mm512_gf2p8affine_epi64_epi8 ( data_vec, taps [ 8 ], 0 ) ;
            temp [ 9 ] = _mm512_gf2p8affine_epi64_epi8 ( data_vec, taps [ 9 ], 0 ) ;
            temp [ 10 ] = _mm512_gf2p8affine_epi64_epi8 ( data_vec, taps [ 10 ], 0 ) ;
            temp [ 11 ] = _mm512_gf2p8affine_epi64_epi8 ( data_vec, taps [ 11 ], 0 ) ;

            // Now cycle the LFSR
            parity [ 0 ] = _mm512_xor_si512 ( parity [ 1 ], temp [ 0 ] ) ;
            parity [ 1 ] = _mm512_xor_si512 ( parity [ 2 ], temp [ 1 ] ) ;
            parity [ 2 ] = _mm512_xor_si512 ( parity [ 3 ], temp [ 2 ] ) ;
            parity [ 3 ] = _mm512_xor_si512 ( parity [ 4 ], temp [ 3 ] ) ;
            parity [ 4 ] = _mm512_xor_si512 ( parity [ 5 ], temp [ 4 ] ) ;
            parity [ 5 ] = _mm512_xor_si512 ( parity [ 6 ], temp [ 5 ] ) ;
            parity [ 6 ] = _mm512_xor_si512 ( parity [ 7 ], temp [ 6 ] ) ;
            parity [ 7 ] = _mm512_xor_si512 ( parity [ 8 ], temp [ 7 ] ) ;
            parity [ 8 ] = _mm512_xor_si512 ( parity [ 9 ], temp [ 8 ] ) ;
            parity [ 9 ] = _mm512_xor_si512 ( parity [ 10 ], temp [ 9 ] ) ;
            parity [ 10 ] = _mm512_xor_si512 ( parity [ 11 ], temp [ 10 ] ) ;
            parity [ 11 ] = _mm512_xor_si512 ( parity [ 12 ], temp [ 11 ] ) ;
            parity [ 12 ] = _mm512_xor_si512 ( parity [ 13 ], temp [ 11 ] ) ;
            parity [ 13 ] = _mm512_xor_si512 ( parity [ 14 ], temp [ 10 ] ) ;
            parity [ 14 ] = _mm512_xor_si512 ( parity [ 15 ], temp [ 9 ] ) ;
            parity [ 15 ] = _mm512_xor_si512 ( parity [ 16 ], temp [ 8 ] ) ;
            parity [ 16 ] = _mm512_xor_si512 ( parity [ 17 ], temp [ 7 ] ) ;
            parity [ 17 ] = _mm512_xor_si512 ( parity [ 18 ], temp [ 6 ] ) ;
            parity [ 18 ] = _mm512_xor_si512 ( parity [ 19 ], temp [ 5 ] ) ;
            parity [ 19 ] = _mm512_xor_si512 ( parity [ 20 ], temp [ 4 ] ) ;
            parity [ 20 ] = _mm512_xor_si512 ( parity [ 21 ], temp [ 3 ] ) ;
            parity [ 21 ] = _mm512_xor_si512 ( parity [ 22 ], temp [ 2 ] ) ;
            parity [ 22 ] = _mm512_xor_si512 ( parity [ 23 ], temp [ 1 ] ) ;
            parity [ 23 ] = _mm512_xor_si512 ( parity [ 24 ], temp [ 0 ] ) ;
            parity [ 24 ] = data_vec ;
        }

        if ( decoder == 0 )
        {
            // Store parity back to memory
            _mm512_stream_si512( (&data [ k+0 ][curPos]), parity [ 0 ] ) ;
            _mm512_stream_si512( (&data [ k+1 ][curPos]), parity [ 1 ] ) ;
            _mm512_stream_si512( (&data [ k+2 ][curPos]), parity [ 2 ] ) ;
            _mm512_stream_si512( (&data [ k+3 ][curPos]), parity [ 3 ] ) ;
            _mm512_stream_si512( (&data [ k+4 ][curPos]), parity [ 4 ] ) ;
            _mm512_stream_si512( (&data [ k+5 ][curPos]), parity [ 5 ] ) ;
            _mm512_stream_si512( (&data [ k+6 ][curPos]), parity [ 6 ] ) ;
            _mm512_stream_si512( (&data [ k+7 ][curPos]), parity [ 7 ] ) ;
            _mm512_stream_si512( (&data [ k+8 ][curPos]), parity [ 8 ] ) ;
            _mm512_stream_si512( (&data [ k+9 ][curPos]), parity [ 9 ] ) ;
            _mm512_stream_si512( (&data [ k+10 ][curPos]), parity [ 10 ] ) ;
            _mm512_stream_si512( (&data [ k+11 ][curPos]), parity [ 11 ] ) ;
            _mm512_stream_si512( (&data [ k+12 ][curPos]), parity [ 12 ] ) ;
            _mm512_stream_si512( (&data [ k+13 ][curPos]), parity [ 13 ] ) ;
            _mm512_stream_si512( (&data [ k+14 ][curPos]), parity [ 14 ] ) ;
            _mm512_stream_si512( (&data [ k+15 ][curPos]), parity [ 15 ] ) ;
            _mm512_stream_si512( (&data [ k+16 ][curPos]), parity [ 16 ] ) ;
            _mm512_stream_si512( (&data [ k+17 ][curPos]), parity [ 17 ] ) ;
            _mm512_stream_si512( (&data [ k+18 ][curPos]), parity [ 18 ] ) ;
            _mm512_stream_si512( (&data [ k+19 ][curPos]), parity [ 19 ] ) ;
            _mm512_stream_si512( (&data [ k+20 ][curPos]), parity [ 20 ] ) ;
            _mm512_stream_si512( (&data [ k+21 ][curPos]), parity [ 21 ] ) ;
            _mm512_stream_si512( (&data [ k+22 ][curPos]), parity [ 22 ] ) ;
            _mm512_stream_si512( (&data [ k+23 ][curPos]), parity [ 23 ] ) ;
            _mm512_stream_si512( (&data [ k+24 ][curPos]), parity [ 24 ] ) ;
        }
        else
        {
            // Verify Syndromes are zero
            data_vec = _mm512_or_si512 ( parity [ 0 ], parity [ 1 ] ) ;
            data_vec = _mm512_ternarylogic_epi32 ( data_vec, parity [ 2 ], parity [ 3 ], 0xFE ) ;
            data_vec = _mm512_ternarylogic_epi32 ( data_vec, parity [ 4 ], parity [ 5 ], 0xFE ) ;
            data_vec = _mm512_ternarylogic_epi32 ( data_vec, parity [ 6 ], parity [ 7 ], 0xFE ) ;
            data_vec = _mm512_ternarylogic_epi32 ( data_vec, parity [ 8 ], parity [ 9 ], 0xFE ) ;
            data_vec = _mm512_ternarylogic_epi32 ( data_vec, parity [ 10 ], parity [ 11 ], 0xFE ) ;
            data_vec = _mm512_ternarylogic_epi32 ( data_vec, parity [ 12 ], parity [ 13 ], 0xFE ) ;
            data_vec = _mm512_ternarylogic_epi32 ( data_vec, parity [ 14 ], parity [ 15 ], 0xFE ) ;
            data_vec = _mm512_ternarylogic_epi32 ( data_vec, parity [ 16 ], parity [ 17 ], 0xFE ) ;
            data_vec = _mm512_ternarylogic_epi32 ( data_vec, parity [ 18 ], parity [ 19 ], 0xFE ) ;
            data_vec = _mm512_ternarylogic_epi32 ( data_vec, parity [ 20 ], parity [ 21 ], 0xFE ) ;
            data_vec = _mm512_ternarylogic_epi32 ( data_vec, parity [ 22 ], parity [ 23 ], 0xFE ) ;
            data_vec = _mm512_ternarylogic_epi32 ( data_vec, parity [ 24 ], parity [ 24 ], 0xFE ) ;
            __mmask64 mask = _mm512_test_epi8_mask ( data_vec, data_vec ) ;

            // If syndromes are not zero then process error
            while ( !_ktestz_mask64_u8 ( mask, mask ) ) 
            {
                PCErrCnt++;
                unsigned char Sr[25], S[25], errLocs[25], errMags[25] ;
                unsigned char Lambda[25+1], B[25+1], T[25+1], Omega[25+1] ;
                unsigned char PTemp[64] ;
                uint64_t pos = _tzcnt_u64 ( mask ) ;

                _mm512_storeu_si512 ( PTemp, parity[0] ) ;
                Sr[0] = PTemp[pos];
                _mm512_storeu_si512 ( PTemp, parity[1] ) ;
                Sr[1] = PTemp[pos];
                _mm512_storeu_si512 ( PTemp, parity[2] ) ;
                Sr[2] = PTemp[pos];
                _mm512_storeu_si512 ( PTemp, parity[3] ) ;
                Sr[3] = PTemp[pos];
                _mm512_storeu_si512 ( PTemp, parity[4] ) ;
                Sr[4] = PTemp[pos];
                _mm512_storeu_si512 ( PTemp, parity[5] ) ;
                Sr[5] = PTemp[pos];
                _mm512_storeu_si512 ( PTemp, parity[6] ) ;
                Sr[6] = PTemp[pos];
                _mm512_storeu_si512 ( PTemp, parity[7] ) ;
                Sr[7] = PTemp[pos];
                _mm512_storeu_si512 ( PTemp, parity[8] ) ;
                Sr[8] = PTemp[pos];
                _mm512_storeu_si512 ( PTemp, parity[9] ) ;
                Sr[9] = PTemp[pos];
                _mm512_storeu_si512 ( PTemp, parity[10] ) ;
                Sr[10] = PTemp[pos];
                _mm512_storeu_si512 ( PTemp, parity[11] ) ;
                Sr[11] = PTemp[pos];
                _mm512_storeu_si512 ( PTemp, parity[12] ) ;
                Sr[12] = PTemp[pos];
                _mm512_storeu_si512 ( PTemp, parity[13] ) ;
                Sr[13] = PTemp[pos];
                _mm512_storeu_si512 ( PTemp, parity[14] ) ;
                Sr[14] = PTemp[pos];
                _mm512_storeu_si512 ( PTemp, parity[15] ) ;
                Sr[15] = PTemp[pos];
                _mm512_storeu_si512 ( PTemp, parity[16] ) ;
                Sr[16] = PTemp[pos];
                _mm512_storeu_si512 ( PTemp, parity[17] ) ;
                Sr[17] = PTemp[pos];
                _mm512_storeu_si512 ( PTemp, parity[18] ) ;
                Sr[18] = PTemp[pos];
                _mm512_storeu_si512 ( PTemp, parity[19] ) ;
                Sr[19] = PTemp[pos];
                _mm512_storeu_si512 ( PTemp, parity[20] ) ;
                Sr[20] = PTemp[pos];
                _mm512_storeu_si512 ( PTemp, parity[21] ) ;
                Sr[21] = PTemp[pos];
                _mm512_storeu_si512 ( PTemp, parity[22] ) ;
                Sr[22] = PTemp[pos];
                _mm512_storeu_si512 ( PTemp, parity[23] ) ;
                Sr[23] = PTemp[pos];
                _mm512_storeu_si512 ( PTemp, parity[24] ) ;
                Sr[24] = PTemp[pos];
                rs_decode_palindromic_GFNI(Sr, S, 25, k, errLocs, errMags,
                             (int)pos+curPos, Lambda, B, T, Omega, data) ;
                mask = _blsr_u64(mask);
            }
        }
    }
}

// Parallel LFSRD_SR Sequencer for P = 26 Codewords
void ParallelLFSRSequencer26_GFNI(int len, int k, unsigned char **data, int decoder)
{
    int curSym, curPos ;                           // Loop counters
    __m512i parity [ 26 ], taps [ 13 ] ;             // Parity registers
    __m512i data_vec, temp [ 13 ] ;
    unsigned char **sPnt ;                         // Data lookup pointers

    // Initialize the taps to the LFSR
   taps [ 0 ] = _mm512_broadcast_i32x2(*( __m128i * ) &PCAffTab[0x27]);
   taps [ 1 ] = _mm512_broadcast_i32x2(*( __m128i * ) &PCAffTab[0x11]);
   taps [ 2 ] = _mm512_broadcast_i32x2(*( __m128i * ) &PCAffTab[0x1b]);
   taps [ 3 ] = _mm512_broadcast_i32x2(*( __m128i * ) &PCAffTab[0xa6]);
   taps [ 4 ] = _mm512_broadcast_i32x2(*( __m128i * ) &PCAffTab[0xfc]);
   taps [ 5 ] = _mm512_broadcast_i32x2(*( __m128i * ) &PCAffTab[0x68]);
   taps [ 6 ] = _mm512_broadcast_i32x2(*( __m128i * ) &PCAffTab[0xb7]);
   taps [ 7 ] = _mm512_broadcast_i32x2(*( __m128i * ) &PCAffTab[0x7b]);
   taps [ 8 ] = _mm512_broadcast_i32x2(*( __m128i * ) &PCAffTab[0xe6]);
   taps [ 9 ] = _mm512_broadcast_i32x2(*( __m128i * ) &PCAffTab[0x1e]);
   taps [ 10 ] = _mm512_broadcast_i32x2(*( __m128i * ) &PCAffTab[0x22]);
   taps [ 11 ] = _mm512_broadcast_i32x2(*( __m128i * ) &PCAffTab[0x6d]);
   taps [ 12 ] = _mm512_broadcast_i32x2(*( __m128i * ) &PCAffTab[0x2e]);

    // Adjust K if we are decoding
    k += decoder ? 26 : 0 ;

    // Loop through each 64 byte codeword
    for ( curPos = 0 ; curPos < len ; curPos += 64 )
    {
        sPnt = data ;
        parity [ 0 ] = _mm512_setzero_si512();
        parity [ 1 ] = _mm512_setzero_si512();
        parity [ 2 ] = _mm512_setzero_si512();
        parity [ 3 ] = _mm512_setzero_si512();
        parity [ 4 ] = _mm512_setzero_si512();
        parity [ 5 ] = _mm512_setzero_si512();
        parity [ 6 ] = _mm512_setzero_si512();
        parity [ 7 ] = _mm512_setzero_si512();
        parity [ 8 ] = _mm512_setzero_si512();
        parity [ 9 ] = _mm512_setzero_si512();
        parity [ 10 ] = _mm512_setzero_si512();
        parity [ 11 ] = _mm512_setzero_si512();
        parity [ 12 ] = _mm512_setzero_si512();
        parity [ 13 ] = _mm512_setzero_si512();
        parity [ 14 ] = _mm512_setzero_si512();
        parity [ 15 ] = _mm512_setzero_si512();
        parity [ 16 ] = _mm512_setzero_si512();
        parity [ 17 ] = _mm512_setzero_si512();
        parity [ 18 ] = _mm512_setzero_si512();
        parity [ 19 ] = _mm512_setzero_si512();
        parity [ 20 ] = _mm512_setzero_si512();
        parity [ 21 ] = _mm512_setzero_si512();
        parity [ 22 ] = _mm512_setzero_si512();
        parity [ 23 ] = _mm512_setzero_si512();
        parity [ 24 ] = _mm512_setzero_si512();
        parity [ 25 ] = _mm512_setzero_si512();

        // Loop through all the 0..k symbols
        for ( curSym = 0 ; curSym < k ; curSym ++ )
        {
            // Load next 64 bytes of Original Data
            data_vec = _mm512_stream_load_si512( *sPnt + curPos ) ;
          __builtin_prefetch ( *sPnt + curPos + 64, 0, 3 ) ;
            sPnt ++ ;

            // Add incoming data to MSB of parity, then update parities using Parallel Multiplier
            data_vec = _mm512_xor_si512( data_vec, parity [ 0 ] ) ;

            // Compute the first half of the tap values for palindromic code generator
            temp [ 0 ] = _mm512_gf2p8affine_epi64_epi8 ( data_vec, taps [ 0 ], 0 ) ;
            temp [ 1 ] = _mm512_gf2p8affine_epi64_epi8 ( data_vec, taps [ 1 ], 0 ) ;
            temp [ 2 ] = _mm512_gf2p8affine_epi64_epi8 ( data_vec, taps [ 2 ], 0 ) ;
            temp [ 3 ] = _mm512_gf2p8affine_epi64_epi8 ( data_vec, taps [ 3 ], 0 ) ;
            temp [ 4 ] = _mm512_gf2p8affine_epi64_epi8 ( data_vec, taps [ 4 ], 0 ) ;
            temp [ 5 ] = _mm512_gf2p8affine_epi64_epi8 ( data_vec, taps [ 5 ], 0 ) ;
            temp [ 6 ] = _mm512_gf2p8affine_epi64_epi8 ( data_vec, taps [ 6 ], 0 ) ;
            temp [ 7 ] = _mm512_gf2p8affine_epi64_epi8 ( data_vec, taps [ 7 ], 0 ) ;
            temp [ 8 ] = _mm512_gf2p8affine_epi64_epi8 ( data_vec, taps [ 8 ], 0 ) ;
            temp [ 9 ] = _mm512_gf2p8affine_epi64_epi8 ( data_vec, taps [ 9 ], 0 ) ;
            temp [ 10 ] = _mm512_gf2p8affine_epi64_epi8 ( data_vec, taps [ 10 ], 0 ) ;
            temp [ 11 ] = _mm512_gf2p8affine_epi64_epi8 ( data_vec, taps [ 11 ], 0 ) ;
            temp [ 12 ] = _mm512_gf2p8affine_epi64_epi8 ( data_vec, taps [ 12 ], 0 ) ;

            // Now cycle the LFSR
            parity [ 0 ] = _mm512_xor_si512 ( parity [ 1 ], temp [ 0 ] ) ;
            parity [ 1 ] = _mm512_xor_si512 ( parity [ 2 ], temp [ 1 ] ) ;
            parity [ 2 ] = _mm512_xor_si512 ( parity [ 3 ], temp [ 2 ] ) ;
            parity [ 3 ] = _mm512_xor_si512 ( parity [ 4 ], temp [ 3 ] ) ;
            parity [ 4 ] = _mm512_xor_si512 ( parity [ 5 ], temp [ 4 ] ) ;
            parity [ 5 ] = _mm512_xor_si512 ( parity [ 6 ], temp [ 5 ] ) ;
            parity [ 6 ] = _mm512_xor_si512 ( parity [ 7 ], temp [ 6 ] ) ;
            parity [ 7 ] = _mm512_xor_si512 ( parity [ 8 ], temp [ 7 ] ) ;
            parity [ 8 ] = _mm512_xor_si512 ( parity [ 9 ], temp [ 8 ] ) ;
            parity [ 9 ] = _mm512_xor_si512 ( parity [ 10 ], temp [ 9 ] ) ;
            parity [ 10 ] = _mm512_xor_si512 ( parity [ 11 ], temp [ 10 ] ) ;
            parity [ 11 ] = _mm512_xor_si512 ( parity [ 12 ], temp [ 11 ] ) ;
            parity [ 12 ] = _mm512_xor_si512 ( parity [ 13 ], temp [ 12 ] ) ;
            parity [ 13 ] = _mm512_xor_si512 ( parity [ 14 ], temp [ 11 ] ) ;
            parity [ 14 ] = _mm512_xor_si512 ( parity [ 15 ], temp [ 10 ] ) ;
            parity [ 15 ] = _mm512_xor_si512 ( parity [ 16 ], temp [ 9 ] ) ;
            parity [ 16 ] = _mm512_xor_si512 ( parity [ 17 ], temp [ 8 ] ) ;
            parity [ 17 ] = _mm512_xor_si512 ( parity [ 18 ], temp [ 7 ] ) ;
            parity [ 18 ] = _mm512_xor_si512 ( parity [ 19 ], temp [ 6 ] ) ;
            parity [ 19 ] = _mm512_xor_si512 ( parity [ 20 ], temp [ 5 ] ) ;
            parity [ 20 ] = _mm512_xor_si512 ( parity [ 21 ], temp [ 4 ] ) ;
            parity [ 21 ] = _mm512_xor_si512 ( parity [ 22 ], temp [ 3 ] ) ;
            parity [ 22 ] = _mm512_xor_si512 ( parity [ 23 ], temp [ 2 ] ) ;
            parity [ 23 ] = _mm512_xor_si512 ( parity [ 24 ], temp [ 1 ] ) ;
            parity [ 24 ] = _mm512_xor_si512 ( parity [ 25 ], temp [ 0 ] ) ;
            parity [ 25 ] = data_vec ;
        }

        if ( decoder == 0 )
        {
            // Store parity back to memory
            _mm512_stream_si512( (&data [ k+0 ][curPos]), parity [ 0 ] ) ;
            _mm512_stream_si512( (&data [ k+1 ][curPos]), parity [ 1 ] ) ;
            _mm512_stream_si512( (&data [ k+2 ][curPos]), parity [ 2 ] ) ;
            _mm512_stream_si512( (&data [ k+3 ][curPos]), parity [ 3 ] ) ;
            _mm512_stream_si512( (&data [ k+4 ][curPos]), parity [ 4 ] ) ;
            _mm512_stream_si512( (&data [ k+5 ][curPos]), parity [ 5 ] ) ;
            _mm512_stream_si512( (&data [ k+6 ][curPos]), parity [ 6 ] ) ;
            _mm512_stream_si512( (&data [ k+7 ][curPos]), parity [ 7 ] ) ;
            _mm512_stream_si512( (&data [ k+8 ][curPos]), parity [ 8 ] ) ;
            _mm512_stream_si512( (&data [ k+9 ][curPos]), parity [ 9 ] ) ;
            _mm512_stream_si512( (&data [ k+10 ][curPos]), parity [ 10 ] ) ;
            _mm512_stream_si512( (&data [ k+11 ][curPos]), parity [ 11 ] ) ;
            _mm512_stream_si512( (&data [ k+12 ][curPos]), parity [ 12 ] ) ;
            _mm512_stream_si512( (&data [ k+13 ][curPos]), parity [ 13 ] ) ;
            _mm512_stream_si512( (&data [ k+14 ][curPos]), parity [ 14 ] ) ;
            _mm512_stream_si512( (&data [ k+15 ][curPos]), parity [ 15 ] ) ;
            _mm512_stream_si512( (&data [ k+16 ][curPos]), parity [ 16 ] ) ;
            _mm512_stream_si512( (&data [ k+17 ][curPos]), parity [ 17 ] ) ;
            _mm512_stream_si512( (&data [ k+18 ][curPos]), parity [ 18 ] ) ;
            _mm512_stream_si512( (&data [ k+19 ][curPos]), parity [ 19 ] ) ;
            _mm512_stream_si512( (&data [ k+20 ][curPos]), parity [ 20 ] ) ;
            _mm512_stream_si512( (&data [ k+21 ][curPos]), parity [ 21 ] ) ;
            _mm512_stream_si512( (&data [ k+22 ][curPos]), parity [ 22 ] ) ;
            _mm512_stream_si512( (&data [ k+23 ][curPos]), parity [ 23 ] ) ;
            _mm512_stream_si512( (&data [ k+24 ][curPos]), parity [ 24 ] ) ;
            _mm512_stream_si512( (&data [ k+25 ][curPos]), parity [ 25 ] ) ;
        }
        else
        {
            // Verify Syndromes are zero
            data_vec = _mm512_or_si512 ( parity [ 0 ], parity [ 1 ] ) ;
            data_vec = _mm512_ternarylogic_epi32 ( data_vec, parity [ 2 ], parity [ 3 ], 0xFE ) ;
            data_vec = _mm512_ternarylogic_epi32 ( data_vec, parity [ 4 ], parity [ 5 ], 0xFE ) ;
            data_vec = _mm512_ternarylogic_epi32 ( data_vec, parity [ 6 ], parity [ 7 ], 0xFE ) ;
            data_vec = _mm512_ternarylogic_epi32 ( data_vec, parity [ 8 ], parity [ 9 ], 0xFE ) ;
            data_vec = _mm512_ternarylogic_epi32 ( data_vec, parity [ 10 ], parity [ 11 ], 0xFE ) ;
            data_vec = _mm512_ternarylogic_epi32 ( data_vec, parity [ 12 ], parity [ 13 ], 0xFE ) ;
            data_vec = _mm512_ternarylogic_epi32 ( data_vec, parity [ 14 ], parity [ 15 ], 0xFE ) ;
            data_vec = _mm512_ternarylogic_epi32 ( data_vec, parity [ 16 ], parity [ 17 ], 0xFE ) ;
            data_vec = _mm512_ternarylogic_epi32 ( data_vec, parity [ 18 ], parity [ 19 ], 0xFE ) ;
            data_vec = _mm512_ternarylogic_epi32 ( data_vec, parity [ 20 ], parity [ 21 ], 0xFE ) ;
            data_vec = _mm512_ternarylogic_epi32 ( data_vec, parity [ 22 ], parity [ 23 ], 0xFE ) ;
            data_vec = _mm512_ternarylogic_epi32 ( data_vec, parity [ 24 ], parity [ 25 ], 0xFE ) ;
            __mmask64 mask = _mm512_test_epi8_mask ( data_vec, data_vec ) ;

            // If syndromes are not zero then process error
            while ( !_ktestz_mask64_u8 ( mask, mask ) ) 
            {
                PCErrCnt++;
                unsigned char Sr[26], S[26], errLocs[26], errMags[26] ;
                unsigned char Lambda[26+1], B[26+1], T[26+1], Omega[26+1] ;
                unsigned char PTemp[64] ;
                uint64_t pos = _tzcnt_u64 ( mask ) ;

                _mm512_storeu_si512 ( PTemp, parity[0] ) ;
                Sr[0] = PTemp[pos];
                _mm512_storeu_si512 ( PTemp, parity[1] ) ;
                Sr[1] = PTemp[pos];
                _mm512_storeu_si512 ( PTemp, parity[2] ) ;
                Sr[2] = PTemp[pos];
                _mm512_storeu_si512 ( PTemp, parity[3] ) ;
                Sr[3] = PTemp[pos];
                _mm512_storeu_si512 ( PTemp, parity[4] ) ;
                Sr[4] = PTemp[pos];
                _mm512_storeu_si512 ( PTemp, parity[5] ) ;
                Sr[5] = PTemp[pos];
                _mm512_storeu_si512 ( PTemp, parity[6] ) ;
                Sr[6] = PTemp[pos];
                _mm512_storeu_si512 ( PTemp, parity[7] ) ;
                Sr[7] = PTemp[pos];
                _mm512_storeu_si512 ( PTemp, parity[8] ) ;
                Sr[8] = PTemp[pos];
                _mm512_storeu_si512 ( PTemp, parity[9] ) ;
                Sr[9] = PTemp[pos];
                _mm512_storeu_si512 ( PTemp, parity[10] ) ;
                Sr[10] = PTemp[pos];
                _mm512_storeu_si512 ( PTemp, parity[11] ) ;
                Sr[11] = PTemp[pos];
                _mm512_storeu_si512 ( PTemp, parity[12] ) ;
                Sr[12] = PTemp[pos];
                _mm512_storeu_si512 ( PTemp, parity[13] ) ;
                Sr[13] = PTemp[pos];
                _mm512_storeu_si512 ( PTemp, parity[14] ) ;
                Sr[14] = PTemp[pos];
                _mm512_storeu_si512 ( PTemp, parity[15] ) ;
                Sr[15] = PTemp[pos];
                _mm512_storeu_si512 ( PTemp, parity[16] ) ;
                Sr[16] = PTemp[pos];
                _mm512_storeu_si512 ( PTemp, parity[17] ) ;
                Sr[17] = PTemp[pos];
                _mm512_storeu_si512 ( PTemp, parity[18] ) ;
                Sr[18] = PTemp[pos];
                _mm512_storeu_si512 ( PTemp, parity[19] ) ;
                Sr[19] = PTemp[pos];
                _mm512_storeu_si512 ( PTemp, parity[20] ) ;
                Sr[20] = PTemp[pos];
                _mm512_storeu_si512 ( PTemp, parity[21] ) ;
                Sr[21] = PTemp[pos];
                _mm512_storeu_si512 ( PTemp, parity[22] ) ;
                Sr[22] = PTemp[pos];
                _mm512_storeu_si512 ( PTemp, parity[23] ) ;
                Sr[23] = PTemp[pos];
                _mm512_storeu_si512 ( PTemp, parity[24] ) ;
                Sr[24] = PTemp[pos];
                _mm512_storeu_si512 ( PTemp, parity[25] ) ;
                Sr[25] = PTemp[pos];
                rs_decode_palindromic_GFNI(Sr, S, 26, k, errLocs, errMags,
                             (int)pos+curPos, Lambda, B, T, Omega, data) ;
                mask = _blsr_u64(mask);
            }
        }
    }
}

// Parallel LFSRD_SR Sequencer for P = 27 Codewords
void ParallelLFSRSequencer27_GFNI(int len, int k, unsigned char **data, int decoder)
{
    int curSym, curPos ;                           // Loop counters
    __m512i parity [ 27 ], taps [ 13 ] ;             // Parity registers
    __m512i data_vec, temp [ 13 ] ;
    unsigned char **sPnt ;                         // Data lookup pointers

    // Initialize the taps to the LFSR
   taps [ 0 ] = _mm512_broadcast_i32x2(*( __m128i * ) &PCAffTab[0x61]);
   taps [ 1 ] = _mm512_broadcast_i32x2(*( __m128i * ) &PCAffTab[0x70]);
   taps [ 2 ] = _mm512_broadcast_i32x2(*( __m128i * ) &PCAffTab[0x12]);
   taps [ 3 ] = _mm512_broadcast_i32x2(*( __m128i * ) &PCAffTab[0x1f]);
   taps [ 4 ] = _mm512_broadcast_i32x2(*( __m128i * ) &PCAffTab[0xbf]);
   taps [ 5 ] = _mm512_broadcast_i32x2(*( __m128i * ) &PCAffTab[0x7]);
   taps [ 6 ] = _mm512_broadcast_i32x2(*( __m128i * ) &PCAffTab[0xe1]);
   taps [ 7 ] = _mm512_broadcast_i32x2(*( __m128i * ) &PCAffTab[0x96]);
   taps [ 8 ] = _mm512_broadcast_i32x2(*( __m128i * ) &PCAffTab[0x29]);
   taps [ 9 ] = _mm512_broadcast_i32x2(*( __m128i * ) &PCAffTab[0x68]);
   taps [ 10 ] = _mm512_broadcast_i32x2(*( __m128i * ) &PCAffTab[0x62]);
   taps [ 11 ] = _mm512_broadcast_i32x2(*( __m128i * ) &PCAffTab[0x7]);
   taps [ 12 ] = _mm512_broadcast_i32x2(*( __m128i * ) &PCAffTab[0x5c]);

    // Adjust K if we are decoding
    k += decoder ? 27 : 0 ;

    // Loop through each 64 byte codeword
    for ( curPos = 0 ; curPos < len ; curPos += 64 )
    {
        sPnt = data ;
        parity [ 0 ] = _mm512_setzero_si512();
        parity [ 1 ] = _mm512_setzero_si512();
        parity [ 2 ] = _mm512_setzero_si512();
        parity [ 3 ] = _mm512_setzero_si512();
        parity [ 4 ] = _mm512_setzero_si512();
        parity [ 5 ] = _mm512_setzero_si512();
        parity [ 6 ] = _mm512_setzero_si512();
        parity [ 7 ] = _mm512_setzero_si512();
        parity [ 8 ] = _mm512_setzero_si512();
        parity [ 9 ] = _mm512_setzero_si512();
        parity [ 10 ] = _mm512_setzero_si512();
        parity [ 11 ] = _mm512_setzero_si512();
        parity [ 12 ] = _mm512_setzero_si512();
        parity [ 13 ] = _mm512_setzero_si512();
        parity [ 14 ] = _mm512_setzero_si512();
        parity [ 15 ] = _mm512_setzero_si512();
        parity [ 16 ] = _mm512_setzero_si512();
        parity [ 17 ] = _mm512_setzero_si512();
        parity [ 18 ] = _mm512_setzero_si512();
        parity [ 19 ] = _mm512_setzero_si512();
        parity [ 20 ] = _mm512_setzero_si512();
        parity [ 21 ] = _mm512_setzero_si512();
        parity [ 22 ] = _mm512_setzero_si512();
        parity [ 23 ] = _mm512_setzero_si512();
        parity [ 24 ] = _mm512_setzero_si512();
        parity [ 25 ] = _mm512_setzero_si512();
        parity [ 26 ] = _mm512_setzero_si512();

        // Loop through all the 0..k symbols
        for ( curSym = 0 ; curSym < k ; curSym ++ )
        {
            // Load next 64 bytes of Original Data
            data_vec = _mm512_stream_load_si512( *sPnt + curPos ) ;
          __builtin_prefetch ( *sPnt + curPos + 64, 0, 3 ) ;
            sPnt ++ ;

            // Add incoming data to MSB of parity, then update parities using Parallel Multiplier
            data_vec = _mm512_xor_si512( data_vec, parity [ 0 ] ) ;

            // Compute the first half of the tap values for palindromic code generator
            temp [ 0 ] = _mm512_gf2p8affine_epi64_epi8 ( data_vec, taps [ 0 ], 0 ) ;
            temp [ 1 ] = _mm512_gf2p8affine_epi64_epi8 ( data_vec, taps [ 1 ], 0 ) ;
            temp [ 2 ] = _mm512_gf2p8affine_epi64_epi8 ( data_vec, taps [ 2 ], 0 ) ;
            temp [ 3 ] = _mm512_gf2p8affine_epi64_epi8 ( data_vec, taps [ 3 ], 0 ) ;
            temp [ 4 ] = _mm512_gf2p8affine_epi64_epi8 ( data_vec, taps [ 4 ], 0 ) ;
            temp [ 5 ] = _mm512_gf2p8affine_epi64_epi8 ( data_vec, taps [ 5 ], 0 ) ;
            temp [ 6 ] = _mm512_gf2p8affine_epi64_epi8 ( data_vec, taps [ 6 ], 0 ) ;
            temp [ 7 ] = _mm512_gf2p8affine_epi64_epi8 ( data_vec, taps [ 7 ], 0 ) ;
            temp [ 8 ] = _mm512_gf2p8affine_epi64_epi8 ( data_vec, taps [ 8 ], 0 ) ;
            temp [ 9 ] = _mm512_gf2p8affine_epi64_epi8 ( data_vec, taps [ 9 ], 0 ) ;
            temp [ 10 ] = _mm512_gf2p8affine_epi64_epi8 ( data_vec, taps [ 10 ], 0 ) ;
            temp [ 11 ] = _mm512_gf2p8affine_epi64_epi8 ( data_vec, taps [ 11 ], 0 ) ;
            temp [ 12 ] = _mm512_gf2p8affine_epi64_epi8 ( data_vec, taps [ 12 ], 0 ) ;

            // Now cycle the LFSR
            parity [ 0 ] = _mm512_xor_si512 ( parity [ 1 ], temp [ 0 ] ) ;
            parity [ 1 ] = _mm512_xor_si512 ( parity [ 2 ], temp [ 1 ] ) ;
            parity [ 2 ] = _mm512_xor_si512 ( parity [ 3 ], temp [ 2 ] ) ;
            parity [ 3 ] = _mm512_xor_si512 ( parity [ 4 ], temp [ 3 ] ) ;
            parity [ 4 ] = _mm512_xor_si512 ( parity [ 5 ], temp [ 4 ] ) ;
            parity [ 5 ] = _mm512_xor_si512 ( parity [ 6 ], temp [ 5 ] ) ;
            parity [ 6 ] = _mm512_xor_si512 ( parity [ 7 ], temp [ 6 ] ) ;
            parity [ 7 ] = _mm512_xor_si512 ( parity [ 8 ], temp [ 7 ] ) ;
            parity [ 8 ] = _mm512_xor_si512 ( parity [ 9 ], temp [ 8 ] ) ;
            parity [ 9 ] = _mm512_xor_si512 ( parity [ 10 ], temp [ 9 ] ) ;
            parity [ 10 ] = _mm512_xor_si512 ( parity [ 11 ], temp [ 10 ] ) ;
            parity [ 11 ] = _mm512_xor_si512 ( parity [ 12 ], temp [ 11 ] ) ;
            parity [ 12 ] = _mm512_xor_si512 ( parity [ 13 ], temp [ 12 ] ) ;
            parity [ 13 ] = _mm512_xor_si512 ( parity [ 14 ], temp [ 12 ] ) ;
            parity [ 14 ] = _mm512_xor_si512 ( parity [ 15 ], temp [ 11 ] ) ;
            parity [ 15 ] = _mm512_xor_si512 ( parity [ 16 ], temp [ 10 ] ) ;
            parity [ 16 ] = _mm512_xor_si512 ( parity [ 17 ], temp [ 9 ] ) ;
            parity [ 17 ] = _mm512_xor_si512 ( parity [ 18 ], temp [ 8 ] ) ;
            parity [ 18 ] = _mm512_xor_si512 ( parity [ 19 ], temp [ 7 ] ) ;
            parity [ 19 ] = _mm512_xor_si512 ( parity [ 20 ], temp [ 6 ] ) ;
            parity [ 20 ] = _mm512_xor_si512 ( parity [ 21 ], temp [ 5 ] ) ;
            parity [ 21 ] = _mm512_xor_si512 ( parity [ 22 ], temp [ 4 ] ) ;
            parity [ 22 ] = _mm512_xor_si512 ( parity [ 23 ], temp [ 3 ] ) ;
            parity [ 23 ] = _mm512_xor_si512 ( parity [ 24 ], temp [ 2 ] ) ;
            parity [ 24 ] = _mm512_xor_si512 ( parity [ 25 ], temp [ 1 ] ) ;
            parity [ 25 ] = _mm512_xor_si512 ( parity [ 26 ], temp [ 0 ] ) ;
            parity [ 26 ] = data_vec ;
        }

        if ( decoder == 0 )
        {
            // Store parity back to memory
            _mm512_stream_si512( (&data [ k+0 ][curPos]), parity [ 0 ] ) ;
            _mm512_stream_si512( (&data [ k+1 ][curPos]), parity [ 1 ] ) ;
            _mm512_stream_si512( (&data [ k+2 ][curPos]), parity [ 2 ] ) ;
            _mm512_stream_si512( (&data [ k+3 ][curPos]), parity [ 3 ] ) ;
            _mm512_stream_si512( (&data [ k+4 ][curPos]), parity [ 4 ] ) ;
            _mm512_stream_si512( (&data [ k+5 ][curPos]), parity [ 5 ] ) ;
            _mm512_stream_si512( (&data [ k+6 ][curPos]), parity [ 6 ] ) ;
            _mm512_stream_si512( (&data [ k+7 ][curPos]), parity [ 7 ] ) ;
            _mm512_stream_si512( (&data [ k+8 ][curPos]), parity [ 8 ] ) ;
            _mm512_stream_si512( (&data [ k+9 ][curPos]), parity [ 9 ] ) ;
            _mm512_stream_si512( (&data [ k+10 ][curPos]), parity [ 10 ] ) ;
            _mm512_stream_si512( (&data [ k+11 ][curPos]), parity [ 11 ] ) ;
            _mm512_stream_si512( (&data [ k+12 ][curPos]), parity [ 12 ] ) ;
            _mm512_stream_si512( (&data [ k+13 ][curPos]), parity [ 13 ] ) ;
            _mm512_stream_si512( (&data [ k+14 ][curPos]), parity [ 14 ] ) ;
            _mm512_stream_si512( (&data [ k+15 ][curPos]), parity [ 15 ] ) ;
            _mm512_stream_si512( (&data [ k+16 ][curPos]), parity [ 16 ] ) ;
            _mm512_stream_si512( (&data [ k+17 ][curPos]), parity [ 17 ] ) ;
            _mm512_stream_si512( (&data [ k+18 ][curPos]), parity [ 18 ] ) ;
            _mm512_stream_si512( (&data [ k+19 ][curPos]), parity [ 19 ] ) ;
            _mm512_stream_si512( (&data [ k+20 ][curPos]), parity [ 20 ] ) ;
            _mm512_stream_si512( (&data [ k+21 ][curPos]), parity [ 21 ] ) ;
            _mm512_stream_si512( (&data [ k+22 ][curPos]), parity [ 22 ] ) ;
            _mm512_stream_si512( (&data [ k+23 ][curPos]), parity [ 23 ] ) ;
            _mm512_stream_si512( (&data [ k+24 ][curPos]), parity [ 24 ] ) ;
            _mm512_stream_si512( (&data [ k+25 ][curPos]), parity [ 25 ] ) ;
            _mm512_stream_si512( (&data [ k+26 ][curPos]), parity [ 26 ] ) ;
        }
        else
        {
            // Verify Syndromes are zero
            data_vec = _mm512_or_si512 ( parity [ 0 ], parity [ 1 ] ) ;
            data_vec = _mm512_ternarylogic_epi32 ( data_vec, parity [ 2 ], parity [ 3 ], 0xFE ) ;
            data_vec = _mm512_ternarylogic_epi32 ( data_vec, parity [ 4 ], parity [ 5 ], 0xFE ) ;
            data_vec = _mm512_ternarylogic_epi32 ( data_vec, parity [ 6 ], parity [ 7 ], 0xFE ) ;
            data_vec = _mm512_ternarylogic_epi32 ( data_vec, parity [ 8 ], parity [ 9 ], 0xFE ) ;
            data_vec = _mm512_ternarylogic_epi32 ( data_vec, parity [ 10 ], parity [ 11 ], 0xFE ) ;
            data_vec = _mm512_ternarylogic_epi32 ( data_vec, parity [ 12 ], parity [ 13 ], 0xFE ) ;
            data_vec = _mm512_ternarylogic_epi32 ( data_vec, parity [ 14 ], parity [ 15 ], 0xFE ) ;
            data_vec = _mm512_ternarylogic_epi32 ( data_vec, parity [ 16 ], parity [ 17 ], 0xFE ) ;
            data_vec = _mm512_ternarylogic_epi32 ( data_vec, parity [ 18 ], parity [ 19 ], 0xFE ) ;
            data_vec = _mm512_ternarylogic_epi32 ( data_vec, parity [ 20 ], parity [ 21 ], 0xFE ) ;
            data_vec = _mm512_ternarylogic_epi32 ( data_vec, parity [ 22 ], parity [ 23 ], 0xFE ) ;
            data_vec = _mm512_ternarylogic_epi32 ( data_vec, parity [ 24 ], parity [ 25 ], 0xFE ) ;
            data_vec = _mm512_ternarylogic_epi32 ( data_vec, parity [ 26 ], parity [ 26 ], 0xFE ) ;
            __mmask64 mask = _mm512_test_epi8_mask ( data_vec, data_vec ) ;

            // If syndromes are not zero then process error
            while ( !_ktestz_mask64_u8 ( mask, mask ) ) 
            {
                PCErrCnt++;
                unsigned char Sr[27], S[27], errLocs[27], errMags[27] ;
                unsigned char Lambda[27+1], B[27+1], T[27+1], Omega[27+1] ;
                unsigned char PTemp[64] ;
                uint64_t pos = _tzcnt_u64 ( mask ) ;

                _mm512_storeu_si512 ( PTemp, parity[0] ) ;
                Sr[0] = PTemp[pos];
                _mm512_storeu_si512 ( PTemp, parity[1] ) ;
                Sr[1] = PTemp[pos];
                _mm512_storeu_si512 ( PTemp, parity[2] ) ;
                Sr[2] = PTemp[pos];
                _mm512_storeu_si512 ( PTemp, parity[3] ) ;
                Sr[3] = PTemp[pos];
                _mm512_storeu_si512 ( PTemp, parity[4] ) ;
                Sr[4] = PTemp[pos];
                _mm512_storeu_si512 ( PTemp, parity[5] ) ;
                Sr[5] = PTemp[pos];
                _mm512_storeu_si512 ( PTemp, parity[6] ) ;
                Sr[6] = PTemp[pos];
                _mm512_storeu_si512 ( PTemp, parity[7] ) ;
                Sr[7] = PTemp[pos];
                _mm512_storeu_si512 ( PTemp, parity[8] ) ;
                Sr[8] = PTemp[pos];
                _mm512_storeu_si512 ( PTemp, parity[9] ) ;
                Sr[9] = PTemp[pos];
                _mm512_storeu_si512 ( PTemp, parity[10] ) ;
                Sr[10] = PTemp[pos];
                _mm512_storeu_si512 ( PTemp, parity[11] ) ;
                Sr[11] = PTemp[pos];
                _mm512_storeu_si512 ( PTemp, parity[12] ) ;
                Sr[12] = PTemp[pos];
                _mm512_storeu_si512 ( PTemp, parity[13] ) ;
                Sr[13] = PTemp[pos];
                _mm512_storeu_si512 ( PTemp, parity[14] ) ;
                Sr[14] = PTemp[pos];
                _mm512_storeu_si512 ( PTemp, parity[15] ) ;
                Sr[15] = PTemp[pos];
                _mm512_storeu_si512 ( PTemp, parity[16] ) ;
                Sr[16] = PTemp[pos];
                _mm512_storeu_si512 ( PTemp, parity[17] ) ;
                Sr[17] = PTemp[pos];
                _mm512_storeu_si512 ( PTemp, parity[18] ) ;
                Sr[18] = PTemp[pos];
                _mm512_storeu_si512 ( PTemp, parity[19] ) ;
                Sr[19] = PTemp[pos];
                _mm512_storeu_si512 ( PTemp, parity[20] ) ;
                Sr[20] = PTemp[pos];
                _mm512_storeu_si512 ( PTemp, parity[21] ) ;
                Sr[21] = PTemp[pos];
                _mm512_storeu_si512 ( PTemp, parity[22] ) ;
                Sr[22] = PTemp[pos];
                _mm512_storeu_si512 ( PTemp, parity[23] ) ;
                Sr[23] = PTemp[pos];
                _mm512_storeu_si512 ( PTemp, parity[24] ) ;
                Sr[24] = PTemp[pos];
                _mm512_storeu_si512 ( PTemp, parity[25] ) ;
                Sr[25] = PTemp[pos];
                _mm512_storeu_si512 ( PTemp, parity[26] ) ;
                Sr[26] = PTemp[pos];
                rs_decode_palindromic_GFNI(Sr, S, 27, k, errLocs, errMags,
                             (int)pos+curPos, Lambda, B, T, Omega, data) ;
                mask = _blsr_u64(mask);
            }
        }
    }
}

// Parallel LFSRD_SR Sequencer for P = 28 Codewords
void ParallelLFSRSequencer28_GFNI(int len, int k, unsigned char **data, int decoder)
{
    int curSym, curPos ;                           // Loop counters
    __m512i parity [ 28 ], taps [ 14 ] ;             // Parity registers
    __m512i data_vec, temp [ 14 ] ;
    unsigned char **sPnt ;                         // Data lookup pointers

    // Initialize the taps to the LFSR
   taps [ 0 ] = _mm512_broadcast_i32x2(*( __m128i * ) &PCAffTab[0xc]);
   taps [ 1 ] = _mm512_broadcast_i32x2(*( __m128i * ) &PCAffTab[0xc8]);
   taps [ 2 ] = _mm512_broadcast_i32x2(*( __m128i * ) &PCAffTab[0x9d]);
   taps [ 3 ] = _mm512_broadcast_i32x2(*( __m128i * ) &PCAffTab[0x5]);
   taps [ 4 ] = _mm512_broadcast_i32x2(*( __m128i * ) &PCAffTab[0x30]);
   taps [ 5 ] = _mm512_broadcast_i32x2(*( __m128i * ) &PCAffTab[0x92]);
   taps [ 6 ] = _mm512_broadcast_i32x2(*( __m128i * ) &PCAffTab[0x15]);
   taps [ 7 ] = _mm512_broadcast_i32x2(*( __m128i * ) &PCAffTab[0x65]);
   taps [ 8 ] = _mm512_broadcast_i32x2(*( __m128i * ) &PCAffTab[0xf8]);
   taps [ 9 ] = _mm512_broadcast_i32x2(*( __m128i * ) &PCAffTab[0xa0]);
   taps [ 10 ] = _mm512_broadcast_i32x2(*( __m128i * ) &PCAffTab[0xf1]);
   taps [ 11 ] = _mm512_broadcast_i32x2(*( __m128i * ) &PCAffTab[0x2c]);
   taps [ 12 ] = _mm512_broadcast_i32x2(*( __m128i * ) &PCAffTab[0xd5]);
   taps [ 13 ] = _mm512_broadcast_i32x2(*( __m128i * ) &PCAffTab[0xb6]);

    // Adjust K if we are decoding
    k += decoder ? 28 : 0 ;

    // Loop through each 64 byte codeword
    for ( curPos = 0 ; curPos < len ; curPos += 64 )
    {
        sPnt = data ;
        parity [ 0 ] = _mm512_setzero_si512();
        parity [ 1 ] = _mm512_setzero_si512();
        parity [ 2 ] = _mm512_setzero_si512();
        parity [ 3 ] = _mm512_setzero_si512();
        parity [ 4 ] = _mm512_setzero_si512();
        parity [ 5 ] = _mm512_setzero_si512();
        parity [ 6 ] = _mm512_setzero_si512();
        parity [ 7 ] = _mm512_setzero_si512();
        parity [ 8 ] = _mm512_setzero_si512();
        parity [ 9 ] = _mm512_setzero_si512();
        parity [ 10 ] = _mm512_setzero_si512();
        parity [ 11 ] = _mm512_setzero_si512();
        parity [ 12 ] = _mm512_setzero_si512();
        parity [ 13 ] = _mm512_setzero_si512();
        parity [ 14 ] = _mm512_setzero_si512();
        parity [ 15 ] = _mm512_setzero_si512();
        parity [ 16 ] = _mm512_setzero_si512();
        parity [ 17 ] = _mm512_setzero_si512();
        parity [ 18 ] = _mm512_setzero_si512();
        parity [ 19 ] = _mm512_setzero_si512();
        parity [ 20 ] = _mm512_setzero_si512();
        parity [ 21 ] = _mm512_setzero_si512();
        parity [ 22 ] = _mm512_setzero_si512();
        parity [ 23 ] = _mm512_setzero_si512();
        parity [ 24 ] = _mm512_setzero_si512();
        parity [ 25 ] = _mm512_setzero_si512();
        parity [ 26 ] = _mm512_setzero_si512();
        parity [ 27 ] = _mm512_setzero_si512();

        // Loop through all the 0..k symbols
        for ( curSym = 0 ; curSym < k ; curSym ++ )
        {
            // Load next 64 bytes of Original Data
            data_vec = _mm512_stream_load_si512( *sPnt + curPos ) ;
          __builtin_prefetch ( *sPnt + curPos + 64, 0, 3 ) ;
            sPnt ++ ;

            // Add incoming data to MSB of parity, then update parities using Parallel Multiplier
            data_vec = _mm512_xor_si512( data_vec, parity [ 0 ] ) ;

            // Compute the first half of the tap values for palindromic code generator
            temp [ 0 ] = _mm512_gf2p8affine_epi64_epi8 ( data_vec, taps [ 0 ], 0 ) ;
            temp [ 1 ] = _mm512_gf2p8affine_epi64_epi8 ( data_vec, taps [ 1 ], 0 ) ;
            temp [ 2 ] = _mm512_gf2p8affine_epi64_epi8 ( data_vec, taps [ 2 ], 0 ) ;
            temp [ 3 ] = _mm512_gf2p8affine_epi64_epi8 ( data_vec, taps [ 3 ], 0 ) ;
            temp [ 4 ] = _mm512_gf2p8affine_epi64_epi8 ( data_vec, taps [ 4 ], 0 ) ;
            temp [ 5 ] = _mm512_gf2p8affine_epi64_epi8 ( data_vec, taps [ 5 ], 0 ) ;
            temp [ 6 ] = _mm512_gf2p8affine_epi64_epi8 ( data_vec, taps [ 6 ], 0 ) ;
            temp [ 7 ] = _mm512_gf2p8affine_epi64_epi8 ( data_vec, taps [ 7 ], 0 ) ;
            temp [ 8 ] = _mm512_gf2p8affine_epi64_epi8 ( data_vec, taps [ 8 ], 0 ) ;
            temp [ 9 ] = _mm512_gf2p8affine_epi64_epi8 ( data_vec, taps [ 9 ], 0 ) ;
            temp [ 10 ] = _mm512_gf2p8affine_epi64_epi8 ( data_vec, taps [ 10 ], 0 ) ;
            temp [ 11 ] = _mm512_gf2p8affine_epi64_epi8 ( data_vec, taps [ 11 ], 0 ) ;
            temp [ 12 ] = _mm512_gf2p8affine_epi64_epi8 ( data_vec, taps [ 12 ], 0 ) ;
            temp [ 13 ] = _mm512_gf2p8affine_epi64_epi8 ( data_vec, taps [ 13 ], 0 ) ;

            // Now cycle the LFSR
            parity [ 0 ] = _mm512_xor_si512 ( parity [ 1 ], temp [ 0 ] ) ;
            parity [ 1 ] = _mm512_xor_si512 ( parity [ 2 ], temp [ 1 ] ) ;
            parity [ 2 ] = _mm512_xor_si512 ( parity [ 3 ], temp [ 2 ] ) ;
            parity [ 3 ] = _mm512_xor_si512 ( parity [ 4 ], temp [ 3 ] ) ;
            parity [ 4 ] = _mm512_xor_si512 ( parity [ 5 ], temp [ 4 ] ) ;
            parity [ 5 ] = _mm512_xor_si512 ( parity [ 6 ], temp [ 5 ] ) ;
            parity [ 6 ] = _mm512_xor_si512 ( parity [ 7 ], temp [ 6 ] ) ;
            parity [ 7 ] = _mm512_xor_si512 ( parity [ 8 ], temp [ 7 ] ) ;
            parity [ 8 ] = _mm512_xor_si512 ( parity [ 9 ], temp [ 8 ] ) ;
            parity [ 9 ] = _mm512_xor_si512 ( parity [ 10 ], temp [ 9 ] ) ;
            parity [ 10 ] = _mm512_xor_si512 ( parity [ 11 ], temp [ 10 ] ) ;
            parity [ 11 ] = _mm512_xor_si512 ( parity [ 12 ], temp [ 11 ] ) ;
            parity [ 12 ] = _mm512_xor_si512 ( parity [ 13 ], temp [ 12 ] ) ;
            parity [ 13 ] = _mm512_xor_si512 ( parity [ 14 ], temp [ 13 ] ) ;
            parity [ 14 ] = _mm512_xor_si512 ( parity [ 15 ], temp [ 12 ] ) ;
            parity [ 15 ] = _mm512_xor_si512 ( parity [ 16 ], temp [ 11 ] ) ;
            parity [ 16 ] = _mm512_xor_si512 ( parity [ 17 ], temp [ 10 ] ) ;
            parity [ 17 ] = _mm512_xor_si512 ( parity [ 18 ], temp [ 9 ] ) ;
            parity [ 18 ] = _mm512_xor_si512 ( parity [ 19 ], temp [ 8 ] ) ;
            parity [ 19 ] = _mm512_xor_si512 ( parity [ 20 ], temp [ 7 ] ) ;
            parity [ 20 ] = _mm512_xor_si512 ( parity [ 21 ], temp [ 6 ] ) ;
            parity [ 21 ] = _mm512_xor_si512 ( parity [ 22 ], temp [ 5 ] ) ;
            parity [ 22 ] = _mm512_xor_si512 ( parity [ 23 ], temp [ 4 ] ) ;
            parity [ 23 ] = _mm512_xor_si512 ( parity [ 24 ], temp [ 3 ] ) ;
            parity [ 24 ] = _mm512_xor_si512 ( parity [ 25 ], temp [ 2 ] ) ;
            parity [ 25 ] = _mm512_xor_si512 ( parity [ 26 ], temp [ 1 ] ) ;
            parity [ 26 ] = _mm512_xor_si512 ( parity [ 27 ], temp [ 0 ] ) ;
            parity [ 27 ] = data_vec ;
        }

        if ( decoder == 0 )
        {
            // Store parity back to memory
            _mm512_stream_si512( (&data [ k+0 ][curPos]), parity [ 0 ] ) ;
            _mm512_stream_si512( (&data [ k+1 ][curPos]), parity [ 1 ] ) ;
            _mm512_stream_si512( (&data [ k+2 ][curPos]), parity [ 2 ] ) ;
            _mm512_stream_si512( (&data [ k+3 ][curPos]), parity [ 3 ] ) ;
            _mm512_stream_si512( (&data [ k+4 ][curPos]), parity [ 4 ] ) ;
            _mm512_stream_si512( (&data [ k+5 ][curPos]), parity [ 5 ] ) ;
            _mm512_stream_si512( (&data [ k+6 ][curPos]), parity [ 6 ] ) ;
            _mm512_stream_si512( (&data [ k+7 ][curPos]), parity [ 7 ] ) ;
            _mm512_stream_si512( (&data [ k+8 ][curPos]), parity [ 8 ] ) ;
            _mm512_stream_si512( (&data [ k+9 ][curPos]), parity [ 9 ] ) ;
            _mm512_stream_si512( (&data [ k+10 ][curPos]), parity [ 10 ] ) ;
            _mm512_stream_si512( (&data [ k+11 ][curPos]), parity [ 11 ] ) ;
            _mm512_stream_si512( (&data [ k+12 ][curPos]), parity [ 12 ] ) ;
            _mm512_stream_si512( (&data [ k+13 ][curPos]), parity [ 13 ] ) ;
            _mm512_stream_si512( (&data [ k+14 ][curPos]), parity [ 14 ] ) ;
            _mm512_stream_si512( (&data [ k+15 ][curPos]), parity [ 15 ] ) ;
            _mm512_stream_si512( (&data [ k+16 ][curPos]), parity [ 16 ] ) ;
            _mm512_stream_si512( (&data [ k+17 ][curPos]), parity [ 17 ] ) ;
            _mm512_stream_si512( (&data [ k+18 ][curPos]), parity [ 18 ] ) ;
            _mm512_stream_si512( (&data [ k+19 ][curPos]), parity [ 19 ] ) ;
            _mm512_stream_si512( (&data [ k+20 ][curPos]), parity [ 20 ] ) ;
            _mm512_stream_si512( (&data [ k+21 ][curPos]), parity [ 21 ] ) ;
            _mm512_stream_si512( (&data [ k+22 ][curPos]), parity [ 22 ] ) ;
            _mm512_stream_si512( (&data [ k+23 ][curPos]), parity [ 23 ] ) ;
            _mm512_stream_si512( (&data [ k+24 ][curPos]), parity [ 24 ] ) ;
            _mm512_stream_si512( (&data [ k+25 ][curPos]), parity [ 25 ] ) ;
            _mm512_stream_si512( (&data [ k+26 ][curPos]), parity [ 26 ] ) ;
            _mm512_stream_si512( (&data [ k+27 ][curPos]), parity [ 27 ] ) ;
        }
        else
        {
            // Verify Syndromes are zero
            data_vec = _mm512_or_si512 ( parity [ 0 ], parity [ 1 ] ) ;
            data_vec = _mm512_ternarylogic_epi32 ( data_vec, parity [ 2 ], parity [ 3 ], 0xFE ) ;
            data_vec = _mm512_ternarylogic_epi32 ( data_vec, parity [ 4 ], parity [ 5 ], 0xFE ) ;
            data_vec = _mm512_ternarylogic_epi32 ( data_vec, parity [ 6 ], parity [ 7 ], 0xFE ) ;
            data_vec = _mm512_ternarylogic_epi32 ( data_vec, parity [ 8 ], parity [ 9 ], 0xFE ) ;
            data_vec = _mm512_ternarylogic_epi32 ( data_vec, parity [ 10 ], parity [ 11 ], 0xFE ) ;
            data_vec = _mm512_ternarylogic_epi32 ( data_vec, parity [ 12 ], parity [ 13 ], 0xFE ) ;
            data_vec = _mm512_ternarylogic_epi32 ( data_vec, parity [ 14 ], parity [ 15 ], 0xFE ) ;
            data_vec = _mm512_ternarylogic_epi32 ( data_vec, parity [ 16 ], parity [ 17 ], 0xFE ) ;
            data_vec = _mm512_ternarylogic_epi32 ( data_vec, parity [ 18 ], parity [ 19 ], 0xFE ) ;
            data_vec = _mm512_ternarylogic_epi32 ( data_vec, parity [ 20 ], parity [ 21 ], 0xFE ) ;
            data_vec = _mm512_ternarylogic_epi32 ( data_vec, parity [ 22 ], parity [ 23 ], 0xFE ) ;
            data_vec = _mm512_ternarylogic_epi32 ( data_vec, parity [ 24 ], parity [ 25 ], 0xFE ) ;
            data_vec = _mm512_ternarylogic_epi32 ( data_vec, parity [ 26 ], parity [ 27 ], 0xFE ) ;
            __mmask64 mask = _mm512_test_epi8_mask ( data_vec, data_vec ) ;

            // If syndromes are not zero then process error
            while ( !_ktestz_mask64_u8 ( mask, mask ) ) 
            {
                PCErrCnt++;
                unsigned char Sr[28], S[28], errLocs[28], errMags[28] ;
                unsigned char Lambda[28+1], B[28+1], T[28+1], Omega[28+1] ;
                unsigned char PTemp[64] ;
                uint64_t pos = _tzcnt_u64 ( mask ) ;

                _mm512_storeu_si512 ( PTemp, parity[0] ) ;
                Sr[0] = PTemp[pos];
                _mm512_storeu_si512 ( PTemp, parity[1] ) ;
                Sr[1] = PTemp[pos];
                _mm512_storeu_si512 ( PTemp, parity[2] ) ;
                Sr[2] = PTemp[pos];
                _mm512_storeu_si512 ( PTemp, parity[3] ) ;
                Sr[3] = PTemp[pos];
                _mm512_storeu_si512 ( PTemp, parity[4] ) ;
                Sr[4] = PTemp[pos];
                _mm512_storeu_si512 ( PTemp, parity[5] ) ;
                Sr[5] = PTemp[pos];
                _mm512_storeu_si512 ( PTemp, parity[6] ) ;
                Sr[6] = PTemp[pos];
                _mm512_storeu_si512 ( PTemp, parity[7] ) ;
                Sr[7] = PTemp[pos];
                _mm512_storeu_si512 ( PTemp, parity[8] ) ;
                Sr[8] = PTemp[pos];
                _mm512_storeu_si512 ( PTemp, parity[9] ) ;
                Sr[9] = PTemp[pos];
                _mm512_storeu_si512 ( PTemp, parity[10] ) ;
                Sr[10] = PTemp[pos];
                _mm512_storeu_si512 ( PTemp, parity[11] ) ;
                Sr[11] = PTemp[pos];
                _mm512_storeu_si512 ( PTemp, parity[12] ) ;
                Sr[12] = PTemp[pos];
                _mm512_storeu_si512 ( PTemp, parity[13] ) ;
                Sr[13] = PTemp[pos];
                _mm512_storeu_si512 ( PTemp, parity[14] ) ;
                Sr[14] = PTemp[pos];
                _mm512_storeu_si512 ( PTemp, parity[15] ) ;
                Sr[15] = PTemp[pos];
                _mm512_storeu_si512 ( PTemp, parity[16] ) ;
                Sr[16] = PTemp[pos];
                _mm512_storeu_si512 ( PTemp, parity[17] ) ;
                Sr[17] = PTemp[pos];
                _mm512_storeu_si512 ( PTemp, parity[18] ) ;
                Sr[18] = PTemp[pos];
                _mm512_storeu_si512 ( PTemp, parity[19] ) ;
                Sr[19] = PTemp[pos];
                _mm512_storeu_si512 ( PTemp, parity[20] ) ;
                Sr[20] = PTemp[pos];
                _mm512_storeu_si512 ( PTemp, parity[21] ) ;
                Sr[21] = PTemp[pos];
                _mm512_storeu_si512 ( PTemp, parity[22] ) ;
                Sr[22] = PTemp[pos];
                _mm512_storeu_si512 ( PTemp, parity[23] ) ;
                Sr[23] = PTemp[pos];
                _mm512_storeu_si512 ( PTemp, parity[24] ) ;
                Sr[24] = PTemp[pos];
                _mm512_storeu_si512 ( PTemp, parity[25] ) ;
                Sr[25] = PTemp[pos];
                _mm512_storeu_si512 ( PTemp, parity[26] ) ;
                Sr[26] = PTemp[pos];
                _mm512_storeu_si512 ( PTemp, parity[27] ) ;
                Sr[27] = PTemp[pos];
                rs_decode_palindromic_GFNI(Sr, S, 28, k, errLocs, errMags,
                             (int)pos+curPos, Lambda, B, T, Omega, data) ;
                mask = _blsr_u64(mask);
            }
        }
    }
}

// Parallel LFSRD_SR Sequencer for P = 29 Codewords
void ParallelLFSRSequencer29_GFNI(int len, int k, unsigned char **data, int decoder)
{
    int curSym, curPos ;                           // Loop counters
    __m512i parity [ 29 ], taps [ 14 ] ;             // Parity registers
    __m512i data_vec, temp [ 14 ] ;
    unsigned char **sPnt ;                         // Data lookup pointers

    // Initialize the taps to the LFSR
   taps [ 0 ] = _mm512_broadcast_i32x2(*( __m128i * ) &PCAffTab[0x2a]);
   taps [ 1 ] = _mm512_broadcast_i32x2(*( __m128i * ) &PCAffTab[0x98]);
   taps [ 2 ] = _mm512_broadcast_i32x2(*( __m128i * ) &PCAffTab[0x15]);
   taps [ 3 ] = _mm512_broadcast_i32x2(*( __m128i * ) &PCAffTab[0x3d]);
   taps [ 4 ] = _mm512_broadcast_i32x2(*( __m128i * ) &PCAffTab[0xe7]);
   taps [ 5 ] = _mm512_broadcast_i32x2(*( __m128i * ) &PCAffTab[0xa9]);
   taps [ 6 ] = _mm512_broadcast_i32x2(*( __m128i * ) &PCAffTab[0xb2]);
   taps [ 7 ] = _mm512_broadcast_i32x2(*( __m128i * ) &PCAffTab[0x16]);
   taps [ 8 ] = _mm512_broadcast_i32x2(*( __m128i * ) &PCAffTab[0xc5]);
   taps [ 9 ] = _mm512_broadcast_i32x2(*( __m128i * ) &PCAffTab[0x42]);
   taps [ 10 ] = _mm512_broadcast_i32x2(*( __m128i * ) &PCAffTab[0x8b]);
   taps [ 11 ] = _mm512_broadcast_i32x2(*( __m128i * ) &PCAffTab[0x5b]);
   taps [ 12 ] = _mm512_broadcast_i32x2(*( __m128i * ) &PCAffTab[0xd2]);
   taps [ 13 ] = _mm512_broadcast_i32x2(*( __m128i * ) &PCAffTab[0xfb]);

    // Adjust K if we are decoding
    k += decoder ? 29 : 0 ;

    // Loop through each 64 byte codeword
    for ( curPos = 0 ; curPos < len ; curPos += 64 )
    {
        sPnt = data ;
        parity [ 0 ] = _mm512_setzero_si512();
        parity [ 1 ] = _mm512_setzero_si512();
        parity [ 2 ] = _mm512_setzero_si512();
        parity [ 3 ] = _mm512_setzero_si512();
        parity [ 4 ] = _mm512_setzero_si512();
        parity [ 5 ] = _mm512_setzero_si512();
        parity [ 6 ] = _mm512_setzero_si512();
        parity [ 7 ] = _mm512_setzero_si512();
        parity [ 8 ] = _mm512_setzero_si512();
        parity [ 9 ] = _mm512_setzero_si512();
        parity [ 10 ] = _mm512_setzero_si512();
        parity [ 11 ] = _mm512_setzero_si512();
        parity [ 12 ] = _mm512_setzero_si512();
        parity [ 13 ] = _mm512_setzero_si512();
        parity [ 14 ] = _mm512_setzero_si512();
        parity [ 15 ] = _mm512_setzero_si512();
        parity [ 16 ] = _mm512_setzero_si512();
        parity [ 17 ] = _mm512_setzero_si512();
        parity [ 18 ] = _mm512_setzero_si512();
        parity [ 19 ] = _mm512_setzero_si512();
        parity [ 20 ] = _mm512_setzero_si512();
        parity [ 21 ] = _mm512_setzero_si512();
        parity [ 22 ] = _mm512_setzero_si512();
        parity [ 23 ] = _mm512_setzero_si512();
        parity [ 24 ] = _mm512_setzero_si512();
        parity [ 25 ] = _mm512_setzero_si512();
        parity [ 26 ] = _mm512_setzero_si512();
        parity [ 27 ] = _mm512_setzero_si512();
        parity [ 28 ] = _mm512_setzero_si512();

        // Loop through all the 0..k symbols
        for ( curSym = 0 ; curSym < k ; curSym ++ )
        {
            // Load next 64 bytes of Original Data
            data_vec = _mm512_stream_load_si512( *sPnt + curPos ) ;
          __builtin_prefetch ( *sPnt + curPos + 64, 0, 3 ) ;
            sPnt ++ ;

            // Add incoming data to MSB of parity, then update parities using Parallel Multiplier
            data_vec = _mm512_xor_si512( data_vec, parity [ 0 ] ) ;

            // Compute the first half of the tap values for palindromic code generator
            temp [ 0 ] = _mm512_gf2p8affine_epi64_epi8 ( data_vec, taps [ 0 ], 0 ) ;
            temp [ 1 ] = _mm512_gf2p8affine_epi64_epi8 ( data_vec, taps [ 1 ], 0 ) ;
            temp [ 2 ] = _mm512_gf2p8affine_epi64_epi8 ( data_vec, taps [ 2 ], 0 ) ;
            temp [ 3 ] = _mm512_gf2p8affine_epi64_epi8 ( data_vec, taps [ 3 ], 0 ) ;
            temp [ 4 ] = _mm512_gf2p8affine_epi64_epi8 ( data_vec, taps [ 4 ], 0 ) ;
            temp [ 5 ] = _mm512_gf2p8affine_epi64_epi8 ( data_vec, taps [ 5 ], 0 ) ;
            temp [ 6 ] = _mm512_gf2p8affine_epi64_epi8 ( data_vec, taps [ 6 ], 0 ) ;
            temp [ 7 ] = _mm512_gf2p8affine_epi64_epi8 ( data_vec, taps [ 7 ], 0 ) ;
            temp [ 8 ] = _mm512_gf2p8affine_epi64_epi8 ( data_vec, taps [ 8 ], 0 ) ;
            temp [ 9 ] = _mm512_gf2p8affine_epi64_epi8 ( data_vec, taps [ 9 ], 0 ) ;
            temp [ 10 ] = _mm512_gf2p8affine_epi64_epi8 ( data_vec, taps [ 10 ], 0 ) ;
            temp [ 11 ] = _mm512_gf2p8affine_epi64_epi8 ( data_vec, taps [ 11 ], 0 ) ;
            temp [ 12 ] = _mm512_gf2p8affine_epi64_epi8 ( data_vec, taps [ 12 ], 0 ) ;
            temp [ 13 ] = _mm512_gf2p8affine_epi64_epi8 ( data_vec, taps [ 13 ], 0 ) ;

            // Now cycle the LFSR
            parity [ 0 ] = _mm512_xor_si512 ( parity [ 1 ], temp [ 0 ] ) ;
            parity [ 1 ] = _mm512_xor_si512 ( parity [ 2 ], temp [ 1 ] ) ;
            parity [ 2 ] = _mm512_xor_si512 ( parity [ 3 ], temp [ 2 ] ) ;
            parity [ 3 ] = _mm512_xor_si512 ( parity [ 4 ], temp [ 3 ] ) ;
            parity [ 4 ] = _mm512_xor_si512 ( parity [ 5 ], temp [ 4 ] ) ;
            parity [ 5 ] = _mm512_xor_si512 ( parity [ 6 ], temp [ 5 ] ) ;
            parity [ 6 ] = _mm512_xor_si512 ( parity [ 7 ], temp [ 6 ] ) ;
            parity [ 7 ] = _mm512_xor_si512 ( parity [ 8 ], temp [ 7 ] ) ;
            parity [ 8 ] = _mm512_xor_si512 ( parity [ 9 ], temp [ 8 ] ) ;
            parity [ 9 ] = _mm512_xor_si512 ( parity [ 10 ], temp [ 9 ] ) ;
            parity [ 10 ] = _mm512_xor_si512 ( parity [ 11 ], temp [ 10 ] ) ;
            parity [ 11 ] = _mm512_xor_si512 ( parity [ 12 ], temp [ 11 ] ) ;
            parity [ 12 ] = _mm512_xor_si512 ( parity [ 13 ], temp [ 12 ] ) ;
            parity [ 13 ] = _mm512_xor_si512 ( parity [ 14 ], temp [ 13 ] ) ;
            parity [ 14 ] = _mm512_xor_si512 ( parity [ 15 ], temp [ 13 ] ) ;
            parity [ 15 ] = _mm512_xor_si512 ( parity [ 16 ], temp [ 12 ] ) ;
            parity [ 16 ] = _mm512_xor_si512 ( parity [ 17 ], temp [ 11 ] ) ;
            parity [ 17 ] = _mm512_xor_si512 ( parity [ 18 ], temp [ 10 ] ) ;
            parity [ 18 ] = _mm512_xor_si512 ( parity [ 19 ], temp [ 9 ] ) ;
            parity [ 19 ] = _mm512_xor_si512 ( parity [ 20 ], temp [ 8 ] ) ;
            parity [ 20 ] = _mm512_xor_si512 ( parity [ 21 ], temp [ 7 ] ) ;
            parity [ 21 ] = _mm512_xor_si512 ( parity [ 22 ], temp [ 6 ] ) ;
            parity [ 22 ] = _mm512_xor_si512 ( parity [ 23 ], temp [ 5 ] ) ;
            parity [ 23 ] = _mm512_xor_si512 ( parity [ 24 ], temp [ 4 ] ) ;
            parity [ 24 ] = _mm512_xor_si512 ( parity [ 25 ], temp [ 3 ] ) ;
            parity [ 25 ] = _mm512_xor_si512 ( parity [ 26 ], temp [ 2 ] ) ;
            parity [ 26 ] = _mm512_xor_si512 ( parity [ 27 ], temp [ 1 ] ) ;
            parity [ 27 ] = _mm512_xor_si512 ( parity [ 28 ], temp [ 0 ] ) ;
            parity [ 28 ] = data_vec ;
        }

        if ( decoder == 0 )
        {
            // Store parity back to memory
            _mm512_stream_si512( (&data [ k+0 ][curPos]), parity [ 0 ] ) ;
            _mm512_stream_si512( (&data [ k+1 ][curPos]), parity [ 1 ] ) ;
            _mm512_stream_si512( (&data [ k+2 ][curPos]), parity [ 2 ] ) ;
            _mm512_stream_si512( (&data [ k+3 ][curPos]), parity [ 3 ] ) ;
            _mm512_stream_si512( (&data [ k+4 ][curPos]), parity [ 4 ] ) ;
            _mm512_stream_si512( (&data [ k+5 ][curPos]), parity [ 5 ] ) ;
            _mm512_stream_si512( (&data [ k+6 ][curPos]), parity [ 6 ] ) ;
            _mm512_stream_si512( (&data [ k+7 ][curPos]), parity [ 7 ] ) ;
            _mm512_stream_si512( (&data [ k+8 ][curPos]), parity [ 8 ] ) ;
            _mm512_stream_si512( (&data [ k+9 ][curPos]), parity [ 9 ] ) ;
            _mm512_stream_si512( (&data [ k+10 ][curPos]), parity [ 10 ] ) ;
            _mm512_stream_si512( (&data [ k+11 ][curPos]), parity [ 11 ] ) ;
            _mm512_stream_si512( (&data [ k+12 ][curPos]), parity [ 12 ] ) ;
            _mm512_stream_si512( (&data [ k+13 ][curPos]), parity [ 13 ] ) ;
            _mm512_stream_si512( (&data [ k+14 ][curPos]), parity [ 14 ] ) ;
            _mm512_stream_si512( (&data [ k+15 ][curPos]), parity [ 15 ] ) ;
            _mm512_stream_si512( (&data [ k+16 ][curPos]), parity [ 16 ] ) ;
            _mm512_stream_si512( (&data [ k+17 ][curPos]), parity [ 17 ] ) ;
            _mm512_stream_si512( (&data [ k+18 ][curPos]), parity [ 18 ] ) ;
            _mm512_stream_si512( (&data [ k+19 ][curPos]), parity [ 19 ] ) ;
            _mm512_stream_si512( (&data [ k+20 ][curPos]), parity [ 20 ] ) ;
            _mm512_stream_si512( (&data [ k+21 ][curPos]), parity [ 21 ] ) ;
            _mm512_stream_si512( (&data [ k+22 ][curPos]), parity [ 22 ] ) ;
            _mm512_stream_si512( (&data [ k+23 ][curPos]), parity [ 23 ] ) ;
            _mm512_stream_si512( (&data [ k+24 ][curPos]), parity [ 24 ] ) ;
            _mm512_stream_si512( (&data [ k+25 ][curPos]), parity [ 25 ] ) ;
            _mm512_stream_si512( (&data [ k+26 ][curPos]), parity [ 26 ] ) ;
            _mm512_stream_si512( (&data [ k+27 ][curPos]), parity [ 27 ] ) ;
            _mm512_stream_si512( (&data [ k+28 ][curPos]), parity [ 28 ] ) ;
        }
        else
        {
            // Verify Syndromes are zero
            data_vec = _mm512_or_si512 ( parity [ 0 ], parity [ 1 ] ) ;
            data_vec = _mm512_ternarylogic_epi32 ( data_vec, parity [ 2 ], parity [ 3 ], 0xFE ) ;
            data_vec = _mm512_ternarylogic_epi32 ( data_vec, parity [ 4 ], parity [ 5 ], 0xFE ) ;
            data_vec = _mm512_ternarylogic_epi32 ( data_vec, parity [ 6 ], parity [ 7 ], 0xFE ) ;
            data_vec = _mm512_ternarylogic_epi32 ( data_vec, parity [ 8 ], parity [ 9 ], 0xFE ) ;
            data_vec = _mm512_ternarylogic_epi32 ( data_vec, parity [ 10 ], parity [ 11 ], 0xFE ) ;
            data_vec = _mm512_ternarylogic_epi32 ( data_vec, parity [ 12 ], parity [ 13 ], 0xFE ) ;
            data_vec = _mm512_ternarylogic_epi32 ( data_vec, parity [ 14 ], parity [ 15 ], 0xFE ) ;
            data_vec = _mm512_ternarylogic_epi32 ( data_vec, parity [ 16 ], parity [ 17 ], 0xFE ) ;
            data_vec = _mm512_ternarylogic_epi32 ( data_vec, parity [ 18 ], parity [ 19 ], 0xFE ) ;
            data_vec = _mm512_ternarylogic_epi32 ( data_vec, parity [ 20 ], parity [ 21 ], 0xFE ) ;
            data_vec = _mm512_ternarylogic_epi32 ( data_vec, parity [ 22 ], parity [ 23 ], 0xFE ) ;
            data_vec = _mm512_ternarylogic_epi32 ( data_vec, parity [ 24 ], parity [ 25 ], 0xFE ) ;
            data_vec = _mm512_ternarylogic_epi32 ( data_vec, parity [ 26 ], parity [ 27 ], 0xFE ) ;
            data_vec = _mm512_ternarylogic_epi32 ( data_vec, parity [ 28 ], parity [ 28 ], 0xFE ) ;
            __mmask64 mask = _mm512_test_epi8_mask ( data_vec, data_vec ) ;

            // If syndromes are not zero then process error
            while ( !_ktestz_mask64_u8 ( mask, mask ) ) 
            {
                PCErrCnt++;
                unsigned char Sr[29], S[29], errLocs[29], errMags[29] ;
                unsigned char Lambda[29+1], B[29+1], T[29+1], Omega[29+1] ;
                unsigned char PTemp[64] ;
                uint64_t pos = _tzcnt_u64 ( mask ) ;

                _mm512_storeu_si512 ( PTemp, parity[0] ) ;
                Sr[0] = PTemp[pos];
                _mm512_storeu_si512 ( PTemp, parity[1] ) ;
                Sr[1] = PTemp[pos];
                _mm512_storeu_si512 ( PTemp, parity[2] ) ;
                Sr[2] = PTemp[pos];
                _mm512_storeu_si512 ( PTemp, parity[3] ) ;
                Sr[3] = PTemp[pos];
                _mm512_storeu_si512 ( PTemp, parity[4] ) ;
                Sr[4] = PTemp[pos];
                _mm512_storeu_si512 ( PTemp, parity[5] ) ;
                Sr[5] = PTemp[pos];
                _mm512_storeu_si512 ( PTemp, parity[6] ) ;
                Sr[6] = PTemp[pos];
                _mm512_storeu_si512 ( PTemp, parity[7] ) ;
                Sr[7] = PTemp[pos];
                _mm512_storeu_si512 ( PTemp, parity[8] ) ;
                Sr[8] = PTemp[pos];
                _mm512_storeu_si512 ( PTemp, parity[9] ) ;
                Sr[9] = PTemp[pos];
                _mm512_storeu_si512 ( PTemp, parity[10] ) ;
                Sr[10] = PTemp[pos];
                _mm512_storeu_si512 ( PTemp, parity[11] ) ;
                Sr[11] = PTemp[pos];
                _mm512_storeu_si512 ( PTemp, parity[12] ) ;
                Sr[12] = PTemp[pos];
                _mm512_storeu_si512 ( PTemp, parity[13] ) ;
                Sr[13] = PTemp[pos];
                _mm512_storeu_si512 ( PTemp, parity[14] ) ;
                Sr[14] = PTemp[pos];
                _mm512_storeu_si512 ( PTemp, parity[15] ) ;
                Sr[15] = PTemp[pos];
                _mm512_storeu_si512 ( PTemp, parity[16] ) ;
                Sr[16] = PTemp[pos];
                _mm512_storeu_si512 ( PTemp, parity[17] ) ;
                Sr[17] = PTemp[pos];
                _mm512_storeu_si512 ( PTemp, parity[18] ) ;
                Sr[18] = PTemp[pos];
                _mm512_storeu_si512 ( PTemp, parity[19] ) ;
                Sr[19] = PTemp[pos];
                _mm512_storeu_si512 ( PTemp, parity[20] ) ;
                Sr[20] = PTemp[pos];
                _mm512_storeu_si512 ( PTemp, parity[21] ) ;
                Sr[21] = PTemp[pos];
                _mm512_storeu_si512 ( PTemp, parity[22] ) ;
                Sr[22] = PTemp[pos];
                _mm512_storeu_si512 ( PTemp, parity[23] ) ;
                Sr[23] = PTemp[pos];
                _mm512_storeu_si512 ( PTemp, parity[24] ) ;
                Sr[24] = PTemp[pos];
                _mm512_storeu_si512 ( PTemp, parity[25] ) ;
                Sr[25] = PTemp[pos];
                _mm512_storeu_si512 ( PTemp, parity[26] ) ;
                Sr[26] = PTemp[pos];
                _mm512_storeu_si512 ( PTemp, parity[27] ) ;
                Sr[27] = PTemp[pos];
                _mm512_storeu_si512 ( PTemp, parity[28] ) ;
                Sr[28] = PTemp[pos];
                rs_decode_palindromic_GFNI(Sr, S, 29, k, errLocs, errMags,
                             (int)pos+curPos, Lambda, B, T, Omega, data) ;
                mask = _blsr_u64(mask);
            }
        }
    }
}

// Parallel LFSRD_SR Sequencer for P = 30 Codewords
void ParallelLFSRSequencer30_GFNI(int len, int k, unsigned char **data, int decoder)
{
    int curSym, curPos ;                           // Loop counters
    __m512i parity [ 30 ], taps [ 15 ] ;             // Parity registers
    __m512i data_vec, temp [ 15 ] ;
    unsigned char **sPnt ;                         // Data lookup pointers

    // Initialize the taps to the LFSR
   taps [ 0 ] = _mm512_broadcast_i32x2(*( __m128i * ) &PCAffTab[0x39]);
   taps [ 1 ] = _mm512_broadcast_i32x2(*( __m128i * ) &PCAffTab[0xa8]);
   taps [ 2 ] = _mm512_broadcast_i32x2(*( __m128i * ) &PCAffTab[0x7a]);
   taps [ 3 ] = _mm512_broadcast_i32x2(*( __m128i * ) &PCAffTab[0x71]);
   taps [ 4 ] = _mm512_broadcast_i32x2(*( __m128i * ) &PCAffTab[0x4c]);
   taps [ 5 ] = _mm512_broadcast_i32x2(*( __m128i * ) &PCAffTab[0xe]);
   taps [ 6 ] = _mm512_broadcast_i32x2(*( __m128i * ) &PCAffTab[0xa7]);
   taps [ 7 ] = _mm512_broadcast_i32x2(*( __m128i * ) &PCAffTab[0x61]);
   taps [ 8 ] = _mm512_broadcast_i32x2(*( __m128i * ) &PCAffTab[0x23]);
   taps [ 9 ] = _mm512_broadcast_i32x2(*( __m128i * ) &PCAffTab[0xb7]);
   taps [ 10 ] = _mm512_broadcast_i32x2(*( __m128i * ) &PCAffTab[0x78]);
   taps [ 11 ] = _mm512_broadcast_i32x2(*( __m128i * ) &PCAffTab[0x7e]);
   taps [ 12 ] = _mm512_broadcast_i32x2(*( __m128i * ) &PCAffTab[0xab]);
   taps [ 13 ] = _mm512_broadcast_i32x2(*( __m128i * ) &PCAffTab[0x52]);
   taps [ 14 ] = _mm512_broadcast_i32x2(*( __m128i * ) &PCAffTab[0xb8]);

    // Adjust K if we are decoding
    k += decoder ? 30 : 0 ;

    // Loop through each 64 byte codeword
    for ( curPos = 0 ; curPos < len ; curPos += 64 )
    {
        sPnt = data ;
        parity [ 0 ] = _mm512_setzero_si512();
        parity [ 1 ] = _mm512_setzero_si512();
        parity [ 2 ] = _mm512_setzero_si512();
        parity [ 3 ] = _mm512_setzero_si512();
        parity [ 4 ] = _mm512_setzero_si512();
        parity [ 5 ] = _mm512_setzero_si512();
        parity [ 6 ] = _mm512_setzero_si512();
        parity [ 7 ] = _mm512_setzero_si512();
        parity [ 8 ] = _mm512_setzero_si512();
        parity [ 9 ] = _mm512_setzero_si512();
        parity [ 10 ] = _mm512_setzero_si512();
        parity [ 11 ] = _mm512_setzero_si512();
        parity [ 12 ] = _mm512_setzero_si512();
        parity [ 13 ] = _mm512_setzero_si512();
        parity [ 14 ] = _mm512_setzero_si512();
        parity [ 15 ] = _mm512_setzero_si512();
        parity [ 16 ] = _mm512_setzero_si512();
        parity [ 17 ] = _mm512_setzero_si512();
        parity [ 18 ] = _mm512_setzero_si512();
        parity [ 19 ] = _mm512_setzero_si512();
        parity [ 20 ] = _mm512_setzero_si512();
        parity [ 21 ] = _mm512_setzero_si512();
        parity [ 22 ] = _mm512_setzero_si512();
        parity [ 23 ] = _mm512_setzero_si512();
        parity [ 24 ] = _mm512_setzero_si512();
        parity [ 25 ] = _mm512_setzero_si512();
        parity [ 26 ] = _mm512_setzero_si512();
        parity [ 27 ] = _mm512_setzero_si512();
        parity [ 28 ] = _mm512_setzero_si512();
        parity [ 29 ] = _mm512_setzero_si512();

        // Loop through all the 0..k symbols
        for ( curSym = 0 ; curSym < k ; curSym ++ )
        {
            // Load next 64 bytes of Original Data
            data_vec = _mm512_stream_load_si512( *sPnt + curPos ) ;
          __builtin_prefetch ( *sPnt + curPos + 64, 0, 3 ) ;
            sPnt ++ ;

            // Add incoming data to MSB of parity, then update parities using Parallel Multiplier
            data_vec = _mm512_xor_si512( data_vec, parity [ 0 ] ) ;

            // Compute the first half of the tap values for palindromic code generator
            temp [ 0 ] = _mm512_gf2p8affine_epi64_epi8 ( data_vec, taps [ 0 ], 0 ) ;
            temp [ 1 ] = _mm512_gf2p8affine_epi64_epi8 ( data_vec, taps [ 1 ], 0 ) ;
            temp [ 2 ] = _mm512_gf2p8affine_epi64_epi8 ( data_vec, taps [ 2 ], 0 ) ;
            temp [ 3 ] = _mm512_gf2p8affine_epi64_epi8 ( data_vec, taps [ 3 ], 0 ) ;
            temp [ 4 ] = _mm512_gf2p8affine_epi64_epi8 ( data_vec, taps [ 4 ], 0 ) ;
            temp [ 5 ] = _mm512_gf2p8affine_epi64_epi8 ( data_vec, taps [ 5 ], 0 ) ;
            temp [ 6 ] = _mm512_gf2p8affine_epi64_epi8 ( data_vec, taps [ 6 ], 0 ) ;
            temp [ 7 ] = _mm512_gf2p8affine_epi64_epi8 ( data_vec, taps [ 7 ], 0 ) ;
            temp [ 8 ] = _mm512_gf2p8affine_epi64_epi8 ( data_vec, taps [ 8 ], 0 ) ;
            temp [ 9 ] = _mm512_gf2p8affine_epi64_epi8 ( data_vec, taps [ 9 ], 0 ) ;
            temp [ 10 ] = _mm512_gf2p8affine_epi64_epi8 ( data_vec, taps [ 10 ], 0 ) ;
            temp [ 11 ] = _mm512_gf2p8affine_epi64_epi8 ( data_vec, taps [ 11 ], 0 ) ;
            temp [ 12 ] = _mm512_gf2p8affine_epi64_epi8 ( data_vec, taps [ 12 ], 0 ) ;
            temp [ 13 ] = _mm512_gf2p8affine_epi64_epi8 ( data_vec, taps [ 13 ], 0 ) ;
            temp [ 14 ] = _mm512_gf2p8affine_epi64_epi8 ( data_vec, taps [ 14 ], 0 ) ;

            // Now cycle the LFSR
            parity [ 0 ] = _mm512_xor_si512 ( parity [ 1 ], temp [ 0 ] ) ;
            parity [ 1 ] = _mm512_xor_si512 ( parity [ 2 ], temp [ 1 ] ) ;
            parity [ 2 ] = _mm512_xor_si512 ( parity [ 3 ], temp [ 2 ] ) ;
            parity [ 3 ] = _mm512_xor_si512 ( parity [ 4 ], temp [ 3 ] ) ;
            parity [ 4 ] = _mm512_xor_si512 ( parity [ 5 ], temp [ 4 ] ) ;
            parity [ 5 ] = _mm512_xor_si512 ( parity [ 6 ], temp [ 5 ] ) ;
            parity [ 6 ] = _mm512_xor_si512 ( parity [ 7 ], temp [ 6 ] ) ;
            parity [ 7 ] = _mm512_xor_si512 ( parity [ 8 ], temp [ 7 ] ) ;
            parity [ 8 ] = _mm512_xor_si512 ( parity [ 9 ], temp [ 8 ] ) ;
            parity [ 9 ] = _mm512_xor_si512 ( parity [ 10 ], temp [ 9 ] ) ;
            parity [ 10 ] = _mm512_xor_si512 ( parity [ 11 ], temp [ 10 ] ) ;
            parity [ 11 ] = _mm512_xor_si512 ( parity [ 12 ], temp [ 11 ] ) ;
            parity [ 12 ] = _mm512_xor_si512 ( parity [ 13 ], temp [ 12 ] ) ;
            parity [ 13 ] = _mm512_xor_si512 ( parity [ 14 ], temp [ 13 ] ) ;
            parity [ 14 ] = _mm512_xor_si512 ( parity [ 15 ], temp [ 14 ] ) ;
            parity [ 15 ] = _mm512_xor_si512 ( parity [ 16 ], temp [ 13 ] ) ;
            parity [ 16 ] = _mm512_xor_si512 ( parity [ 17 ], temp [ 12 ] ) ;
            parity [ 17 ] = _mm512_xor_si512 ( parity [ 18 ], temp [ 11 ] ) ;
            parity [ 18 ] = _mm512_xor_si512 ( parity [ 19 ], temp [ 10 ] ) ;
            parity [ 19 ] = _mm512_xor_si512 ( parity [ 20 ], temp [ 9 ] ) ;
            parity [ 20 ] = _mm512_xor_si512 ( parity [ 21 ], temp [ 8 ] ) ;
            parity [ 21 ] = _mm512_xor_si512 ( parity [ 22 ], temp [ 7 ] ) ;
            parity [ 22 ] = _mm512_xor_si512 ( parity [ 23 ], temp [ 6 ] ) ;
            parity [ 23 ] = _mm512_xor_si512 ( parity [ 24 ], temp [ 5 ] ) ;
            parity [ 24 ] = _mm512_xor_si512 ( parity [ 25 ], temp [ 4 ] ) ;
            parity [ 25 ] = _mm512_xor_si512 ( parity [ 26 ], temp [ 3 ] ) ;
            parity [ 26 ] = _mm512_xor_si512 ( parity [ 27 ], temp [ 2 ] ) ;
            parity [ 27 ] = _mm512_xor_si512 ( parity [ 28 ], temp [ 1 ] ) ;
            parity [ 28 ] = _mm512_xor_si512 ( parity [ 29 ], temp [ 0 ] ) ;
            parity [ 29 ] = data_vec ;
        }

        if ( decoder == 0 )
        {
            // Store parity back to memory
            _mm512_stream_si512( (&data [ k+0 ][curPos]), parity [ 0 ] ) ;
            _mm512_stream_si512( (&data [ k+1 ][curPos]), parity [ 1 ] ) ;
            _mm512_stream_si512( (&data [ k+2 ][curPos]), parity [ 2 ] ) ;
            _mm512_stream_si512( (&data [ k+3 ][curPos]), parity [ 3 ] ) ;
            _mm512_stream_si512( (&data [ k+4 ][curPos]), parity [ 4 ] ) ;
            _mm512_stream_si512( (&data [ k+5 ][curPos]), parity [ 5 ] ) ;
            _mm512_stream_si512( (&data [ k+6 ][curPos]), parity [ 6 ] ) ;
            _mm512_stream_si512( (&data [ k+7 ][curPos]), parity [ 7 ] ) ;
            _mm512_stream_si512( (&data [ k+8 ][curPos]), parity [ 8 ] ) ;
            _mm512_stream_si512( (&data [ k+9 ][curPos]), parity [ 9 ] ) ;
            _mm512_stream_si512( (&data [ k+10 ][curPos]), parity [ 10 ] ) ;
            _mm512_stream_si512( (&data [ k+11 ][curPos]), parity [ 11 ] ) ;
            _mm512_stream_si512( (&data [ k+12 ][curPos]), parity [ 12 ] ) ;
            _mm512_stream_si512( (&data [ k+13 ][curPos]), parity [ 13 ] ) ;
            _mm512_stream_si512( (&data [ k+14 ][curPos]), parity [ 14 ] ) ;
            _mm512_stream_si512( (&data [ k+15 ][curPos]), parity [ 15 ] ) ;
            _mm512_stream_si512( (&data [ k+16 ][curPos]), parity [ 16 ] ) ;
            _mm512_stream_si512( (&data [ k+17 ][curPos]), parity [ 17 ] ) ;
            _mm512_stream_si512( (&data [ k+18 ][curPos]), parity [ 18 ] ) ;
            _mm512_stream_si512( (&data [ k+19 ][curPos]), parity [ 19 ] ) ;
            _mm512_stream_si512( (&data [ k+20 ][curPos]), parity [ 20 ] ) ;
            _mm512_stream_si512( (&data [ k+21 ][curPos]), parity [ 21 ] ) ;
            _mm512_stream_si512( (&data [ k+22 ][curPos]), parity [ 22 ] ) ;
            _mm512_stream_si512( (&data [ k+23 ][curPos]), parity [ 23 ] ) ;
            _mm512_stream_si512( (&data [ k+24 ][curPos]), parity [ 24 ] ) ;
            _mm512_stream_si512( (&data [ k+25 ][curPos]), parity [ 25 ] ) ;
            _mm512_stream_si512( (&data [ k+26 ][curPos]), parity [ 26 ] ) ;
            _mm512_stream_si512( (&data [ k+27 ][curPos]), parity [ 27 ] ) ;
            _mm512_stream_si512( (&data [ k+28 ][curPos]), parity [ 28 ] ) ;
            _mm512_stream_si512( (&data [ k+29 ][curPos]), parity [ 29 ] ) ;
        }
        else
        {
            // Verify Syndromes are zero
            data_vec = _mm512_or_si512 ( parity [ 0 ], parity [ 1 ] ) ;
            data_vec = _mm512_ternarylogic_epi32 ( data_vec, parity [ 2 ], parity [ 3 ], 0xFE ) ;
            data_vec = _mm512_ternarylogic_epi32 ( data_vec, parity [ 4 ], parity [ 5 ], 0xFE ) ;
            data_vec = _mm512_ternarylogic_epi32 ( data_vec, parity [ 6 ], parity [ 7 ], 0xFE ) ;
            data_vec = _mm512_ternarylogic_epi32 ( data_vec, parity [ 8 ], parity [ 9 ], 0xFE ) ;
            data_vec = _mm512_ternarylogic_epi32 ( data_vec, parity [ 10 ], parity [ 11 ], 0xFE ) ;
            data_vec = _mm512_ternarylogic_epi32 ( data_vec, parity [ 12 ], parity [ 13 ], 0xFE ) ;
            data_vec = _mm512_ternarylogic_epi32 ( data_vec, parity [ 14 ], parity [ 15 ], 0xFE ) ;
            data_vec = _mm512_ternarylogic_epi32 ( data_vec, parity [ 16 ], parity [ 17 ], 0xFE ) ;
            data_vec = _mm512_ternarylogic_epi32 ( data_vec, parity [ 18 ], parity [ 19 ], 0xFE ) ;
            data_vec = _mm512_ternarylogic_epi32 ( data_vec, parity [ 20 ], parity [ 21 ], 0xFE ) ;
            data_vec = _mm512_ternarylogic_epi32 ( data_vec, parity [ 22 ], parity [ 23 ], 0xFE ) ;
            data_vec = _mm512_ternarylogic_epi32 ( data_vec, parity [ 24 ], parity [ 25 ], 0xFE ) ;
            data_vec = _mm512_ternarylogic_epi32 ( data_vec, parity [ 26 ], parity [ 27 ], 0xFE ) ;
            data_vec = _mm512_ternarylogic_epi32 ( data_vec, parity [ 28 ], parity [ 29 ], 0xFE ) ;
            __mmask64 mask = _mm512_test_epi8_mask ( data_vec, data_vec ) ;

            // If syndromes are not zero then process error
            while ( !_ktestz_mask64_u8 ( mask, mask ) ) 
            {
                PCErrCnt++;
                unsigned char Sr[30], S[30], errLocs[30], errMags[30] ;
                unsigned char Lambda[30+1], B[30+1], T[30+1], Omega[30+1] ;
                unsigned char PTemp[64] ;
                uint64_t pos = _tzcnt_u64 ( mask ) ;

                _mm512_storeu_si512 ( PTemp, parity[0] ) ;
                Sr[0] = PTemp[pos];
                _mm512_storeu_si512 ( PTemp, parity[1] ) ;
                Sr[1] = PTemp[pos];
                _mm512_storeu_si512 ( PTemp, parity[2] ) ;
                Sr[2] = PTemp[pos];
                _mm512_storeu_si512 ( PTemp, parity[3] ) ;
                Sr[3] = PTemp[pos];
                _mm512_storeu_si512 ( PTemp, parity[4] ) ;
                Sr[4] = PTemp[pos];
                _mm512_storeu_si512 ( PTemp, parity[5] ) ;
                Sr[5] = PTemp[pos];
                _mm512_storeu_si512 ( PTemp, parity[6] ) ;
                Sr[6] = PTemp[pos];
                _mm512_storeu_si512 ( PTemp, parity[7] ) ;
                Sr[7] = PTemp[pos];
                _mm512_storeu_si512 ( PTemp, parity[8] ) ;
                Sr[8] = PTemp[pos];
                _mm512_storeu_si512 ( PTemp, parity[9] ) ;
                Sr[9] = PTemp[pos];
                _mm512_storeu_si512 ( PTemp, parity[10] ) ;
                Sr[10] = PTemp[pos];
                _mm512_storeu_si512 ( PTemp, parity[11] ) ;
                Sr[11] = PTemp[pos];
                _mm512_storeu_si512 ( PTemp, parity[12] ) ;
                Sr[12] = PTemp[pos];
                _mm512_storeu_si512 ( PTemp, parity[13] ) ;
                Sr[13] = PTemp[pos];
                _mm512_storeu_si512 ( PTemp, parity[14] ) ;
                Sr[14] = PTemp[pos];
                _mm512_storeu_si512 ( PTemp, parity[15] ) ;
                Sr[15] = PTemp[pos];
                _mm512_storeu_si512 ( PTemp, parity[16] ) ;
                Sr[16] = PTemp[pos];
                _mm512_storeu_si512 ( PTemp, parity[17] ) ;
                Sr[17] = PTemp[pos];
                _mm512_storeu_si512 ( PTemp, parity[18] ) ;
                Sr[18] = PTemp[pos];
                _mm512_storeu_si512 ( PTemp, parity[19] ) ;
                Sr[19] = PTemp[pos];
                _mm512_storeu_si512 ( PTemp, parity[20] ) ;
                Sr[20] = PTemp[pos];
                _mm512_storeu_si512 ( PTemp, parity[21] ) ;
                Sr[21] = PTemp[pos];
                _mm512_storeu_si512 ( PTemp, parity[22] ) ;
                Sr[22] = PTemp[pos];
                _mm512_storeu_si512 ( PTemp, parity[23] ) ;
                Sr[23] = PTemp[pos];
                _mm512_storeu_si512 ( PTemp, parity[24] ) ;
                Sr[24] = PTemp[pos];
                _mm512_storeu_si512 ( PTemp, parity[25] ) ;
                Sr[25] = PTemp[pos];
                _mm512_storeu_si512 ( PTemp, parity[26] ) ;
                Sr[26] = PTemp[pos];
                _mm512_storeu_si512 ( PTemp, parity[27] ) ;
                Sr[27] = PTemp[pos];
                _mm512_storeu_si512 ( PTemp, parity[28] ) ;
                Sr[28] = PTemp[pos];
                _mm512_storeu_si512 ( PTemp, parity[29] ) ;
                Sr[29] = PTemp[pos];
                rs_decode_palindromic_GFNI(Sr, S, 30, k, errLocs, errMags,
                             (int)pos+curPos, Lambda, B, T, Omega, data) ;
                mask = _blsr_u64(mask);
            }
        }
    }
}

// Parallel LFSRD_SR Sequencer for P = 31 Codewords
void ParallelLFSRSequencer31_GFNI(int len, int k, unsigned char **data, int decoder)
{
    int curSym, curPos ;                           // Loop counters
    __m512i parity [ 31 ], taps [ 15 ] ;             // Parity registers
    __m512i data_vec, temp [ 15 ] ;
    unsigned char **sPnt ;                         // Data lookup pointers

    // Initialize the taps to the LFSR
   taps [ 0 ] = _mm512_broadcast_i32x2(*( __m128i * ) &PCAffTab[0x20]);
   taps [ 1 ] = _mm512_broadcast_i32x2(*( __m128i * ) &PCAffTab[0x80]);
   taps [ 2 ] = _mm512_broadcast_i32x2(*( __m128i * ) &PCAffTab[0xa6]);
   taps [ 3 ] = _mm512_broadcast_i32x2(*( __m128i * ) &PCAffTab[0x27]);
   taps [ 4 ] = _mm512_broadcast_i32x2(*( __m128i * ) &PCAffTab[0x7d]);
   taps [ 5 ] = _mm512_broadcast_i32x2(*( __m128i * ) &PCAffTab[0x2c]);
   taps [ 6 ] = _mm512_broadcast_i32x2(*( __m128i * ) &PCAffTab[0x3b]);
   taps [ 7 ] = _mm512_broadcast_i32x2(*( __m128i * ) &PCAffTab[0x3f]);
   taps [ 8 ] = _mm512_broadcast_i32x2(*( __m128i * ) &PCAffTab[0xeb]);
   taps [ 9 ] = _mm512_broadcast_i32x2(*( __m128i * ) &PCAffTab[0xa5]);
   taps [ 10 ] = _mm512_broadcast_i32x2(*( __m128i * ) &PCAffTab[0xe0]);
   taps [ 11 ] = _mm512_broadcast_i32x2(*( __m128i * ) &PCAffTab[0x3e]);
   taps [ 12 ] = _mm512_broadcast_i32x2(*( __m128i * ) &PCAffTab[0xd]);
   taps [ 13 ] = _mm512_broadcast_i32x2(*( __m128i * ) &PCAffTab[0xc7]);
   taps [ 14 ] = _mm512_broadcast_i32x2(*( __m128i * ) &PCAffTab[0x49]);

    // Adjust K if we are decoding
    k += decoder ? 31 : 0 ;

    // Loop through each 64 byte codeword
    for ( curPos = 0 ; curPos < len ; curPos += 64 )
    {
        sPnt = data ;
        parity [ 0 ] = _mm512_setzero_si512();
        parity [ 1 ] = _mm512_setzero_si512();
        parity [ 2 ] = _mm512_setzero_si512();
        parity [ 3 ] = _mm512_setzero_si512();
        parity [ 4 ] = _mm512_setzero_si512();
        parity [ 5 ] = _mm512_setzero_si512();
        parity [ 6 ] = _mm512_setzero_si512();
        parity [ 7 ] = _mm512_setzero_si512();
        parity [ 8 ] = _mm512_setzero_si512();
        parity [ 9 ] = _mm512_setzero_si512();
        parity [ 10 ] = _mm512_setzero_si512();
        parity [ 11 ] = _mm512_setzero_si512();
        parity [ 12 ] = _mm512_setzero_si512();
        parity [ 13 ] = _mm512_setzero_si512();
        parity [ 14 ] = _mm512_setzero_si512();
        parity [ 15 ] = _mm512_setzero_si512();
        parity [ 16 ] = _mm512_setzero_si512();
        parity [ 17 ] = _mm512_setzero_si512();
        parity [ 18 ] = _mm512_setzero_si512();
        parity [ 19 ] = _mm512_setzero_si512();
        parity [ 20 ] = _mm512_setzero_si512();
        parity [ 21 ] = _mm512_setzero_si512();
        parity [ 22 ] = _mm512_setzero_si512();
        parity [ 23 ] = _mm512_setzero_si512();
        parity [ 24 ] = _mm512_setzero_si512();
        parity [ 25 ] = _mm512_setzero_si512();
        parity [ 26 ] = _mm512_setzero_si512();
        parity [ 27 ] = _mm512_setzero_si512();
        parity [ 28 ] = _mm512_setzero_si512();
        parity [ 29 ] = _mm512_setzero_si512();
        parity [ 30 ] = _mm512_setzero_si512();

        // Loop through all the 0..k symbols
        for ( curSym = 0 ; curSym < k ; curSym ++ )
        {
            // Load next 64 bytes of Original Data
            data_vec = _mm512_stream_load_si512( *sPnt + curPos ) ;
          __builtin_prefetch ( *sPnt + curPos + 64, 0, 3 ) ;
            sPnt ++ ;

            // Add incoming data to MSB of parity, then update parities using Parallel Multiplier
            data_vec = _mm512_xor_si512( data_vec, parity [ 0 ] ) ;

            // Compute the first half of the tap values for palindromic code generator
            temp [ 0 ] = _mm512_gf2p8affine_epi64_epi8 ( data_vec, taps [ 0 ], 0 ) ;
            temp [ 1 ] = _mm512_gf2p8affine_epi64_epi8 ( data_vec, taps [ 1 ], 0 ) ;
            temp [ 2 ] = _mm512_gf2p8affine_epi64_epi8 ( data_vec, taps [ 2 ], 0 ) ;
            temp [ 3 ] = _mm512_gf2p8affine_epi64_epi8 ( data_vec, taps [ 3 ], 0 ) ;
            temp [ 4 ] = _mm512_gf2p8affine_epi64_epi8 ( data_vec, taps [ 4 ], 0 ) ;
            temp [ 5 ] = _mm512_gf2p8affine_epi64_epi8 ( data_vec, taps [ 5 ], 0 ) ;
            temp [ 6 ] = _mm512_gf2p8affine_epi64_epi8 ( data_vec, taps [ 6 ], 0 ) ;
            temp [ 7 ] = _mm512_gf2p8affine_epi64_epi8 ( data_vec, taps [ 7 ], 0 ) ;
            temp [ 8 ] = _mm512_gf2p8affine_epi64_epi8 ( data_vec, taps [ 8 ], 0 ) ;
            temp [ 9 ] = _mm512_gf2p8affine_epi64_epi8 ( data_vec, taps [ 9 ], 0 ) ;
            temp [ 10 ] = _mm512_gf2p8affine_epi64_epi8 ( data_vec, taps [ 10 ], 0 ) ;
            temp [ 11 ] = _mm512_gf2p8affine_epi64_epi8 ( data_vec, taps [ 11 ], 0 ) ;
            temp [ 12 ] = _mm512_gf2p8affine_epi64_epi8 ( data_vec, taps [ 12 ], 0 ) ;
            temp [ 13 ] = _mm512_gf2p8affine_epi64_epi8 ( data_vec, taps [ 13 ], 0 ) ;
            temp [ 14 ] = _mm512_gf2p8affine_epi64_epi8 ( data_vec, taps [ 14 ], 0 ) ;

            // Now cycle the LFSR
            parity [ 0 ] = _mm512_xor_si512 ( parity [ 1 ], temp [ 0 ] ) ;
            parity [ 1 ] = _mm512_xor_si512 ( parity [ 2 ], temp [ 1 ] ) ;
            parity [ 2 ] = _mm512_xor_si512 ( parity [ 3 ], temp [ 2 ] ) ;
            parity [ 3 ] = _mm512_xor_si512 ( parity [ 4 ], temp [ 3 ] ) ;
            parity [ 4 ] = _mm512_xor_si512 ( parity [ 5 ], temp [ 4 ] ) ;
            parity [ 5 ] = _mm512_xor_si512 ( parity [ 6 ], temp [ 5 ] ) ;
            parity [ 6 ] = _mm512_xor_si512 ( parity [ 7 ], temp [ 6 ] ) ;
            parity [ 7 ] = _mm512_xor_si512 ( parity [ 8 ], temp [ 7 ] ) ;
            parity [ 8 ] = _mm512_xor_si512 ( parity [ 9 ], temp [ 8 ] ) ;
            parity [ 9 ] = _mm512_xor_si512 ( parity [ 10 ], temp [ 9 ] ) ;
            parity [ 10 ] = _mm512_xor_si512 ( parity [ 11 ], temp [ 10 ] ) ;
            parity [ 11 ] = _mm512_xor_si512 ( parity [ 12 ], temp [ 11 ] ) ;
            parity [ 12 ] = _mm512_xor_si512 ( parity [ 13 ], temp [ 12 ] ) ;
            parity [ 13 ] = _mm512_xor_si512 ( parity [ 14 ], temp [ 13 ] ) ;
            parity [ 14 ] = _mm512_xor_si512 ( parity [ 15 ], temp [ 14 ] ) ;
            parity [ 15 ] = _mm512_xor_si512 ( parity [ 16 ], temp [ 14 ] ) ;
            parity [ 16 ] = _mm512_xor_si512 ( parity [ 17 ], temp [ 13 ] ) ;
            parity [ 17 ] = _mm512_xor_si512 ( parity [ 18 ], temp [ 12 ] ) ;
            parity [ 18 ] = _mm512_xor_si512 ( parity [ 19 ], temp [ 11 ] ) ;
            parity [ 19 ] = _mm512_xor_si512 ( parity [ 20 ], temp [ 10 ] ) ;
            parity [ 20 ] = _mm512_xor_si512 ( parity [ 21 ], temp [ 9 ] ) ;
            parity [ 21 ] = _mm512_xor_si512 ( parity [ 22 ], temp [ 8 ] ) ;
            parity [ 22 ] = _mm512_xor_si512 ( parity [ 23 ], temp [ 7 ] ) ;
            parity [ 23 ] = _mm512_xor_si512 ( parity [ 24 ], temp [ 6 ] ) ;
            parity [ 24 ] = _mm512_xor_si512 ( parity [ 25 ], temp [ 5 ] ) ;
            parity [ 25 ] = _mm512_xor_si512 ( parity [ 26 ], temp [ 4 ] ) ;
            parity [ 26 ] = _mm512_xor_si512 ( parity [ 27 ], temp [ 3 ] ) ;
            parity [ 27 ] = _mm512_xor_si512 ( parity [ 28 ], temp [ 2 ] ) ;
            parity [ 28 ] = _mm512_xor_si512 ( parity [ 29 ], temp [ 1 ] ) ;
            parity [ 29 ] = _mm512_xor_si512 ( parity [ 30 ], temp [ 0 ] ) ;
            parity [ 30 ] = data_vec ;
        }

        if ( decoder == 0 )
        {
            // Store parity back to memory
            _mm512_stream_si512( (&data [ k+0 ][curPos]), parity [ 0 ] ) ;
            _mm512_stream_si512( (&data [ k+1 ][curPos]), parity [ 1 ] ) ;
            _mm512_stream_si512( (&data [ k+2 ][curPos]), parity [ 2 ] ) ;
            _mm512_stream_si512( (&data [ k+3 ][curPos]), parity [ 3 ] ) ;
            _mm512_stream_si512( (&data [ k+4 ][curPos]), parity [ 4 ] ) ;
            _mm512_stream_si512( (&data [ k+5 ][curPos]), parity [ 5 ] ) ;
            _mm512_stream_si512( (&data [ k+6 ][curPos]), parity [ 6 ] ) ;
            _mm512_stream_si512( (&data [ k+7 ][curPos]), parity [ 7 ] ) ;
            _mm512_stream_si512( (&data [ k+8 ][curPos]), parity [ 8 ] ) ;
            _mm512_stream_si512( (&data [ k+9 ][curPos]), parity [ 9 ] ) ;
            _mm512_stream_si512( (&data [ k+10 ][curPos]), parity [ 10 ] ) ;
            _mm512_stream_si512( (&data [ k+11 ][curPos]), parity [ 11 ] ) ;
            _mm512_stream_si512( (&data [ k+12 ][curPos]), parity [ 12 ] ) ;
            _mm512_stream_si512( (&data [ k+13 ][curPos]), parity [ 13 ] ) ;
            _mm512_stream_si512( (&data [ k+14 ][curPos]), parity [ 14 ] ) ;
            _mm512_stream_si512( (&data [ k+15 ][curPos]), parity [ 15 ] ) ;
            _mm512_stream_si512( (&data [ k+16 ][curPos]), parity [ 16 ] ) ;
            _mm512_stream_si512( (&data [ k+17 ][curPos]), parity [ 17 ] ) ;
            _mm512_stream_si512( (&data [ k+18 ][curPos]), parity [ 18 ] ) ;
            _mm512_stream_si512( (&data [ k+19 ][curPos]), parity [ 19 ] ) ;
            _mm512_stream_si512( (&data [ k+20 ][curPos]), parity [ 20 ] ) ;
            _mm512_stream_si512( (&data [ k+21 ][curPos]), parity [ 21 ] ) ;
            _mm512_stream_si512( (&data [ k+22 ][curPos]), parity [ 22 ] ) ;
            _mm512_stream_si512( (&data [ k+23 ][curPos]), parity [ 23 ] ) ;
            _mm512_stream_si512( (&data [ k+24 ][curPos]), parity [ 24 ] ) ;
            _mm512_stream_si512( (&data [ k+25 ][curPos]), parity [ 25 ] ) ;
            _mm512_stream_si512( (&data [ k+26 ][curPos]), parity [ 26 ] ) ;
            _mm512_stream_si512( (&data [ k+27 ][curPos]), parity [ 27 ] ) ;
            _mm512_stream_si512( (&data [ k+28 ][curPos]), parity [ 28 ] ) ;
            _mm512_stream_si512( (&data [ k+29 ][curPos]), parity [ 29 ] ) ;
            _mm512_stream_si512( (&data [ k+30 ][curPos]), parity [ 30 ] ) ;
        }
        else
        {
            // Verify Syndromes are zero
            data_vec = _mm512_or_si512 ( parity [ 0 ], parity [ 1 ] ) ;
            data_vec = _mm512_ternarylogic_epi32 ( data_vec, parity [ 2 ], parity [ 3 ], 0xFE ) ;
            data_vec = _mm512_ternarylogic_epi32 ( data_vec, parity [ 4 ], parity [ 5 ], 0xFE ) ;
            data_vec = _mm512_ternarylogic_epi32 ( data_vec, parity [ 6 ], parity [ 7 ], 0xFE ) ;
            data_vec = _mm512_ternarylogic_epi32 ( data_vec, parity [ 8 ], parity [ 9 ], 0xFE ) ;
            data_vec = _mm512_ternarylogic_epi32 ( data_vec, parity [ 10 ], parity [ 11 ], 0xFE ) ;
            data_vec = _mm512_ternarylogic_epi32 ( data_vec, parity [ 12 ], parity [ 13 ], 0xFE ) ;
            data_vec = _mm512_ternarylogic_epi32 ( data_vec, parity [ 14 ], parity [ 15 ], 0xFE ) ;
            data_vec = _mm512_ternarylogic_epi32 ( data_vec, parity [ 16 ], parity [ 17 ], 0xFE ) ;
            data_vec = _mm512_ternarylogic_epi32 ( data_vec, parity [ 18 ], parity [ 19 ], 0xFE ) ;
            data_vec = _mm512_ternarylogic_epi32 ( data_vec, parity [ 20 ], parity [ 21 ], 0xFE ) ;
            data_vec = _mm512_ternarylogic_epi32 ( data_vec, parity [ 22 ], parity [ 23 ], 0xFE ) ;
            data_vec = _mm512_ternarylogic_epi32 ( data_vec, parity [ 24 ], parity [ 25 ], 0xFE ) ;
            data_vec = _mm512_ternarylogic_epi32 ( data_vec, parity [ 26 ], parity [ 27 ], 0xFE ) ;
            data_vec = _mm512_ternarylogic_epi32 ( data_vec, parity [ 28 ], parity [ 29 ], 0xFE ) ;
            data_vec = _mm512_ternarylogic_epi32 ( data_vec, parity [ 30 ], parity [ 30 ], 0xFE ) ;
            __mmask64 mask = _mm512_test_epi8_mask ( data_vec, data_vec ) ;

            // If syndromes are not zero then process error
            while ( !_ktestz_mask64_u8 ( mask, mask ) ) 
            {
                PCErrCnt++;
                unsigned char Sr[31], S[31], errLocs[31], errMags[31] ;
                unsigned char Lambda[31+1], B[31+1], T[31+1], Omega[31+1] ;
                unsigned char PTemp[64] ;
                uint64_t pos = _tzcnt_u64 ( mask ) ;

                _mm512_storeu_si512 ( PTemp, parity[0] ) ;
                Sr[0] = PTemp[pos];
                _mm512_storeu_si512 ( PTemp, parity[1] ) ;
                Sr[1] = PTemp[pos];
                _mm512_storeu_si512 ( PTemp, parity[2] ) ;
                Sr[2] = PTemp[pos];
                _mm512_storeu_si512 ( PTemp, parity[3] ) ;
                Sr[3] = PTemp[pos];
                _mm512_storeu_si512 ( PTemp, parity[4] ) ;
                Sr[4] = PTemp[pos];
                _mm512_storeu_si512 ( PTemp, parity[5] ) ;
                Sr[5] = PTemp[pos];
                _mm512_storeu_si512 ( PTemp, parity[6] ) ;
                Sr[6] = PTemp[pos];
                _mm512_storeu_si512 ( PTemp, parity[7] ) ;
                Sr[7] = PTemp[pos];
                _mm512_storeu_si512 ( PTemp, parity[8] ) ;
                Sr[8] = PTemp[pos];
                _mm512_storeu_si512 ( PTemp, parity[9] ) ;
                Sr[9] = PTemp[pos];
                _mm512_storeu_si512 ( PTemp, parity[10] ) ;
                Sr[10] = PTemp[pos];
                _mm512_storeu_si512 ( PTemp, parity[11] ) ;
                Sr[11] = PTemp[pos];
                _mm512_storeu_si512 ( PTemp, parity[12] ) ;
                Sr[12] = PTemp[pos];
                _mm512_storeu_si512 ( PTemp, parity[13] ) ;
                Sr[13] = PTemp[pos];
                _mm512_storeu_si512 ( PTemp, parity[14] ) ;
                Sr[14] = PTemp[pos];
                _mm512_storeu_si512 ( PTemp, parity[15] ) ;
                Sr[15] = PTemp[pos];
                _mm512_storeu_si512 ( PTemp, parity[16] ) ;
                Sr[16] = PTemp[pos];
                _mm512_storeu_si512 ( PTemp, parity[17] ) ;
                Sr[17] = PTemp[pos];
                _mm512_storeu_si512 ( PTemp, parity[18] ) ;
                Sr[18] = PTemp[pos];
                _mm512_storeu_si512 ( PTemp, parity[19] ) ;
                Sr[19] = PTemp[pos];
                _mm512_storeu_si512 ( PTemp, parity[20] ) ;
                Sr[20] = PTemp[pos];
                _mm512_storeu_si512 ( PTemp, parity[21] ) ;
                Sr[21] = PTemp[pos];
                _mm512_storeu_si512 ( PTemp, parity[22] ) ;
                Sr[22] = PTemp[pos];
                _mm512_storeu_si512 ( PTemp, parity[23] ) ;
                Sr[23] = PTemp[pos];
                _mm512_storeu_si512 ( PTemp, parity[24] ) ;
                Sr[24] = PTemp[pos];
                _mm512_storeu_si512 ( PTemp, parity[25] ) ;
                Sr[25] = PTemp[pos];
                _mm512_storeu_si512 ( PTemp, parity[26] ) ;
                Sr[26] = PTemp[pos];
                _mm512_storeu_si512 ( PTemp, parity[27] ) ;
                Sr[27] = PTemp[pos];
                _mm512_storeu_si512 ( PTemp, parity[28] ) ;
                Sr[28] = PTemp[pos];
                _mm512_storeu_si512 ( PTemp, parity[29] ) ;
                Sr[29] = PTemp[pos];
                _mm512_storeu_si512 ( PTemp, parity[30] ) ;
                Sr[30] = PTemp[pos];
                rs_decode_palindromic_GFNI(Sr, S, 31, k, errLocs, errMags,
                             (int)pos+curPos, Lambda, B, T, Omega, data) ;
                mask = _blsr_u64(mask);
            }
        }
    }
}

// Parallel LFSRD_SR Sequencer for P = 32 Codewords
void ParallelLFSRSequencer32_GFNI(int len, int k, unsigned char **data, int decoder)
{
    int curSym, curPos ;                           // Loop counters
    __m512i parity [ 32 ], taps [ 16 ] ;             // Parity registers
    __m512i data_vec, temp [ 16 ] ;
    unsigned char **sPnt ;                         // Data lookup pointers

    // Initialize the taps to the LFSR
   taps [ 0 ] = _mm512_broadcast_i32x2(*( __m128i * ) &PCAffTab[0xec]);
   taps [ 1 ] = _mm512_broadcast_i32x2(*( __m128i * ) &PCAffTab[0xf4]);
   taps [ 2 ] = _mm512_broadcast_i32x2(*( __m128i * ) &PCAffTab[0xdc]);
   taps [ 3 ] = _mm512_broadcast_i32x2(*( __m128i * ) &PCAffTab[0x85]);
   taps [ 4 ] = _mm512_broadcast_i32x2(*( __m128i * ) &PCAffTab[0xee]);
   taps [ 5 ] = _mm512_broadcast_i32x2(*( __m128i * ) &PCAffTab[0x89]);
   taps [ 6 ] = _mm512_broadcast_i32x2(*( __m128i * ) &PCAffTab[0xc9]);
   taps [ 7 ] = _mm512_broadcast_i32x2(*( __m128i * ) &PCAffTab[0x7]);
   taps [ 8 ] = _mm512_broadcast_i32x2(*( __m128i * ) &PCAffTab[0x8d]);
   taps [ 9 ] = _mm512_broadcast_i32x2(*( __m128i * ) &PCAffTab[0xb]);
   taps [ 10 ] = _mm512_broadcast_i32x2(*( __m128i * ) &PCAffTab[0xe2]);
   taps [ 11 ] = _mm512_broadcast_i32x2(*( __m128i * ) &PCAffTab[0x22]);
   taps [ 12 ] = _mm512_broadcast_i32x2(*( __m128i * ) &PCAffTab[0xfc]);
   taps [ 13 ] = _mm512_broadcast_i32x2(*( __m128i * ) &PCAffTab[0xd1]);
   taps [ 14 ] = _mm512_broadcast_i32x2(*( __m128i * ) &PCAffTab[0x16]);
   taps [ 15 ] = _mm512_broadcast_i32x2(*( __m128i * ) &PCAffTab[0x4e]);

    // Adjust K if we are decoding
    k += decoder ? 32 : 0 ;

    // Loop through each 64 byte codeword
    for ( curPos = 0 ; curPos < len ; curPos += 64 )
    {
        sPnt = data ;
        parity [ 0 ] = _mm512_setzero_si512();
        parity [ 1 ] = _mm512_setzero_si512();
        parity [ 2 ] = _mm512_setzero_si512();
        parity [ 3 ] = _mm512_setzero_si512();
        parity [ 4 ] = _mm512_setzero_si512();
        parity [ 5 ] = _mm512_setzero_si512();
        parity [ 6 ] = _mm512_setzero_si512();
        parity [ 7 ] = _mm512_setzero_si512();
        parity [ 8 ] = _mm512_setzero_si512();
        parity [ 9 ] = _mm512_setzero_si512();
        parity [ 10 ] = _mm512_setzero_si512();
        parity [ 11 ] = _mm512_setzero_si512();
        parity [ 12 ] = _mm512_setzero_si512();
        parity [ 13 ] = _mm512_setzero_si512();
        parity [ 14 ] = _mm512_setzero_si512();
        parity [ 15 ] = _mm512_setzero_si512();
        parity [ 16 ] = _mm512_setzero_si512();
        parity [ 17 ] = _mm512_setzero_si512();
        parity [ 18 ] = _mm512_setzero_si512();
        parity [ 19 ] = _mm512_setzero_si512();
        parity [ 20 ] = _mm512_setzero_si512();
        parity [ 21 ] = _mm512_setzero_si512();
        parity [ 22 ] = _mm512_setzero_si512();
        parity [ 23 ] = _mm512_setzero_si512();
        parity [ 24 ] = _mm512_setzero_si512();
        parity [ 25 ] = _mm512_setzero_si512();
        parity [ 26 ] = _mm512_setzero_si512();
        parity [ 27 ] = _mm512_setzero_si512();
        parity [ 28 ] = _mm512_setzero_si512();
        parity [ 29 ] = _mm512_setzero_si512();
        parity [ 30 ] = _mm512_setzero_si512();
        parity [ 31 ] = _mm512_setzero_si512();

        // Loop through all the 0..k symbols
        for ( curSym = 0 ; curSym < k ; curSym ++ )
        {
            // Load next 64 bytes of Original Data
            data_vec = _mm512_stream_load_si512( *sPnt + curPos ) ;
          __builtin_prefetch ( *sPnt + curPos + 64, 0, 3 ) ;
            sPnt ++ ;

            // Add incoming data to MSB of parity, then update parities using Parallel Multiplier
            data_vec = _mm512_xor_si512( data_vec, parity [ 0 ] ) ;

            // Compute the first half of the tap values for palindromic code generator
            temp [ 0 ] = _mm512_gf2p8affine_epi64_epi8 ( data_vec, taps [ 0 ], 0 ) ;
            temp [ 1 ] = _mm512_gf2p8affine_epi64_epi8 ( data_vec, taps [ 1 ], 0 ) ;
            temp [ 2 ] = _mm512_gf2p8affine_epi64_epi8 ( data_vec, taps [ 2 ], 0 ) ;
            temp [ 3 ] = _mm512_gf2p8affine_epi64_epi8 ( data_vec, taps [ 3 ], 0 ) ;
            temp [ 4 ] = _mm512_gf2p8affine_epi64_epi8 ( data_vec, taps [ 4 ], 0 ) ;
            temp [ 5 ] = _mm512_gf2p8affine_epi64_epi8 ( data_vec, taps [ 5 ], 0 ) ;
            temp [ 6 ] = _mm512_gf2p8affine_epi64_epi8 ( data_vec, taps [ 6 ], 0 ) ;
            temp [ 7 ] = _mm512_gf2p8affine_epi64_epi8 ( data_vec, taps [ 7 ], 0 ) ;
            temp [ 8 ] = _mm512_gf2p8affine_epi64_epi8 ( data_vec, taps [ 8 ], 0 ) ;
            temp [ 9 ] = _mm512_gf2p8affine_epi64_epi8 ( data_vec, taps [ 9 ], 0 ) ;
            temp [ 10 ] = _mm512_gf2p8affine_epi64_epi8 ( data_vec, taps [ 10 ], 0 ) ;
            temp [ 11 ] = _mm512_gf2p8affine_epi64_epi8 ( data_vec, taps [ 11 ], 0 ) ;
            temp [ 12 ] = _mm512_gf2p8affine_epi64_epi8 ( data_vec, taps [ 12 ], 0 ) ;
            temp [ 13 ] = _mm512_gf2p8affine_epi64_epi8 ( data_vec, taps [ 13 ], 0 ) ;
            temp [ 14 ] = _mm512_gf2p8affine_epi64_epi8 ( data_vec, taps [ 14 ], 0 ) ;
            temp [ 15 ] = _mm512_gf2p8affine_epi64_epi8 ( data_vec, taps [ 15 ], 0 ) ;

            // Now cycle the LFSR
            parity [ 0 ] = _mm512_xor_si512 ( parity [ 1 ], temp [ 0 ] ) ;
            parity [ 1 ] = _mm512_xor_si512 ( parity [ 2 ], temp [ 1 ] ) ;
            parity [ 2 ] = _mm512_xor_si512 ( parity [ 3 ], temp [ 2 ] ) ;
            parity [ 3 ] = _mm512_xor_si512 ( parity [ 4 ], temp [ 3 ] ) ;
            parity [ 4 ] = _mm512_xor_si512 ( parity [ 5 ], temp [ 4 ] ) ;
            parity [ 5 ] = _mm512_xor_si512 ( parity [ 6 ], temp [ 5 ] ) ;
            parity [ 6 ] = _mm512_xor_si512 ( parity [ 7 ], temp [ 6 ] ) ;
            parity [ 7 ] = _mm512_xor_si512 ( parity [ 8 ], temp [ 7 ] ) ;
            parity [ 8 ] = _mm512_xor_si512 ( parity [ 9 ], temp [ 8 ] ) ;
            parity [ 9 ] = _mm512_xor_si512 ( parity [ 10 ], temp [ 9 ] ) ;
            parity [ 10 ] = _mm512_xor_si512 ( parity [ 11 ], temp [ 10 ] ) ;
            parity [ 11 ] = _mm512_xor_si512 ( parity [ 12 ], temp [ 11 ] ) ;
            parity [ 12 ] = _mm512_xor_si512 ( parity [ 13 ], temp [ 12 ] ) ;
            parity [ 13 ] = _mm512_xor_si512 ( parity [ 14 ], temp [ 13 ] ) ;
            parity [ 14 ] = _mm512_xor_si512 ( parity [ 15 ], temp [ 14 ] ) ;
            parity [ 15 ] = _mm512_xor_si512 ( parity [ 16 ], temp [ 15 ] ) ;
            parity [ 16 ] = _mm512_xor_si512 ( parity [ 17 ], temp [ 14 ] ) ;
            parity [ 17 ] = _mm512_xor_si512 ( parity [ 18 ], temp [ 13 ] ) ;
            parity [ 18 ] = _mm512_xor_si512 ( parity [ 19 ], temp [ 12 ] ) ;
            parity [ 19 ] = _mm512_xor_si512 ( parity [ 20 ], temp [ 11 ] ) ;
            parity [ 20 ] = _mm512_xor_si512 ( parity [ 21 ], temp [ 10 ] ) ;
            parity [ 21 ] = _mm512_xor_si512 ( parity [ 22 ], temp [ 9 ] ) ;
            parity [ 22 ] = _mm512_xor_si512 ( parity [ 23 ], temp [ 8 ] ) ;
            parity [ 23 ] = _mm512_xor_si512 ( parity [ 24 ], temp [ 7 ] ) ;
            parity [ 24 ] = _mm512_xor_si512 ( parity [ 25 ], temp [ 6 ] ) ;
            parity [ 25 ] = _mm512_xor_si512 ( parity [ 26 ], temp [ 5 ] ) ;
            parity [ 26 ] = _mm512_xor_si512 ( parity [ 27 ], temp [ 4 ] ) ;
            parity [ 27 ] = _mm512_xor_si512 ( parity [ 28 ], temp [ 3 ] ) ;
            parity [ 28 ] = _mm512_xor_si512 ( parity [ 29 ], temp [ 2 ] ) ;
            parity [ 29 ] = _mm512_xor_si512 ( parity [ 30 ], temp [ 1 ] ) ;
            parity [ 30 ] = _mm512_xor_si512 ( parity [ 31 ], temp [ 0 ] ) ;
            parity [ 31 ] = data_vec ;
        }

        if ( decoder == 0 )
        {
            // Store parity back to memory
            _mm512_stream_si512( (&data [ k+0 ][curPos]), parity [ 0 ] ) ;
            _mm512_stream_si512( (&data [ k+1 ][curPos]), parity [ 1 ] ) ;
            _mm512_stream_si512( (&data [ k+2 ][curPos]), parity [ 2 ] ) ;
            _mm512_stream_si512( (&data [ k+3 ][curPos]), parity [ 3 ] ) ;
            _mm512_stream_si512( (&data [ k+4 ][curPos]), parity [ 4 ] ) ;
            _mm512_stream_si512( (&data [ k+5 ][curPos]), parity [ 5 ] ) ;
            _mm512_stream_si512( (&data [ k+6 ][curPos]), parity [ 6 ] ) ;
            _mm512_stream_si512( (&data [ k+7 ][curPos]), parity [ 7 ] ) ;
            _mm512_stream_si512( (&data [ k+8 ][curPos]), parity [ 8 ] ) ;
            _mm512_stream_si512( (&data [ k+9 ][curPos]), parity [ 9 ] ) ;
            _mm512_stream_si512( (&data [ k+10 ][curPos]), parity [ 10 ] ) ;
            _mm512_stream_si512( (&data [ k+11 ][curPos]), parity [ 11 ] ) ;
            _mm512_stream_si512( (&data [ k+12 ][curPos]), parity [ 12 ] ) ;
            _mm512_stream_si512( (&data [ k+13 ][curPos]), parity [ 13 ] ) ;
            _mm512_stream_si512( (&data [ k+14 ][curPos]), parity [ 14 ] ) ;
            _mm512_stream_si512( (&data [ k+15 ][curPos]), parity [ 15 ] ) ;
            _mm512_stream_si512( (&data [ k+16 ][curPos]), parity [ 16 ] ) ;
            _mm512_stream_si512( (&data [ k+17 ][curPos]), parity [ 17 ] ) ;
            _mm512_stream_si512( (&data [ k+18 ][curPos]), parity [ 18 ] ) ;
            _mm512_stream_si512( (&data [ k+19 ][curPos]), parity [ 19 ] ) ;
            _mm512_stream_si512( (&data [ k+20 ][curPos]), parity [ 20 ] ) ;
            _mm512_stream_si512( (&data [ k+21 ][curPos]), parity [ 21 ] ) ;
            _mm512_stream_si512( (&data [ k+22 ][curPos]), parity [ 22 ] ) ;
            _mm512_stream_si512( (&data [ k+23 ][curPos]), parity [ 23 ] ) ;
            _mm512_stream_si512( (&data [ k+24 ][curPos]), parity [ 24 ] ) ;
            _mm512_stream_si512( (&data [ k+25 ][curPos]), parity [ 25 ] ) ;
            _mm512_stream_si512( (&data [ k+26 ][curPos]), parity [ 26 ] ) ;
            _mm512_stream_si512( (&data [ k+27 ][curPos]), parity [ 27 ] ) ;
            _mm512_stream_si512( (&data [ k+28 ][curPos]), parity [ 28 ] ) ;
            _mm512_stream_si512( (&data [ k+29 ][curPos]), parity [ 29 ] ) ;
            _mm512_stream_si512( (&data [ k+30 ][curPos]), parity [ 30 ] ) ;
            _mm512_stream_si512( (&data [ k+31 ][curPos]), parity [ 31 ] ) ;
        }
        else
        {
            // Verify Syndromes are zero
            data_vec = _mm512_or_si512 ( parity [ 0 ], parity [ 1 ] ) ;
            data_vec = _mm512_ternarylogic_epi32 ( data_vec, parity [ 2 ], parity [ 3 ], 0xFE ) ;
            data_vec = _mm512_ternarylogic_epi32 ( data_vec, parity [ 4 ], parity [ 5 ], 0xFE ) ;
            data_vec = _mm512_ternarylogic_epi32 ( data_vec, parity [ 6 ], parity [ 7 ], 0xFE ) ;
            data_vec = _mm512_ternarylogic_epi32 ( data_vec, parity [ 8 ], parity [ 9 ], 0xFE ) ;
            data_vec = _mm512_ternarylogic_epi32 ( data_vec, parity [ 10 ], parity [ 11 ], 0xFE ) ;
            data_vec = _mm512_ternarylogic_epi32 ( data_vec, parity [ 12 ], parity [ 13 ], 0xFE ) ;
            data_vec = _mm512_ternarylogic_epi32 ( data_vec, parity [ 14 ], parity [ 15 ], 0xFE ) ;
            data_vec = _mm512_ternarylogic_epi32 ( data_vec, parity [ 16 ], parity [ 17 ], 0xFE ) ;
            data_vec = _mm512_ternarylogic_epi32 ( data_vec, parity [ 18 ], parity [ 19 ], 0xFE ) ;
            data_vec = _mm512_ternarylogic_epi32 ( data_vec, parity [ 20 ], parity [ 21 ], 0xFE ) ;
            data_vec = _mm512_ternarylogic_epi32 ( data_vec, parity [ 22 ], parity [ 23 ], 0xFE ) ;
            data_vec = _mm512_ternarylogic_epi32 ( data_vec, parity [ 24 ], parity [ 25 ], 0xFE ) ;
            data_vec = _mm512_ternarylogic_epi32 ( data_vec, parity [ 26 ], parity [ 27 ], 0xFE ) ;
            data_vec = _mm512_ternarylogic_epi32 ( data_vec, parity [ 28 ], parity [ 29 ], 0xFE ) ;
            data_vec = _mm512_ternarylogic_epi32 ( data_vec, parity [ 30 ], parity [ 31 ], 0xFE ) ;
            __mmask64 mask = _mm512_test_epi8_mask ( data_vec, data_vec ) ;

            // If syndromes are not zero then process error
            while ( !_ktestz_mask64_u8 ( mask, mask ) ) 
            {
                PCErrCnt++;
                unsigned char Sr[32], S[32], errLocs[32], errMags[32] ;
                unsigned char Lambda[32+1], B[32+1], T[32+1], Omega[32+1] ;
                unsigned char PTemp[64] ;
                uint64_t pos = _tzcnt_u64 ( mask ) ;

                _mm512_storeu_si512 ( PTemp, parity[0] ) ;
                Sr[0] = PTemp[pos];
                _mm512_storeu_si512 ( PTemp, parity[1] ) ;
                Sr[1] = PTemp[pos];
                _mm512_storeu_si512 ( PTemp, parity[2] ) ;
                Sr[2] = PTemp[pos];
                _mm512_storeu_si512 ( PTemp, parity[3] ) ;
                Sr[3] = PTemp[pos];
                _mm512_storeu_si512 ( PTemp, parity[4] ) ;
                Sr[4] = PTemp[pos];
                _mm512_storeu_si512 ( PTemp, parity[5] ) ;
                Sr[5] = PTemp[pos];
                _mm512_storeu_si512 ( PTemp, parity[6] ) ;
                Sr[6] = PTemp[pos];
                _mm512_storeu_si512 ( PTemp, parity[7] ) ;
                Sr[7] = PTemp[pos];
                _mm512_storeu_si512 ( PTemp, parity[8] ) ;
                Sr[8] = PTemp[pos];
                _mm512_storeu_si512 ( PTemp, parity[9] ) ;
                Sr[9] = PTemp[pos];
                _mm512_storeu_si512 ( PTemp, parity[10] ) ;
                Sr[10] = PTemp[pos];
                _mm512_storeu_si512 ( PTemp, parity[11] ) ;
                Sr[11] = PTemp[pos];
                _mm512_storeu_si512 ( PTemp, parity[12] ) ;
                Sr[12] = PTemp[pos];
                _mm512_storeu_si512 ( PTemp, parity[13] ) ;
                Sr[13] = PTemp[pos];
                _mm512_storeu_si512 ( PTemp, parity[14] ) ;
                Sr[14] = PTemp[pos];
                _mm512_storeu_si512 ( PTemp, parity[15] ) ;
                Sr[15] = PTemp[pos];
                _mm512_storeu_si512 ( PTemp, parity[16] ) ;
                Sr[16] = PTemp[pos];
                _mm512_storeu_si512 ( PTemp, parity[17] ) ;
                Sr[17] = PTemp[pos];
                _mm512_storeu_si512 ( PTemp, parity[18] ) ;
                Sr[18] = PTemp[pos];
                _mm512_storeu_si512 ( PTemp, parity[19] ) ;
                Sr[19] = PTemp[pos];
                _mm512_storeu_si512 ( PTemp, parity[20] ) ;
                Sr[20] = PTemp[pos];
                _mm512_storeu_si512 ( PTemp, parity[21] ) ;
                Sr[21] = PTemp[pos];
                _mm512_storeu_si512 ( PTemp, parity[22] ) ;
                Sr[22] = PTemp[pos];
                _mm512_storeu_si512 ( PTemp, parity[23] ) ;
                Sr[23] = PTemp[pos];
                _mm512_storeu_si512 ( PTemp, parity[24] ) ;
                Sr[24] = PTemp[pos];
                _mm512_storeu_si512 ( PTemp, parity[25] ) ;
                Sr[25] = PTemp[pos];
                _mm512_storeu_si512 ( PTemp, parity[26] ) ;
                Sr[26] = PTemp[pos];
                _mm512_storeu_si512 ( PTemp, parity[27] ) ;
                Sr[27] = PTemp[pos];
                _mm512_storeu_si512 ( PTemp, parity[28] ) ;
                Sr[28] = PTemp[pos];
                _mm512_storeu_si512 ( PTemp, parity[29] ) ;
                Sr[29] = PTemp[pos];
                _mm512_storeu_si512 ( PTemp, parity[30] ) ;
                Sr[30] = PTemp[pos];
                _mm512_storeu_si512 ( PTemp, parity[31] ) ;
                Sr[31] = PTemp[pos];
                rs_decode_palindromic_GFNI(Sr, S, 32, k, errLocs, errMags,
                             (int)pos+curPos, Lambda, B, T, Omega, data) ;
                mask = _blsr_u64(mask);
            }
        }
    }
}

// Single function to access each unrolled LFSR Decode
void ParallelLFSRSequencer_GFNI(int len, int k, int p, unsigned char **data, int decode)
{
        switch (p) {
        case 2: ParallelLFSRSequencer2_GFNI(len, k, data, decode);
                 break ;
        case 3: ParallelLFSRSequencer3_GFNI(len, k, data, decode);
                 break ;
        case 4: ParallelLFSRSequencer4_GFNI(len, k, data, decode);
                 break ;
        case 5: ParallelLFSRSequencer5_GFNI(len, k, data, decode);
                 break ;
        case 6: ParallelLFSRSequencer6_GFNI(len, k, data, decode);
                 break ;
        case 7: ParallelLFSRSequencer7_GFNI(len, k, data, decode);
                 break ;
        case 8: ParallelLFSRSequencer8_GFNI(len, k, data, decode);
                 break ;
        case 9: ParallelLFSRSequencer9_GFNI(len, k, data, decode);
                 break ;
        case 10: ParallelLFSRSequencer10_GFNI(len, k, data, decode);
                 break ;
        case 11: ParallelLFSRSequencer11_GFNI(len, k, data, decode);
                 break ;
        case 12: ParallelLFSRSequencer12_GFNI(len, k, data, decode);
                 break ;
        case 13: ParallelLFSRSequencer13_GFNI(len, k, data, decode);
                 break ;
        case 14: ParallelLFSRSequencer14_GFNI(len, k, data, decode);
                 break ;
        case 15: ParallelLFSRSequencer15_GFNI(len, k, data, decode);
                 break ;
        case 16: ParallelLFSRSequencer16_GFNI(len, k, data, decode);
                 break ;
        case 17: ParallelLFSRSequencer17_GFNI(len, k, data, decode);
                 break ;
        case 18: ParallelLFSRSequencer18_GFNI(len, k, data, decode);
                 break ;
        case 19: ParallelLFSRSequencer19_GFNI(len, k, data, decode);
                 break ;
        case 20: ParallelLFSRSequencer20_GFNI(len, k, data, decode);
                 break ;
        case 21: ParallelLFSRSequencer21_GFNI(len, k, data, decode);
                 break ;
        case 22: ParallelLFSRSequencer22_GFNI(len, k, data, decode);
                 break ;
        case 23: ParallelLFSRSequencer23_GFNI(len, k, data, decode);
                 break ;
        case 24: ParallelLFSRSequencer24_GFNI(len, k, data, decode);
                 break ;
        case 25: ParallelLFSRSequencer25_GFNI(len, k, data, decode);
                 break ;
        case 26: ParallelLFSRSequencer26_GFNI(len, k, data, decode);
                 break ;
        case 27: ParallelLFSRSequencer27_GFNI(len, k, data, decode);
                 break ;
        case 28: ParallelLFSRSequencer28_GFNI(len, k, data, decode);
                 break ;
        case 29: ParallelLFSRSequencer29_GFNI(len, k, data, decode);
                 break ;
        case 30: ParallelLFSRSequencer30_GFNI(len, k, data, decode);
                 break ;
        case 31: ParallelLFSRSequencer31_GFNI(len, k, data, decode);
                 break ;
        case 32: ParallelLFSRSequencer32_GFNI(len, k, data, decode);
                 break ;
        }
}
