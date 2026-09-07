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

#include "DECODER_GFNI.h"

/* ------------------------------------------------------------------------------------ */
/*                                  GFNI Section                                        */
/* ------------------------------------------------------------------------------------ */

/**
 * Decodes a Reed-Solomon codeword block with a self-reciprocal (palindromic) configuration.
 *
 * @param Sr            Pointer to the LFSR Decoder output.
 * @param S             Pointer to the calculated Syndromes array in power sum form.
 * @param syndrome_cnt  Total number of syndromes (equal to 2t, the number of parity symbols).
 * @param codeword_len  Entire block size of the message (e.g., 255).
 * @param err_locs      Output array to store codeword indices where errors exist.
 * @param err_mags      Output array to store calculated error values for correction.
 * @param boff          Offset into memory for correction, legal values are 0-3.
 * @param Lambda        Error locator polynomial.
 * @param B             Helper polynomial for Berlekamp decoder.
 * @param T             Temporary storage for Berlekamp decoder.
 * @param Omega         Error evaluator polynomial.
 * @return              The number of errors corrected, or -1 if the block is uncorrectable.
 */
int rs_decode_palindromic_GFNI(const unsigned char* Sr, unsigned char* S, int syndrome_cnt,
    int codeword_len, unsigned char* err_locs, unsigned char* err_mags, int boff,
    unsigned char* Lambda, unsigned char* B, unsigned char* T, unsigned char* Omega,
    unsigned char ** data)
{
    /* ---------------------------------------------------------------------
     * AUTOMATIC PALINDROMIC ROOT SELECTION
     * Calculates f0 such that generator roots center symmetrically around alpha^0.
     * --------------------------------------------------------------------- */
    int exponent_sum = 255 - (syndrome_cnt - 1);
    if (exponent_sum % 2 != 0) {
        exponent_sum += 255;
    }
    int f0 = (exponent_sum / 2) % 255;

    // Start at higher powers
    int tf0 = f0 + syndrome_cnt;

    /* ---------------------------------------------------------------------
     * EVALUATE LFSR TERMS AT PALINDROMIC ROOTS
     * Calculates the power sum values from the LFSR result
     * --------------------------------------------------------------------- */
    for (int i = 0; i < syndrome_cnt; i++)
    {
        S[i] = 0;
        tf0--;
        //for (int j = 0; j < syndrome_cnt; j++)
        for (int j = syndrome_cnt - 1; j >= 0; j--)
        {
            S[i] ^= Sr[j];
            S[i] = gf_mul_GFNI(S[i], PCPowTab[tf0]);
        }
    }

    /* ---------------------------------------------------------------------
     * 1. BERLEKAMP-MASSEY ALGORITHM (Key Equation Solver)
     * Finds Error Locator Lambda(x) such that: S(x) * Lambda(x) = Omega(x) mod x^(2t)
     * --------------------------------------------------------------------- */
    memset(Lambda, 0, syndrome_cnt + 1);
    Lambda[0] = 1;
    memset(B, 0, syndrome_cnt + 1);
    B[0] = 1;

    int L = 0;           // Current assumed number of errors (degree of Lambda)
    int m = 1;           // Shift counter tracking steps since last L update
    unsigned char b = 1; // Discrepancy value from the last degree increase

    for (int r = 0; r < syndrome_cnt; r++) {
        unsigned char d = S[r];
        for (int i = 1; i <= L; i++) {
            d ^= gf_mul_GFNI(Lambda[i], S[r - i]); // Galois field addition is XOR
        }

        if (d == 0) {
            m++;
        }
        else {
            memcpy(T, Lambda, syndrome_cnt + 1);
            unsigned char scale = gf_div_GFNI(d, b);

            // Update Error Locator: Lambda(x) = Lambda(x) ^ (scale * x^m * B(x))
            for (int i = 0; i <= syndrome_cnt; i++) {
                if (i + m <= syndrome_cnt) {
                    Lambda[i + m] ^= gf_mul_GFNI(scale, B[i]);
                }
            }

            if (2 * L <= r) {
                L = r + 1 - L;
                memcpy(B, T, syndrome_cnt + 1);
                b = d;
                m = 1;
            }
            else {
                m++;
            }
        }
    }

    // Sanity boundary check: sequence exceeds core error correction budget (t)
    if (L * 2 > syndrome_cnt || L == 0) {
        return (L == 0) ? 0 : -1;
    }

    /* ---------------------------------------------------------------------
     * 2. ERROR EVALUATOR COMPUTATION
     * Computes Omega(x) = S(x) * Lambda(x) mod x^L
     * --------------------------------------------------------------------- */
    for (int i = 0; i < L; i++) {
        unsigned char sum = 0;
        for (int j = 0; j <= i; j++) {
            sum ^= gf_mul_GFNI(S[i - j], Lambda[j]);
        }
        Omega[i] = sum;
    }

    /* ---------------------------------------------------------------------
     * 3. CHIEN SEARCH & FORNEY ALGORITHM
     * Evaluates roots utilizing log/exp array indexing.
     * --------------------------------------------------------------------- */
    int error_idx = 0;

    for (int i = 0; i < codeword_len; i++) {
        // Maps physical position i directly to the matching inverse root element (alpha^-i)
        int target_exp = (255 - i) % 255;
        unsigned char inv_root = PCPowTab[target_exp];

        // Evaluate Lambda(inv_root)
        unsigned char lambda_val = 0;
        for (int j = 0; j <= L; j++) {
            unsigned char term_pow = 1;
            if (j > 0 && inv_root != 0) {
                int term_exp = (PCLogTab[inv_root] * j) % 255;
                if (term_exp < 0) term_exp += 255;
                term_pow = PCPowTab[term_exp];
            }
            else if (j > 0) {
                term_pow = 0;
            }
            lambda_val ^= gf_mul_GFNI(Lambda[j], term_pow);
        }

        // If lambda_val is 0, inv_root is an algebraic root (error located at position i)
        if (lambda_val == 0) {
            err_locs[error_idx] = i;

            // Evaluate Omega(inv_root)
            unsigned char omega_val = 0;
            for (int j = 0; j < L; j++) {
                unsigned char term_pow = 1;
                if (j > 0 && inv_root != 0) {
                    int term_exp = (PCLogTab[inv_root] * j) % 255;
                    if (term_exp < 0) term_exp += 255;
                    term_pow = PCPowTab[term_exp];
                }
                else if (j > 0) {
                    term_pow = 0;
                }
                omega_val ^= gf_mul_GFNI(Omega[j], term_pow);
            }

            // Evaluate Lambda'(inv_root) - formal derivative of Lambda
            unsigned char lambda_deriv = 0;
            for (int j = 1; j <= L; j += 2) { // Even indices cancel out to 0 under GF(2)
                unsigned char term_pow = 1;
                int deriv_pow = j - 1;
                if (deriv_pow > 0 && inv_root != 0) {
                    int term_exp = (PCLogTab[inv_root] * deriv_pow) % 255;
                    if (term_exp < 0) term_exp += 255;
                    term_pow = PCPowTab[term_exp];
                }
                else if (deriv_pow > 0) {
                    term_pow = 0;
                }
                lambda_deriv ^= gf_mul_GFNI(Lambda[j], term_pow);
            }

            /* ---------------------------------------------------------------------
             * FIXED FORNEY MAGNITUDE SCALING
             * Evaluates scaling using inv_root (X_k^-1) raised to (f0 - 1).
             * --------------------------------------------------------------------- */
            unsigned char num_scaling = 1;
            int scale_exp = f0 - 1;

            if (scale_exp != 0 && inv_root != 0) {
                int term_exp = (PCLogTab[inv_root] * scale_exp) % 255;
                if (term_exp < 0) term_exp += 255;
                num_scaling = PCPowTab[term_exp];
            }
            else if (scale_exp != 0) {
                num_scaling = 0;
            }

            unsigned char numerator = gf_mul_GFNI(omega_val, num_scaling);
            err_mags[error_idx] = gf_div_GFNI(numerator, lambda_deriv);
            error_idx++;

            if (error_idx >= L) break;
        }
    }

    // Correct the data
    for (int i = 0; i < L; i++)
    {
        int idx = codeword_len - err_locs[i] - 1;
        printf("GFNI Correcting buffer %d offset %d with %x\n", idx, boff, err_mags[i]);
        data[idx][boff] ^= err_mags[i];
    }

    return (error_idx == L) ? L : -1;
}

