#ifndef CUDANET_VECTOR_H
#define CUDANET_VECTOR_H

namespace CUDANet::Utils {


/**
 * @brief Utility function that prints a vector
 * 
 * @param d_vec Pointer to the vector on device
 * @param length Length of the vector
 */
void print_vec(const float *d_vec, const unsigned int length);

/**
 * @brief Utility function that clears a vector
 * 
 * @param d_vector Pointer to the vector on device
 * @param len Length of the vector
 */
void clear(float *d_vector, const unsigned int len);


/**
 * @brief Utility function that returns the sum of a vector
 * 
 * @param d_vec Pointer to the vector
 * @param length Length of the vector
 */
void sum(const float *d_vec, float *d_sum, const unsigned int length);


/**
 * @brief Get the max of a vector
 * 
 * @param d_vec Pointer to the vector
 * @param length Length of the vector
 */
void max(const float *d_vec, float *d_max, const unsigned int length);


/**
 * @brief Compute the mean of the vector
 * 
 * @param d_vec Device pointer to the vector
 * @param d_mean Device pointer to the mean
 * @param d_length Device pointer to the length
 * @param length Length of the vector
 */
void mean(const float *d_vec, float *d_mean, float *d_length, int length);

/**
 * @brief Compute the variance of a vector
 * 
 * @param d_vec 
 * @param d_var 
 * @param length 
 */
void var(float *d_vec, float *d_var, float *d_length, const unsigned int length);

}  // namespace CUDANet::Utils

#endif  // CUDANET_VECTOR_H