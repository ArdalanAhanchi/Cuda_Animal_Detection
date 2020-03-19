/**
 *  An implementation for a class which implements the ops interface for basic matrix 
 *  operations on the gpu. This is used as a total gpu implementation.
 *
 *  @author Ardalan Ahanchi
 *  @date March 2020
 */

#include "ops_gpu.cuh"
#include "mat.hpp"

#include "device_launch_parameters.h"
#include <cuda_runtime.h>
#include <iostream>
#include <cmath>                                                        //For exponents.

//Define a device number if not defined in compilation.
#ifndef DEVICE_NUM
#define DEVICE_NUM 0
#endif

//Uses a TILE x TILE block for tiling. It can support up to 32x32 tiles.
#ifndef TILE 
#define TILE 8
#endif

//Number of threads launched per block for simple functions (all except matrix mult).
#define TPB 256

namespace anr {

/** The offset index used for each naive kernel (since the kernel exectues many times). */
__constant__ size_t kernel_offset[1];

/** The constant value which is used for the scale kernel. */
__constant__ type scale_number[1];

//Thes values are only used for matrix multiplication.
__constant__ size_t a_dims[2];  /**< Number of rows [0], and cols [1] in first matrix. */
__constant__ size_t b_dims[2];  /**< Number of rows [0], and cols [1] in second matrix. */
__constant__ size_t c_dims[2];  /**< Number of rows [0], and cols [1] in output matrix. */


/**
 *  A kernel which performs addition on the gpu. it adds every element of a and b and
 *  stores it in matrix c. It also takes into account the kerenel offset.
 *
 *  @param a_gpu location of the first matrix on GPU.
 *  @param b_gpu location of the second matrix on GPU.
 *  @param c_gpu location of the output matrix on GPU.
 */
__global__ void add_kernel(type* a_gpu, type* b_gpu, type* c_gpu) {
    //Calculate the current index in all the matrices and perform the addition.
    size_t idx = kernel_offset[0] + threadIdx.x + (blockIdx.x * blockDim.x);
    c_gpu[idx] = a_gpu[idx] + b_gpu[idx];
}


/**
 *  A kernel which performs subtraction on the gpu. it subtracts every element of 
 *  b from a and stores it in matrix c. It also takes into account the kerenel offset.
 *
 *  @param a_gpu location of the first matrix on GPU.
 *  @param b_gpu location of the second matrix on GPU.
 *  @param c_gpu location of the output matrix on GPU.
 */
__global__ void sub_kernel(type* a_gpu, type* b_gpu, type* c_gpu) {
    //Calculate the current index in all the matrices and perform the subtraction.
    size_t idx = kernel_offset[0] + threadIdx.x + (blockIdx.x * blockDim.x);
    c_gpu[idx] = a_gpu[idx] - b_gpu[idx];
}


/**
 *  A kernel which performs element by element multiplication on the gpu. it multiplies 
 *  every element of a into b and stores it in matrix c. It also takes into account the 
 *  kerenel offset.
 *
 *  @param a_gpu location of the first matrix on GPU.
 *  @param b_gpu location of the second matrix on GPU.
 *  @param c_gpu location of the output matrix on GPU.
 */
__global__ void e_mult_kernel(type* a_gpu, type* b_gpu, type* c_gpu) {
    //Calculate the current index in all the matrices and perform the multiplication.
    size_t idx = kernel_offset[0] + threadIdx.x + (blockIdx.x * blockDim.x);
    c_gpu[idx] = a_gpu[idx] * b_gpu[idx];
}


/**
 *  A kernel which performs tiled matrix multiplication on the gpu. It also uses shared
 *  memory and it can perform on arbitrary sized matrices.
 *
 *  @param a_gpu location of the first matrix on GPU.
 *  @param b_gpu location of the second matrix on GPU.
 *  @param c_gpu location of the output matrix on GPU.
 */
__global__ void mult_kernel(type* a_gpu, type* b_gpu, type* c_gpu) {
    //Holds each tile with TILE_SIDE x TILE_SIDE size for A and B matrices.
    __shared__ type a_tile[TILE * TILE];
	__shared__ type b_tile[TILE * TILE];

    //Seperately stored for caching, and simplifying the complex indexing.
	int tx = threadIdx.x;
    int ty = threadIdx.y;

	//Find the row and column for the output matrix (C).
	int row = blockIdx.x * TILE + tx;
	int col = blockIdx.y * TILE + ty;

    //Used to find C at the end of calculations within the block.
	type sum_c = 0;

	//Go through every tile (in a single direction).
	for(int t = 0; t < ((a_dims[0] - 1) / TILE + 1) ; t++)
    {
        //Load from A matrix into shared memory tile (if not a boundry).
        a_tile[(tx * TILE) + ty] = 
            (row < a_dims[0] && ((t * TILE) + ty) < a_dims[1]) ? 
            a_gpu[(row * a_dims[1]) + t * TILE + ty] : 0.0;

        //Load from B matrix into shared memory tile (if not a boundry).
        b_tile[(tx * TILE) + ty] = 
            (col < b_dims[1] && ((t * TILE) + tx) < b_dims[0]) ?
            b_gpu[col + ((t * TILE) + ty) * b_dims[1]] : 0.0;

		//Wait till all the data is completely stored in shared memory.
		__syncthreads();

        //Multiply the values from the matrices stored in shared memory.
		for(int i = 0; i < TILE; i++)
			sum_c += a_tile[(tx * TILE) + i] * b_tile[(i * TILE) + ty];

		//Wait till the sum is calculated before adding it up for the assignment.
		__syncthreads();
	}

	//Check if the current row/col are within the range, and then assign the value.
	if(row < c_dims[0] && col < c_dims[1])
		c_gpu[row * c_dims[1] + col] = sum_c;
}


/**
 *  A kernel which performs scaling of every matrix element and stores it in outputs.
 *  the amount of scaling is stored in a GPU constant called scale.
 *
 *  @param input location of the input matrix on GPU.
 *  @param input location of the input matrix on GPU.
 */
__global__ void scale_kernel(type* input, type* output) {
    //Calculate the current index in the matrix and perform the scaling.
    size_t idx = kernel_offset[0] + threadIdx.x + (blockIdx.x * blockDim.x);
    output[idx] = input[idx] * scale_number[0];
}


/**
 *  A kernel which performs a sigmoid for every element of the matrix in-place 
 *  and overwrites the current element of the matrix.
 *
 *  @param input location of the input matrix on GPU.
 */
__global__ void sigmoid_kernel(type* input) {
    //Calculate the current index in the matrix and perform the function.
    size_t idx = kernel_offset[0] + threadIdx.x + (blockIdx.x * blockDim.x);
    input[idx] = 1.0 / (1.0 + exp(-input[idx]));
}


/**
 *  A kernel which performs a sigmoid prime for every element of the matrix in-place 
 *  and overwrites the current element of the matrix.
 *
 *  @param input location of the input matrix on GPU.
 */
__global__ void d_sigmoid_kernel(type* input) {
    //Calculate the current index in the matrix and perform the function.
    size_t idx = kernel_offset[0] + threadIdx.x + (blockIdx.x * blockDim.x);
    input[idx] = exp(-input[idx]) / pow((1.0 + exp(-input[idx])), 2);
}


/**
 *  A kernel which performs a relu for every element of the matrix in-place 
 *  and overwrites the current element of the matrix.
 *
 *  @param input location of the input matrix.
 */
__global__ void relu_kernel(type* input) {
    //Calculate the current index in the matrix and perform the function.
    size_t idx = kernel_offset[0] + threadIdx.x + (blockIdx.x * blockDim.x);

    //Perform the relu function.
    if(input[idx] < 0.0)
        input[idx] = 0.0;
}


/**
 *  A kernel which performs a relu prime for every element of the matrix in-place 
 *  and overwrites the current element of the matrix.
 *
 *  @param input location of the input matrix on GPU.
 */
__global__ void d_relu_kernel(type* input) {
    //Calculate the current index in the matrix.
    size_t idx = kernel_offset[0] + threadIdx.x + (blockIdx.x * blockDim.x);

    //Perform the relu prime function.
    if (input[idx] <= 0.0)
        input[idx] = 0.0;
    else
        input[idx] = 1.0;
}


/**
 *  A function which adds matrix a and b, and returns a results matrix. a, and b 
 *  should be of exactly the same size. Should return a 0x0 matrix if error occured.
 *
 *  @param a The first matrix for the addition.
 *  @param b The second matrix for the addition.
 *  @return The results for the addition (only the pointer is passed by value).
 */
Mat Ops_gpu::add(const Mat& a, const Mat& b) {
    //Check if the matrices not the same size, print an error and return a 0x0 mat.
    if(a.rows() != b.rows() || a.cols() != b.cols())
        return error("Matrices should be the same dimentions", Op_Code::_add);

    //Define the output matrix to be the same size as a and b.
    Mat output(a.rows(), a.cols());

    //Transfer the input matrices to the GPU.
    type* a_gpu = this->transfer_to_gpu(a);
    type* b_gpu = this->transfer_to_gpu(b);

    //Allocate some memory on the GPU for the output matrix based on the size.
    type* output_gpu;
    size_t output_size = output.rows() * output.cols();
    cudaMalloc((void**) &output_gpu, ((size_t) sizeof(type)) * output_size);

    //Call the operations function with the correct opcode.
    this->operation(Op_Code::_add, output_size, a_gpu, b_gpu, output_gpu);

    //Transfer the data back into the output matrix.
    this->transfer_from_gpu(output_gpu, output);

    //Deallocate the memory on the GPU.
    cudaFree(a_gpu);
    cudaFree(b_gpu);
    cudaFree(output_gpu);

    //Return the results.
    return output;
}


/**
 *  A function which subtracts matrix a and b, and returns a results matrix. a, and b 
 *  should be of exactly the same size. It basically returns a - b. Should return a 
 *  0x0 matrix if error occured.
 *
 *  @param a The first matrix for the subtraction.
 *  @param b The second matrix for the subtraction.
 *  @return The results for the subtraction (only the pointer is passed by value).
 */
Mat Ops_gpu::sub(const Mat& a, const Mat& b) {
    //Check if the matrices not the same size, print an error and return a 0x0 mat.
    if(a.rows() != b.rows() || a.cols() != b.cols())
        return error("Matrices should be the same dimentions", Op_Code::_sub);

    //Define the output matrix to be the same size as a and b.
    Mat output(a.rows(), a.cols());

    //Transfer the input matrices to the GPU.
    type* a_gpu = this->transfer_to_gpu(a);
    type* b_gpu = this->transfer_to_gpu(b);

    //Allocate some memory on the GPU for the output matrix based on the size.
    type* output_gpu;
    size_t output_size = output.rows() * output.cols();
    cudaMalloc((void**) &output_gpu, ((size_t) sizeof(type)) * output_size);

    //Call the operations function with the correct opcode.
    this->operation(Op_Code::_sub, output_size, a_gpu, b_gpu, output_gpu);

    //Transfer the data back into the output matrix.
    this->transfer_from_gpu(output_gpu, output);

    //Deallocate the memory on the GPU.
    cudaFree(a_gpu);
    cudaFree(b_gpu);
    cudaFree(output_gpu);

    //Return the results.
    return output;
}


/**
 *  A function which performs an element by element multiplication of matrices a 
 *  and b. The matrices should be the same size.
 *
 *  @param a The first matrix for the matrix multiplication.
 *  @param b The second matrix for the matrix multiplication.
 *  @return The results for the multiplication (only the pointer is passed by value).
 */
Mat Ops_gpu::e_mult(const Mat& a, const Mat& b) {
    //Check if the matrices not the same size, print an error and return a 0x0 mat.
    if(a.rows() != b.rows() || a.cols() != b.cols())
        return error("Matrices should be the same dimentions", Op_Code::_e_mult);

    //Define the output matrix to be the same size as a and b.
    Mat output(a.rows(), a.cols());

    //Transfer the input matrices to the GPU.
    type* a_gpu = this->transfer_to_gpu(a);
    type* b_gpu = this->transfer_to_gpu(b);

    //Allocate some memory on the GPU for the output matrix based on the size.
    type* output_gpu;
    size_t output_size = output.rows() * output.cols();
    cudaMalloc((void**) &output_gpu, ((size_t) sizeof(type)) * output_size);

    //Transfer the data back into the output matrix.
    this->transfer_from_gpu(output_gpu, output);

    //Deallocate the memory on the GPU.
    cudaFree(a_gpu);
    cudaFree(b_gpu);
    cudaFree(output_gpu);

    //Return the results.
    return output;
}


/**
 *  A function which multiplies matrix a and b, and returns a results matrix. a should
 *  have the same number of cols, as b's rows. It basically returns a * b. Should
 *  return a 0x0 matrix if error occured.
 *
 *  @param a The first matrix for the matrix multiplication.
 *  @param b The second matrix for the matrix multiplication.
 *  @return The results for the multiplication (only the pointer is passed by value).
 */
Mat Ops_gpu::mult(const Mat& a, const Mat& b) {
    //Check if we have invalid sizes for the matrices.
    if(a.cols() != b.rows())
        return error("Invalid sizes for matrix multiplication", Op_Code::_mult);

    //Define the output matrix to be the correct size for output.
    Mat c(a.rows(), b.cols());

    //Transfer the input matrices to the GPU.
    type* a_gpu = this->transfer_to_gpu(a);
    type* b_gpu = this->transfer_to_gpu(b);

    //Allocate some memory on the GPU for the output matrix based on the size.
    type* c_gpu;
    size_t c_size = c.rows() * c.cols();
    cudaMalloc((void**) &c_gpu, ((size_t) sizeof(type)) * c_size);

    //Store the dimetions in arrays for transfering to constant memory.
    size_t a_dims_host[2] = {a.rows(), a.cols()};
    size_t b_dims_host[2] = {b.rows(), b.cols()};
    size_t c_dims_host[2] = {c.rows(), c.cols()};

    //Store the dimentions of the matrices in constant memory in gpu.
    cudaMemcpyToSymbol(a_dims, a_dims_host, sizeof(size_t) * 2);
    cudaMemcpyToSymbol(b_dims, b_dims_host, sizeof(size_t) * 2);
    cudaMemcpyToSymbol(c_dims, c_dims_host, sizeof(size_t) * 2);

    //Determine the size for 2D grids, and blocks using the tile sizes.    
    dim3 threads(TILE, TILE);

    //Side of the grid is rows/TILE_SIDE x cols/TILE_SIDE.
    dim3 blocks(std::ceil((type) c.rows() / (type) TILE), 
        std::ceil((type) c.cols() / (type) TILE));

    //Run the kernel with the calculated number of blocks and threads.
    mult_kernel<<<blocks, threads>>>(a_gpu, b_gpu, c_gpu);

    //Transfer the data back into the output matrix.
    this->transfer_from_gpu(c_gpu, c);

    //Deallocate the memory on the GPU.
    cudaFree(a_gpu);
    cudaFree(b_gpu);
    cudaFree(c_gpu);

    //Return the results.
    return c;
}


/**
 *  A function which scales the matrix a by a scalar value. It Should
 *  return a 0x0 matrix if error occured.
 *
 *  @param a The matrix which we're scaling.
 *  @param scale The scalar value which is multiplied to every element of a.
 *  @return The results for the scaling (only the pointer is passed by value).
 */ 
Mat Ops_gpu::scale(const Mat& a, const type& scale) {
    //Define the output matrix to be the same size as a.
    Mat output(a.rows(), a.cols());

    //Transfer the input matrix to the GPU.
    type* a_gpu = this->transfer_to_gpu(a);

    //Allocate some memory on the GPU for the output matrix based on the size.
    type* output_gpu;
    size_t output_size = output.rows() * output.cols();
    cudaMalloc((void**) &output_gpu, ((size_t) sizeof(type)) * output_size);

    //Store the scale value on the GPU's constant memory.
    type scale_host[1] = { (scale) };
    cudaMemcpyToSymbol(scale_number, scale_host, sizeof(type));

    //Call the operations function with the correct opcode.
    this->operation(Op_Code::_scale, output_size, a_gpu, nullptr, output_gpu);

    //Transfer the data back into the output matrix.
    this->transfer_from_gpu(output_gpu, output);

    //Deallocate the memory on the GPU.
    cudaFree(a_gpu);
    cudaFree(output_gpu);

    //Return the output matrix.
    return output;
}


/**
 *  A method which applies the sigmoid function to the passed matrix.
 *
 *  @param input The matrix where we're applying the sigmoid to. 
 */
void Ops_gpu::sigmoid(Mat& input) {
    //Transfer the input matrix to the GPU.
    type* input_gpu = this->transfer_to_gpu(input);
    size_t input_size = input.rows() * input.cols();

    //Call the operations function with the correct opcode.
    this->operation(Op_Code::_sigmoid, input_size, input_gpu);

    //Transfer the data back into the input matrix and override it.
    this->transfer_from_gpu(input_gpu, input);

    //Deallocate the GPU allocation.
    cudaFree(input_gpu);
} 


/**
 *  A method which applies the derivative of sigmoid function to the passed matrix.
 *
 *  @param input The matrix where we're applying the sigmoid to. 
 */
void Ops_gpu::deriv_sigmoid(Mat& input) {
    //Transfer the input matrix to the GPU.
    type* input_gpu = this->transfer_to_gpu(input);
    size_t input_size = input.rows() * input.cols();

    //Call the operations function with the correct opcode.
    this->operation(Op_Code::_d_sigmoid, input_size, input_gpu);

    //Transfer the data back into the input matrix and override it.
    this->transfer_from_gpu(input_gpu, input);

    //Deallocate the GPU allocation.
    cudaFree(input_gpu);
}


/**
 *  A method which applies the relu function to the passed matrix.
 *
 *  @param input The matrix where we're applying the relu to.
 */
void Ops_gpu::relu(Mat& input) {
    //Transfer the input matrix to the GPU.
    type* input_gpu = this->transfer_to_gpu(input);
    size_t input_size = input.rows() * input.cols();

    //Call the operations function with the correct opcode.
    this->operation(Op_Code::_relu, input_size, input_gpu);

    //Transfer the data back into the input matrix and override it.
    this->transfer_from_gpu(input_gpu, input);

    //Deallocate the GPU allocation.
    cudaFree(input_gpu);
}


/**
 *  A method which applies the derivative of the relu function.
 *
 *  @param input The matrix where we're applying the relu to.
 */
void Ops_gpu::deriv_relu(Mat& input) {
    //Transfer the input matrix to the GPU.
    type* input_gpu = this->transfer_to_gpu(input);
    size_t input_size = input.rows() * input.cols();

    //Call the operations function with the correct opcode.
    this->operation(Op_Code::_d_relu, input_size, input_gpu);

    //Transfer the data back into the input matrix and override it.
    this->transfer_from_gpu(input_gpu, input);

    //Deallocate the GPU allocation.
    cudaFree(input_gpu);
}


/**
 *  A function which transfers the given matrix to the GPU for calculations.
 *
 *  @param input The matrix object which we're transferring.
 *  @return The returned GPU address for the allocated memory.
 */
type* Ops_gpu::transfer_to_gpu(const Mat& input) {
    //Calculate the memory size based on the size of matrix.
    size_t memory_size = ((size_t)sizeof(type)) * input.rows() * input.cols();
    type* output; 
    cudaMalloc((void**) &output, memory_size);

    //Copy the data from the CPU to GPU.
    cudaMemcpy(output, input.data, memory_size, cudaMemcpyHostToDevice);

    //Return a pointer to the memory on GPU.
    return output;
}


/**
 *  A function which transfers back the results from the GPU after calculations.
 *
 *  @param source The GPU memory location for the transfer.
 *  @param output The destination host mat object used to write the data.
 */
void Ops_gpu::transfer_from_gpu(type* source, Mat& output) {
    //Calculate the memory size based on the size of output matrix.
    size_t memory_size = ((size_t)sizeof(type)) * output.rows() * output.cols();

    //Trasfer the matrix back to the host.
    cudaMemcpy(output.data, source, memory_size, cudaMemcpyDeviceToHost);
}


/**
 *  A function which performs simple cuda operations based on a given opcode.
 *  It accepts up to 3 matrices, and it will call the appropriate function based on
 *  the kernel_call function. Matrices b, and c could be nullptr for some opcodes.
 *
 *  @param opcode The operation code based on the opcode struct.
 *  @param output_size The number of elements in the output matrix (either a or c).
 *  @param a A pointer to the location of the first input matrix on GPU.
 *  @param b A pointer to the location of the second input matrix on GPU.
 *  @param c A pointer to the location of the output matrix on GPU.
 */
void Ops_gpu::operation(size_t opcode, size_t output_size, 
    type* a_gpu, type* b_gpu, type* c_gpu) {

    //Get the device properties for optimizing the memory usage.
    cudaDeviceProp stats;
    cudaGetDeviceProperties(&stats, DEVICE_NUM);

    //Get the maximum number of threads, and blocks based on the current architecture.
    size_t max_threads = (size_t) stats.maxThreadsPerBlock;
    size_t max_blocks = (size_t) stats.maxGridSize[0];

    //Select the correct number of threads based on the maximum supported, and requested.
    max_threads = (TPB > max_threads ? max_threads : TPB);

    //Holds the total number of calculations we need.
    size_t num_calcs = output_size;

    //Go through and calculate the kernel untill we finish the calculation.
    while(num_calcs != 0) {
        //Find the number of threads needed (based on how many calculations we have left).
        size_t num_threads = (num_calcs > max_threads ? max_threads : num_calcs);

        //Calculate the number of blocks required.
        size_t num_blocks = std::floor(num_calcs / num_threads);

        //Check if the number of blocks is more than maximum supported on the architecture.
        if(num_blocks > max_blocks)
            num_blocks = max_blocks;

        //Save the kernel offset in constant memory.
        size_t kernel_offset_host[1] = { (output_size - num_calcs) };
        cudaMemcpyToSymbol(kernel_offset, kernel_offset_host, sizeof(size_t));

        //Perform the kernel call based on the opcode.
        this->kernel_call(opcode, num_blocks, num_threads, a_gpu, b_gpu, c_gpu);

        //Reduce the number of calculations left based on the amount calculated.
        num_calcs -= (num_blocks * num_threads);
    }
}


/**
 *  A function which prints an error message (passed to it), and then returns a 0x0
 *  matrix (to be returned by the calling function. It also accepts an opcode which
 *  it uses for printing out errors in a nice format.
 *
 *  @param msg The message which will be printed.
 *  @param opcode The opcode for the calling function.
 *  @return A 0x0 matrix which can be returned by the function.
 */
Mat Ops_gpu::error(std::string msg, size_t opcode) {
    std::cerr << "Ops_gpu: " << Op_Names[opcode] << ": Error: " << msg << std::endl;
    return Mat(0, 0);
}


/**
 *  A function which calls the appropriate kernel for the opcode.
 *
 *  @param opcode The opcode corresponding to the kernel.
 *  @param blocks The number of blocks used for the execution.
 *  @param threads The number of threads used for the execution.
 *  @param a A pointer to the location of the first input matrix on GPU.
 *  @param b A pointer to the location of the second input matrix on GPU.
 *  @param c A pointer to the location of the output matrix on GPU.
 */
void Ops_gpu::kernel_call(size_t opcode, size_t blocks, size_t threads, 
    type* a, type* b, type* c) {

    //Perform the operation based on the opcode.
    switch(opcode) {
        //Add kernel.
        case Op_Code::_add: 
            add_kernel<<<blocks, threads>>>(a, b, c);
            break;

        //Subtract kernel.
        case Op_Code::_sub:
            sub_kernel<<<blocks, threads>>>(a, b, c);
            break;
    
        //Element multiplication kernel.
        case Op_Code::_e_mult:
            e_mult_kernel<<<blocks, threads>>>(a, b, c);
            break;

        //Element multiplication kernel.
        case Op_Code::_scale:
            scale_kernel<<<blocks, threads>>>(a, c);
            break;

        //Sigmoid kernel.
        case Op_Code::_sigmoid:
            sigmoid_kernel<<<blocks, threads>>>(a);
            break;

        //Derivitive of sigmoid kernel.
        case Op_Code::_d_sigmoid:
            d_sigmoid_kernel<<<blocks, threads>>>(a);
            break;

        //Relu kernel.
        case Op_Code::_relu:
            relu_kernel<<<blocks, threads>>>(a);
            break;

        //Derivitive of relu kernel.
        case Op_Code::_d_relu:
            d_relu_kernel<<<blocks, threads>>>(a);
            break;

        //Unsupported/Invalid opcode.
        default:
            std::cerr << "Ops_gpu: kernel_call: Error: Unsupported opcode." << std::endl;
            break;
    }
}


}
