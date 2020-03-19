#ifndef OPS_GPU_CUH
#define OPS_GPU_CUH

#include "mat.hpp"
#include "ops.hpp"

#include <string>

namespace anr {

/**
 *  A class which implements the ops interface for basic matrix operations on the gpu.
 *  this is used as a total gpu implementation (all functions on the gpu).
 *
 *  @author Ardalan Ahanchi
 *  @date March 2020
 */
class Ops_gpu : public Ops {
  public:

    /**
     *  A function which adds matrix a and b, and returns a results matrix. a, and b 
     *  should be of exactly the same size. Should return a 0x0 matrix if error occured.
     *
     *  @param a The first matrix for the addition.
     *  @param b The second matrix for the addition.
     *  @return The results for the addition (only the pointer is passed by value).
     */
    Mat add(const Mat& a, const Mat& b);


    /**
     *  A function which subtracts matrix a and b, and returns a results matrix. a, and b 
     *  should be of exactly the same size. It basically returns a - b. Should return a 
     *  0x0 matrix if error occured.
     *
     *  @param a The first matrix for the subtraction.
     *  @param b The second matrix for the subtraction.
     *  @return The results for the subtraction (only the pointer is passed by value).
     */
    Mat sub(const Mat& a, const Mat& b);


    /**
     *  A function which performs an element by element multiplication of matrices a 
     *  and b. The matrices should be the same size.
     *
     *  @param a The first matrix for the matrix multiplication.
     *  @param b The second matrix for the matrix multiplication.
     *  @return The results for the multiplication (only the pointer is passed by value).
     */
    Mat e_mult(const Mat& a, const Mat& b);


    /**
     *  A function which multiplies matrix a and b, and returns a results matrix. a should
     *  have the same number of cols, as b's rows. It basically returns a * b. Should
     *  return a 0x0 matrix if error occured.
     *
     *  @param a The first matrix for the matrix multiplication.
     *  @param b The second matrix for the matrix multiplication.
     *  @return The results for the multiplication (only the pointer is passed by value).
     */
    Mat mult(const Mat& a, const Mat& b);      


    /**
     *  A function which scales the matrix a by a scalar value. It Should
     *  return a 0x0 matrix if error occured.
     *
     *  @param a The matrix which we're scaling.
     *  @param scale The scalar value which is multiplied to every element of a.
     *  @return The results for the scaling (only the pointer is passed by value).
     */ 
    Mat scale(const Mat& a, const type& scale);

    
    /**
     *  A method which applies the sigmoid function to the passed matrix.
     *
     *  @param input The matrix where we're applying the sigmoid to. 
     */
    void sigmoid(Mat& input);    


    /**
     *  A method which applies the derivative of sigmoid function to the passed matrix.
     *
     *  @param input The matrix where we're applying the sigmoid to. 
     */
    void deriv_sigmoid(Mat& input); 


    /**
     *  A method which applies the relu function to the passed matrix.
     *
     *  @param input The matrix where we're applying the relu to.
     */
    void relu(Mat& input);


    /**
     *  A method which applies the derivative of the relu function.
     *
     *  @param input The matrix where we're applying the relu to.
     */
    void deriv_relu(Mat& input);

  private:

    /** An enum which represents the different operations. */
    enum Op_Code { _add, _sub, _e_mult, _mult, _scale, _sigmoid, _d_sigmoid, _relu, _d_relu };

    /** A constant array which holds the names for the opcodes. Used for printing. */
    const char* Op_Names[9] = {"add", "sub", "e_mult", "mult", "scale", "sigmoid", 
        "deriv_sigmoid", "relu", "deriv_relu"};

    
    /**
     *  A function which transfers the given matrix to the GPU for calculations.
     *
     *  @param input The matrix object which we're transferring.
     *  @return The returned GPU address for the allocated memory.
     */
    type* transfer_to_gpu(const Mat& input);


    /**
     *  A function which transfers back the results from the GPU after calculations.
     *
     *  @param source The GPU memory location for the transfer.
     *  @param output The destination host mat object used to write the data.
     */
    void transfer_from_gpu(type* source, Mat& output);


    /**
     *  A function which performs simple cuda operations based on a given opcode.
     *  It accepts up to 3 matrices, and it will call the appropriate function based on
     *  the kernel_call function. Matrices b, and c could be nullptr for some opcodes.
     *
     *  @param opcode The operation code based on the opcode struct.
     *  @param output_size The number of elements in the output matrix (either a or c).
     *  @param a A pointer to the location of the first input matrix on GPU.
     *  @param b An optional pointer to the location of the second input matrix on GPU.
     *  @param c An optional pointer to the location of the output matrix on GPU.
     */
    void operation(size_t opcode, size_t output_size, 
        type* a_gpu, type* b_gpu = nullptr, type* c_gpu = nullptr);


    /**
     *  A function which prints an error message (passed to it), and then returns a 0x0
     *  matrix (to be returned by the calling function. It also accepts an opcode which
     *  it uses for printing out errors in a nice format.
     *
     *  @param msg The message which will be printed.
     *  @param opcode The opcode for the calling function.
     *  @return A 0x0 matrix which can be returned by the function.
     */
    Mat error(std::string msg, size_t opcode);


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
    void kernel_call(size_t opcode, size_t blocks, size_t threads, 
        type* a, type* b, type* c);
};  

}

#endif
