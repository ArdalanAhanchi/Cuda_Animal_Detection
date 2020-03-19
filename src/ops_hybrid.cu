/**
 *  A class which implements the ops interface for basic matrix operations on cpu and gpu.
 *  It uses the most optimized version based on the function.
 *
 *  @author Ardalan Ahanchi
 *  @date March 2020
 */

#include "ops_hybrid.cuh"
#include "ops_cpu.hpp"
#include "ops_gpu.cuh"
#include "mat.hpp"

#include <iostream>
#include <cmath>                                                        //For exponents.

//The N (for a NxN matrix), in which the rest of the calculations will be done on GPU.
#define CPU_LIMIT 350

namespace anr {


/**
 *  A function which adds matrix a and b, and returns a results matrix. a, and b 
 *  should be of exactly the same size. Should return a 0x0 matrix if error occured.
 *
 *  @param a The first matrix for the addition.
 *  @param b The second matrix for the addition.
 *  @return The results for the addition (only the pointer is passed by value).
 */
Mat Ops_hybrid::add(const Mat& a, const Mat& b) {
    //Call ops on cpu since add is more efficient on cpu.
    Ops_cpu ops;
    return ops.add(a, b);
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
Mat Ops_hybrid::sub(const Mat& a, const Mat& b) {
    //Call ops on cpu since sub is more efficient on cpu.
    Ops_cpu ops;
    return ops.sub(a, b);
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
Mat Ops_hybrid::mult(const Mat& a, const Mat& b)  {
    Ops* ops;

    //Check if we reach the limit, and then assign it to cpu or gpu.
    if(a.rows() > CPU_LIMIT && a.cols() > CPU_LIMIT && b.cols() > CPU_LIMIT) {
        Ops_gpu g;        
        ops = &g;
    } else {
        Ops_cpu c;
        ops = &c;
    }
        
    return ops->mult(a, b);
}  


/**
 *  A function which performs an element by element multiplication of matrices a 
 *  and b. The matrices should be the same size.
 *
 *  @param a The first matrix for the matrix multiplication.
 *  @param b The second matrix for the matrix multiplication.
 *  @return The results for the multiplication (only the pointer is passed by value).
 */
Mat Ops_hybrid::e_mult(const Mat& a, const Mat& b) {
    //Call ops on cpu since add is more efficient on cpu.
    Ops_cpu ops;
    return ops.e_mult(a, b);
}  


/**
 *  A function which scales the matrix a by a scalar value. It Should
 *  return a 0x0 matrix if error occured.
 *
 *  @param a The matrix which we're scaling.
 *  @param scale The scalar value which is multiplied to every element of a.
 *  @return The results for the scaling (only the pointer is passed by value).
 */ 
Mat Ops_hybrid::scale(const Mat& a, const type& scale) {
    //Call ops on cpu since add is more efficient on cpu.
    Ops_cpu ops;
    return ops.scale(a, scale);
}


    
/**
 *  A method which applies the sigmoid function to the passed matrix.
 *
 *  @param input The matrix where we're applying the sigmoid to. 
 */
void Ops_hybrid::sigmoid(Mat& input) {
    //Call ops on cpu since sigmoid is more efficient on cpu.
    Ops_cpu ops;
    return ops.sigmoid(input);
}   


/**
 *  A method which applies the derivative of sigmoid function to the passed matrix.
 *
 *  @param input The matrix where we're applying the sigmoid to. 
 */
void Ops_hybrid::deriv_sigmoid(Mat& input) {
    //Call ops on cpu since add is more efficient on cpu.
    Ops_cpu ops;
    return ops.deriv_sigmoid(input);
}


/**
 * A method which applies the relu function to the passed matrix.
 *
 * @param input The matrix where we're applying the sigmoid to. 
 */
void Ops_hybrid::relu(Mat& input) {
    //Call ops on cpu since add is more efficient on cpu.
    Ops_cpu ops;
    return ops.relu(input);
}


/**
 * A method which applies the derivative of the relu function to the passed matrix.
 *
 * @param input The matrix where we're applying the sigmoid to. 
 */
void Ops_hybrid::deriv_relu(Mat & input) {
    //Call ops on cpu since add is more efficient on cpu.
    Ops_cpu ops;
    return ops.deriv_relu(input);
}

}
