#ifndef OPS_HYBRID_CUH
#define OPS_HYBRID_CUH

#include "mat.hpp"
#include "ops.hpp"

#include <string>

namespace anr {

/**
 *  A class which implements the ops interface for basic matrix operations on cpu and gpu.
 *  It uses the most optimized version based on the function.
 *
 *  @author Ardalan Ahanchi
 *  @date March 2020
 */
class Ops_hybrid : public Ops {
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
};  

}

#endif
