#ifndef OPS_HPP
#define OPS_HPP

#include "mat.hpp"

namespace anr {

/**
 *  An interface class which represents basic matrix operations. It can be extended to
 *  implement operations on whichever hardware being used by the implementer.
 *
 *  @author Ardalan Ahanchi
 *  @date March 2020
 */
class Ops {
  public:

    /**
     *  A function which adds matrix a and b, and returns a results matrix. a, and b 
     *  should be of exactly the same size. Should return a 0x0 matrix if error occured.
     *
     *  @param a The first matrix for the addition.
     *  @param b The second matrix for the addition.
     *  @return The results for the addition (only the pointer is passed by value).
     */
    virtual Mat add(const Mat& a, const Mat& b) = 0;


    /**
     *  A function which subtracts matrix a and b, and returns a results matrix. a, and b 
     *  should be of exactly the same size. It basically returns a - b. Should return a 
     *  0x0 matrix if error occured.
     *
     *  @param a The first matrix for the subtraction.
     *  @param b The second matrix for the subtraction.
     *  @return The results for the subtraction (only the pointer is passed by value).
     */
    virtual Mat sub(const Mat& a, const Mat& b) = 0;


    /**
     *  A function which multiplies matrix a and b, and returns a results matrix. a should
     *  have the same number of cols, as b's rows. It basically returns a * b. Should
     *  return a 0x0 matrix if error occured.
     *
     *  @param a The first matrix for the matrix multiplication.
     *  @param b The second matrix for the matrix multiplication.
     *  @return The results for the multiplication (only the pointer is passed by value).
     */
    virtual Mat mult(const Mat& a, const Mat& b) = 0;   


    /**
     *  A function which performs an element by element multiplication of matrices a 
     *  and b. The matrices should be the same size.
     *
     *  @param a The first matrix for the matrix multiplication.
     *  @param b The second matrix for the matrix multiplication.
     *  @return The results for the multiplication (only the pointer is passed by value).
     */
    virtual Mat e_mult(const Mat& a, const Mat& b) = 0;   


    /**
     *  A function which scales the matrix a by a scalar value. It Should
     *  return a 0x0 matrix if error occured.
     *
     *  @param a The matrix which we're scaling.
     *  @param scale The scalar value which is multiplied to every element of a.
     *  @return The results for the scaling (only the pointer is passed by value).
     */ 
    virtual Mat scale(const Mat& a, const type& scale) = 0;


    /**
     *  A method which applies the sigmoid function to the passed matrix.
     *
     *  @param input The matrix where we're applying the sigmoid to. 
     */
    virtual void sigmoid(Mat& input) = 0;    


    /**
     *  A method which applies the derivative of sigmoid function to the passed matrix.
     *
     *  @param input The matrix where we're applying the sigmoid to. 
     */
    virtual void deriv_sigmoid(Mat& input) = 0; 


    /**
     *  A method which applies the relu function to the passed matrix.
     *
     *  @param input The matrix where we're applying the relu to.
     */
    virtual void relu(Mat& input) = 0;


    /**
     *  A method which applies the softmax function to the passed matrix.
     *
     *  @param input The matrix where we're applying the softmax to.
     */
    virtual void softmax(Mat& input) = 0;


    /**
     *  A method which applies the derivative of the relu function.
     *
     *  @param input The matrix where we're applying the relu to.
     */
    virtual void deriv_relu(Mat& input) = 0;
};

}

#endif
