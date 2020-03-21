/**
 *  An implementation for  a class which implements the ops interface for basic matrix operations on the gpu.
 *  this is used as a total gpu implementation (all functions on the gpu).
 *
 *  @author Ardalan Ahanchi
 *  @date March 2020
 */

#include "ops_cpu.hpp"
#include "mat.hpp"

#include <iostream>
#include <cmath>                                                        //For exponents.

namespace anr {


/**
 *  A function which adds matrix a and b, and returns a results matrix. a, and b 
 *  should be of exactly the same size. Should return a 0x0 matrix if error occured.
 *
 *  @param a The first matrix for the addition.
 *  @param b The second matrix for the addition.
 *  @return The results for the addition (only the pointer is passed by value).
 */
Mat Ops_cpu::add(const Mat& a, const Mat& b) {
    //If the matrices are not the same size, return a 0x0 matrix.
    if(a.rows() != b.rows() || a.cols() != b.cols()) {
        std::cerr << "Ops_cpu: add: Error: Matrices should be the same dimentions." << std::endl;
        return Mat(0, 0);
    }


    //If the matrices are the same size, initialize the output matrix.
    Mat output(a.rows(), a.cols());

    //Calculate and return the results for the operation.
    for(size_t r = 0; r < output.rows(); r++) {
        for(size_t c = 0; c < output.cols(); c++) {
            size_t idx = r * output.cols() + c;
            output.data[idx] = a.data[idx] + b.data[idx];
        }
    }

    return output;  //Returns by value, but the data is on heap, so it's a shallow copy.
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
Mat Ops_cpu::sub(const Mat& a, const Mat& b) {
    //If the matrices are not the same size, return a 0x0 matrix.
    if(a.rows() != b.rows() || a.cols() != b.cols()){
        std::cerr << "Ops_cpu: sub: Error: Matrices should be the same dimentions." << std::endl;
        return Mat(0, 0);
    }

    //If the matrices are the same size, initialize the output matrix.
    Mat output(a.rows(), a.cols());

    //Calculate and return the results for the operation.
    for(size_t r = 0; r < output.rows(); r++) {
        for(size_t c = 0; c < output.cols(); c++) {
            size_t idx = r * output.cols() + c;
            output.data[idx] = a.data[idx] - b.data[idx];
        }
    }

    return output;  //Returns by value, but the data is on heap, so it's a shallow copy.
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
Mat Ops_cpu::mult(const Mat& a, const Mat& b)  {
    //If the matrices are not the correct size, return a 0x0 matrix.
    if(a.cols() != b.rows()) {
        std::cerr << "Ops_cpu: mult: Error: Invalid sizes for multiplication." << std::endl;
        return Mat(0, 0);
    }

    //If the matrices are the same size, initialize the output matrix to the correct size.
    Mat output(a.rows(), b.cols());

    //Three nested loops to calculate a basic matrix multiplication.
    for(size_t i = 0; i < a.rows(); i++)
        for(size_t j = 0; j < b.cols(); j++)
            for(size_t k = 0; k < b.rows(); k++)
                output.data[i * output.cols() + j] += 
                    a.data[i * a.cols() + k] * b.data[k * b.cols() + j];

    return output;  //Returns by value, but the data is on heap, so it's a shallow copy.
}  


/**
 *  A function which performs an element by element multiplication of matrices a 
 *  and b. The matrices should be the same size.
 *
 *  @param a The first matrix for the matrix multiplication.
 *  @param b The second matrix for the matrix multiplication.
 *  @return The results for the multiplication (only the pointer is passed by value).
 */
Mat Ops_cpu::e_mult(const Mat& a, const Mat& b) {
    //If the matrices are not the same size, return a 0x0 matrix.
    if(a.rows() != b.rows() || a.cols() != b.cols()) {
        std::cerr << "Ops_cpu: e_mult: Error: Matrices should be the same dimentions." 
            << std::endl;

        return Mat(0, 0);
    }

    //If the matrices are the same size, initialize the output matrix.
    Mat output(a.rows(), a.cols());

    //Calculate and return the results for the operation.
    for(size_t r = 0; r < output.rows(); r++) {
        for(size_t c = 0; c < output.cols(); c++) {
            size_t idx = r * output.cols() + c;
            output.data[idx] = a.data[idx] * b.data[idx];
        }
    }

    return output;  //Returns by value, but the data is on heap, so it's a shallow copy.
}  


/**
 *  A function which scales the matrix a by a scalar value. It Should
 *  return a 0x0 matrix if error occured.
 *
 *  @param a The matrix which we're scaling.
 *  @param scale The scalar value which is multiplied to every element of a.
 *  @return The results for the scaling (only the pointer is passed by value).
 */ 
Mat Ops_cpu::scale(const Mat& a, const type& scale) {
    //Initialize the output matrix.
    Mat output(a.rows(), a.cols());

    //Calculate the results for the scaling operation.
    for(size_t r = 0; r < output.rows(); r++)
        for(size_t c = 0; c < output.cols(); c++)
            output.data[r * output.cols() + c] = a.data[r * output.cols() + c] * scale;

    return output;  //Returns by value, but the data is on heap, so it's a shallow copy.
}


    
/**
 *  A method which applies the sigmoid function to the passed matrix.
 *
 *  @param input The matrix where we're applying the sigmoid to. 
 */
void Ops_cpu::sigmoid(Mat& input) {
    //Apply the sigmoid to each element.
    for(size_t r = 0; r < input.rows(); r++) {
        for(size_t c = 0; c < input.cols(); c++) {
            size_t idx = r * input.cols() + c;
            input.data[idx] = 1.0 / (1.0 + std::exp(-input.data[idx])); 
        }
    }
}   


/**
 *  A method which applies the derivative of sigmoid function to the passed matrix.
 *
 *  @param input The matrix where we're applying the sigmoid to. 
 */
void Ops_cpu::deriv_sigmoid(Mat& input) {
    //Apply the derivative of sigmoid to each element.
    for(size_t r = 0; r < input.rows(); r++) {
        for(size_t c = 0; c < input.cols(); c++) {
            //Get the current index.
            size_t idx = r * input.cols() + c;

            //Calculate the derivative of sigmoid.
            input.data[idx] = std::exp(-input.data[idx]) 
                / std::pow((1 + std::exp(-input.data[idx])), 2); 
        }
    }
}


/**
 * A method which applies the relu function to the passed matrix.
 *
 * @param input The matrix where we're applying the sigmoid to. 
 */
void Ops_cpu::relu(Mat& input) {
    //Apply the relu to each element.
    for (size_t r = 0; r < input.rows(); r++) {
        for (size_t c = 0; c < input.cols(); c++) {
            size_t idx = r * input.cols() + c;
            if (input.data[idx] < 0)
                input.data[idx] = 0;
        }
    }
}


/**
 * A method which applies the derivative of the relu function to the passed matrix.
 *
 * @param input The matrix where we're applying the sigmoid to. 
 */
void Ops_cpu::deriv_relu(Mat & input) {
    //Apply the derivative to each element.
    for (size_t r = 0; r < input.rows(); r++) {
        for (size_t c = 0; c < input.cols(); c++) {
            size_t idx = r * input.cols() + c;
            if (input.data[idx] < 0)
                input.data[idx] = 0;
            else
                input.data[idx] = 1;
        }
    }
}


/**
 * A method which applies the softmax function to the passed matrix.
 *
 * @param input The matrix where we're applying the softmax to.
 */
void Ops_cpu::softmax(Mat& input) {
    //Apply the softmax to each element.
    type sum = 0;
    for (size_t r = 0; r < input.rows(); r++) {
        for (size_t c = 0; c < input.cols(); c++) {
            size_t idx = r * input.cols() + c;
            type exp = std::exp(input.data[idx]);
            sum += exp;
        }
    }

    for (size_t r = 0; r < input.rows(); r++) {
        for (size_t c = 0; c < input.cols(); c++) {
            size_t idx = r * input.cols() + c;
            type exp = std::exp(input.data[idx]);
            input.data[idx] = exp / sum;
        }
    }
}

}
