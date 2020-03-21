/**
 *  A class which implements the ops interface for basic matrix operations on cpu and gpu.
 *  It uses the most optimized version based on the function.
 *
 *  @author Ardalan Ahanchi
 *  @date March 2020
 */

#include "ops_hybrid.cuh"
#include "ops_hybrid_triggers.cuh"

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
Mat Ops_hybrid::add(const Mat& a, const Mat& b) {
    //Check if we're in Vector mode.
    if(a.rows() == 1) {
        //Check if we reach the trigger, then run it in GPU mode.
        if(ADD_V_TRIGGER != NEVER_TRIGGER && a.cols() > ADD_V_TRIGGER)
            return ops_g.add(a, b);

    } else {          //Matrix mode.
        //Check if we reach the trigger, then run it in GPU mode.
        if(ADD_M_TRIGGER != NEVER_TRIGGER 
            && a.cols() > ADD_M_TRIGGER && a.rows() > ADD_M_TRIGGER)
            return ops_g.add(a, b);
    }

    //If we reach here, call ops on CPU since it didn't qualify for GPU.
    return ops_c.add(a, b);
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
    //Check if we're in Vector mode.
    if(a.rows() == 1) {
        //Check if we reach the trigger, then run it in GPU mode.
        if(SUB_V_TRIGGER != NEVER_TRIGGER && a.cols() > SUB_V_TRIGGER)
            return ops_g.sub(a, b);

    } else {          //Matrix mode.
        //Check if we reach the trigger, then run it in GPU mode.
        if(SUB_M_TRIGGER != NEVER_TRIGGER 
            && a.cols() > SUB_M_TRIGGER && a.rows() > SUB_M_TRIGGER)
            return ops_g.sub(a, b);
    }

    //If we reach here, call ops on CPU since it didn't qualify for GPU.
    return ops_c.sub(a, b);
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
   //Check if we're in Vector mode.
    if(a.rows() == 1) {
        //Check if we reach the trigger, then run it in GPU mode.
        if(MULT_V_TRIGGER != NEVER_TRIGGER && a.cols() > MULT_V_TRIGGER)
            return ops_g.mult(a, b);

    } else {          //Matrix mode.
        //Check if we reach the trigger, then run it in GPU mode.
        if(MULT_M_TRIGGER != NEVER_TRIGGER 
            && a.cols() > MULT_M_TRIGGER && a.rows() > MULT_M_TRIGGER)
            return ops_g.mult(a, b);
    }

    //If we reach here, call ops on CPU since it didn't qualify for GPU.
    return ops_c.mult(a, b);
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
    //Check if we're in Vector mode.
    if(a.rows() == 1) {
        //Check if we reach the trigger, then run it in GPU mode.
        if(EMULT_V_TRIGGER != NEVER_TRIGGER && a.cols() > EMULT_V_TRIGGER)
            return ops_g.e_mult(a, b);

    } else {          //Matrix mode.
        //Check if we reach the trigger, then run it in GPU mode.
        if(EMULT_M_TRIGGER != NEVER_TRIGGER 
            && a.cols() > EMULT_M_TRIGGER && a.rows() > EMULT_M_TRIGGER)
            return ops_g.e_mult(a, b);
    }

    //If we reach here, call ops on CPU since it didn't qualify for GPU.
    return ops_c.e_mult(a, b);
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
    //Check if we're in Vector mode.
    if(a.rows() == 1) {
        //Check if we reach the trigger, then run it in GPU mode.
        if(SCALE_V_TRIGGER != NEVER_TRIGGER && a.cols() > SCALE_V_TRIGGER)
            return ops_g.scale(a, scale);

    } else {          //Matrix mode.
        //Check if we reach the trigger, then run it in GPU mode.
        if(SCALE_M_TRIGGER != NEVER_TRIGGER 
            && a.cols() > SCALE_M_TRIGGER && a.rows() > SCALE_M_TRIGGER)
            return ops_g.scale(a, scale);
    }

    //If we reach here, call ops on CPU since it didn't qualify for GPU.
    return ops_c.scale(a, scale);
}


    
/**
 *  A method which applies the sigmoid function to the passed matrix.
 *
 *  @param input The matrix where we're applying the sigmoid to. 
 */
void Ops_hybrid::sigmoid(Mat& input) {
    //Check if we're in Vector mode.
    if(input.rows() == 1) {
        //Check if we reach the trigger, then run it in GPU mode.
        if(SIG_V_TRIGGER != NEVER_TRIGGER && input.cols() > SIG_V_TRIGGER)
            return ops_g.sigmoid(input);

    } else {          //Matrix mode.
        //Check if we reach the trigger, then run it in GPU mode.
        if(SIG_M_TRIGGER != NEVER_TRIGGER 
            && input.cols() > SIG_M_TRIGGER && input.rows() > SIG_M_TRIGGER)
            return ops_g.sigmoid(input);
    }

    //If we reach here, call ops on CPU since it didn't qualify for GPU.
    return ops_c.sigmoid(input);
}   


/**
 *  A method which applies the derivative of sigmoid function to the passed matrix.
 *
 *  @param input The matrix where we're applying the sigmoid to. 
 */
void Ops_hybrid::deriv_sigmoid(Mat& input) {
    //Check if we're in Vector mode.
    if(input.rows() == 1) {
        //Check if we reach the trigger, then run it in GPU mode.
        if(DSIG_V_TRIGGER != NEVER_TRIGGER && input.cols() > DSIG_V_TRIGGER)
            return ops_g.deriv_sigmoid(input);

    } else {          //Matrix mode.
        //Check if we reach the trigger, then run it in GPU mode.
        if(DSIG_M_TRIGGER != NEVER_TRIGGER 
            && input.cols() > DSIG_M_TRIGGER && input.rows() > DSIG_M_TRIGGER)
            return ops_g.deriv_sigmoid(input);
    }

    //If we reach here, call ops on CPU since it didn't qualify for GPU.
    return ops_c.deriv_sigmoid(input);
}


/**
 * A method which applies the relu function to the passed matrix.
 *
 * @param input The matrix where we're applying the sigmoid to. 
 */
void Ops_hybrid::relu(Mat& input) {
    //Check if we're in Vector mode.
    if(input.rows() == 1) {
        //Check if we reach the trigger, then run it in GPU mode.
        if(RELU_V_TRIGGER != NEVER_TRIGGER && input.cols() > RELU_V_TRIGGER)
            return ops_g.relu(input);

    } else {          //Matrix mode.
        //Check if we reach the trigger, then run it in GPU mode.
        if(RELU_M_TRIGGER != NEVER_TRIGGER 
            && input.cols() > RELU_M_TRIGGER && input.rows() > RELU_M_TRIGGER)
            return ops_g.relu(input);
    }

    //If we reach here, call ops on CPU since it didn't qualify for GPU.
    return ops_c.relu(input);
}


/**
 * A method which applies the derivative of the relu function to the passed matrix.
 *
 * @param input The matrix where we're applying the sigmoid to. 
 */
void Ops_hybrid::deriv_relu(Mat & input) {
    //Check if we're in Vector mode.
    if(input.rows() == 1) {
        //Check if we reach the trigger, then run it in GPU mode.
        if(DRELU_V_TRIGGER != NEVER_TRIGGER && input.cols() > DRELU_V_TRIGGER)
            return ops_g.deriv_relu(input);

    } else {          //Matrix mode.
        //Check if we reach the trigger, then run it in GPU mode.
        if(DRELU_M_TRIGGER != NEVER_TRIGGER 
            && input.cols() > DRELU_M_TRIGGER && input.rows() > DRELU_M_TRIGGER)
            return ops_g.deriv_relu(input);
    }

    //If we reach here, call ops on CPU since it didn't qualify for GPU.
    return ops_c.deriv_relu(input);
}

}
