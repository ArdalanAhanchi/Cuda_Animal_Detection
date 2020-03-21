/**
 *  A header file which includes the trigger points (from cpu, to gpu) for the hybrid
 *  ops class. It is used to trigger when the size of the matrix is larger than a certain
 *  size. Sometimes, the operation will be faster on CPU.
 *
 *  @author Ardalan Ahanchi
 *  @date   Winter 2020
 */

//Value if we never want to trigger to gpu.
#define NEVER_TRIGGER -1

//Trigger (to gpu) values for addition operator (vector and matrix modes).
#define ADD_M_TRIGGER 750
#define ADD_V_TRIGGER -1

//Trigger (to gpu) values for subtraction operator (vector and matrix modes).
#define SUB_M_TRIGGER 700
#define SUB_V_TRIGGER -1

//Trigger (to gpu) values for multiplication operator (vector and matrix modes).
#define MULT_M_TRIGGER 150
#define MULT_V_TRIGGER -1

//Trigger (to gpu) values for element multiplication operator (vector and matrix modes).
#define EMULT_M_TRIGGER 500
#define EMULT_V_TRIGGER -1

//Trigger (to gpu) values for scaling operator (vector and matrix modes).
#define SCALE_M_TRIGGER 250
#define SCALE_V_TRIGGER -1

//Trigger (to gpu) values for sigmoid operator (vector and matrix modes).
#define SIG_M_TRIGGER 250
#define SIG_V_TRIGGER -1

//Trigger (to gpu) values for sigmoid prime operator (vector and matrix modes).
#define DSIG_M_TRIGGER 200
#define DSIG_V_TRIGGER -1

//Trigger (to gpu) values for Relu operator (vector and matrix modes).
#define RELU_M_TRIGGER 400
#define RELU_V_TRIGGER -1

//Trigger (to gpu) values for Relu Prime operator (vector and matrix modes).
#define DRELU_M_TRIGGER 300
#define DRELU_V_TRIGGER -1
