/**
 *  The implementation for a class which implements a basic multi-layer perceptron. There
 *  is one input layer, one output layer, and the number of hidden layers can be variable.
 *  (However, that feature is not yet fully implemented, right now it's only 1 layer).
 *
 *  @author Ardalan Ahanchi
 *  @date March 2020
 */

#include "mlp.hpp"

#include "mat.hpp"
#include "ops.hpp"
#include "ops_cpu.hpp"

#include <vector>
#include <iostream>

namespace anr {

//Min/Max numbers used for random number generation.
#define MAT_RAND_MAX 0.50
#define MAT_RAND_MIN -0.50

//Min/Max values for Step Down/Up functionality.
#define STEP_HIGH 0.9
#define STEP_LOW 0.1

//Total value Min/Max.
#define VALUE_MAX 1.0
#define VALUE_MIN 0.0

/**
 *  Constructor which initializes the network with the proper values. The
 *  initialization depends on the sizes passed to this constructor. The layer sizes
 *  should at least have 3 values (one input, one hidden, and one output layer).
 *
 *  @param layer_sizes A vector which holds the sizes for each layer (in/hiddens/out).
 *  @param rate The rate the learning for this specific network.
 */
Mlp::Mlp(const std::vector<size_t>& layer_sizes, const type& rate) {
    //Set the rate in the network.
    this->_rate = rate;

    //Check the hidden layer sizes, it should be at least one.
    if(layer_sizes.size() < 3)
        throw "Mlp: Error: There should be at least three layers (input/hidden/output).";

    //Initialize all the class' dynamic arrays.
    this->_num_layers = layer_sizes.size();
    this->_layers = new Mat[this->_num_layers];
    this->_weights = new Mat[this->_num_layers - 1];
    this->_biases = new Mat[this->_num_layers - 1];
    this->_weight_gradients = new Mat[this->_num_layers - 1];
    this->_bias_gradients = new Mat[this->_num_layers - 1];

    //Go through all layers and add the weights and biases. Then randomize them.
    for(size_t i = 0; i < layer_sizes.size() - 1; i++) {
        //Add the weights matrix to the weights, and randomize it.
        this->_weights[i] = Mat(layer_sizes[i], layer_sizes[i + 1]);
        this->_weights[i].randomize(MAT_RAND_MIN, MAT_RAND_MAX);

        //Add the biases matrix to the biases, and randomize it.
        this->_biases[i] = Mat(1, layer_sizes[i + 1]);
        this->_biases[i].randomize(MAT_RAND_MIN, MAT_RAND_MAX);
    }
}

/**
 *  Destructor which deallocates the layers array.
 */
Mlp::~Mlp() {
    //Deallocate all the dynamic arrays.
    delete[] this->_layers;
    delete[] this->_weights;
    delete[] this->_biases;
    delete[] this->_weight_gradients;
    delete[] this->_bias_gradients;
}


/**
 *  A method which trains the initialized model based on the given input, and the
 *  expected output. This is the starting point of all calculations on this mlp.
 *
 *  @param input The input matrix (vector in this case) of size input_size. 
 *  @param expected The expected output (perfect case) for training the model.
 */
void Mlp::train(Mat& input, Mat& expected) {
    //Check if input or expected are invalid sizes.
    if(input.rows() != 1 || input.cols() == 0 || expected.rows() != 1 || expected.cols() == 0)
        throw "Mlp: Error: Invalid input or expected vectors passed (num_rows == 1).";

    //Check if we're not set up.
    if(this->_num_layers == 0)
        throw "Mlp: Error: The network was not setup properly.";

    //Place a shallow copy of input at the first element of the layers.
    this->_layers[0] = input;

    //Initialize the operations class (will be changed based on the mode in the future).
    Ops_cpu ops_cpu;
    Ops* ops = &ops_cpu;

    /* Forward Prop *********************************************************************/
    this->forward(input);

    /** Back Prop ***********************************************************************/

    //Go through the layers in reverse order and compute the gradients (up to the output).
    //It also updates the weights based on the calculations (back-prop).
    for(size_t i = this->_num_layers - 1; i > 0 ; i--) {
        Mat bias_gradient_temp, temp;

        //If we're on the last layer, use the difference.
        if(i == this->_num_layers - 1) {
            //Find the difference between expected and output.
            temp = ops->sub(this->_layers[i], expected);
        } else {
            //Otherise multiply the last gradient by last weight (transposed).
            Mat transposed_w(this->_weights[i], true);
            temp = ops->mult(this->_bias_gradients[i], transposed_w);
        }

        //Set it to matrix multiplication of last layer, and it's weights.
        bias_gradient_temp = ops->mult(this->_layers[i - 1], this->_weights[i - 1]);

        //Add the biases to it.
        bias_gradient_temp = ops->add(bias_gradient_temp, this->_biases[i - 1]);

        //Apply the derivative of sigmoid to it.
        ops->deriv_sigmoid(bias_gradient_temp);

        //Multiply element by element and set the bias gradient.
        this->_bias_gradients[i - 1] = ops->e_mult(temp, bias_gradient_temp);

        //Transpose the last layer and save it.
        Mat weight_gradient_temp(this->_layers[i - 1], true);

        //Multiply it by the bias gradiant and update weight gradients.
        this->_weight_gradients[i - 1] = 
            ops->mult(weight_gradient_temp, this->_bias_gradients[i - 1]);
    }

    //Go through all and update weights and biases based on the learning rate.
    for(size_t i = 0; i < this->_num_layers - 1; i++) {
        //Scale the weight and bias gradients by the learning rate.
        Mat scaled_weight_g = ops->scale(this->_weight_gradients[i], this->_rate);
        Mat scaled_bias_g = ops->scale(this->_bias_gradients[i], this->_rate);

        //Update the weights and biases.
        this->_weights[i] = ops->sub(this->_weights[i], scaled_weight_g);
        this->_biases[i] = ops->sub(this->_biases[i], scaled_bias_g);
    }

    /********************************************************************************/
}

/** 
 *  A method which predicts the classification value based on the given input.
 *  it should only be run when the network is fully trained.
 *
 *  @param input The data input matrix.
 *  @return The classification result.
 */
Mat Mlp::predict(Mat& input) {
    //Perform forward prop based on the input.
    this->forward(input);

    //Store a shallow copy of the output matrix.
    Mat output = this->_layers[this->_num_layers - 1];

    //Apply the stepping function, and return the results.
    for(size_t r = 0; r < output.rows(); r++) {
        for(size_t c = 0; c < output.cols(); c++) {
            //Compare the output and step it up/down accordingly.
            if(output.get(r, c) > STEP_HIGH)
                output.at(r, c) = VALUE_MAX;
            else if(output.get(r, c) < STEP_LOW)
                output.at(r, c) = VALUE_MIN;
        }
    }

    //Return the results.
    return output;    
}

/**
 *  A method which performs forward propogation in the network. It is used both in
 *  training and prediction.
 *
 *  @param input The current input for prop, will be placed at the 0th layer.
 */
void Mlp::forward(Mat& input) {
    //Initialize the operations class (will be changed based on the mode in the future).
    Ops_cpu ops_cpu;
    Ops* ops = &ops_cpu;

    //Place a shallow copy of input at the first element of the layers.
    this->_layers[0] = input;

    //Go through the layers and compute the layers (up to the output).
    for(size_t i = 0; i < this->_num_layers - 1; i++) {
        Mat layer;
        layer = ops->mult(this->_layers[i], this->_weights[i]);     //Multiply the weights.
        layer = ops->add(layer, this->_biases[i]);                  //Add the biases.
        ops->sigmoid(layer);                                        //Apply sigmoid.

        //Assign the layer build to the correct index.
        this->_layers[i + 1] = layer;
    }
}


/**
 *  A method which prints this neural network in a nice easy to read format. 
 *  it prints the current state of the system.
 */
void Mlp::print() const {
    std::cerr << "* MLP ***********************************************\n" << std::endl;
    
    //Print the input layer.
    this->_layers[0].print("Input Layer");

    //Print the weights, biases, and next layers.
    for(size_t i = 0; i < this->_num_layers - 1; i++) {
        std::cerr << std::endl << "* Layer " << i + 1 <<  " *****************" << std::endl;
        this->_weights[i].print("\nWeights");
        this->_biases[i].print("\nBiases");
        this->_layers[i + 1].print("\nLayer");
        std::cerr << std::endl << "***************************" << std::endl;
    }

    std::cerr << "\n*****************************************************\n" << std::endl;
}



}
