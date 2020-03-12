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

namespace anr {

#define MAT_RAND_MIN 0.01
#define MAT_RAND_MAX 1.00

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
    if(input.rows() != 1 || input.cols() == 0 || expected.rows() != 1 || expected.cols() != 0)
        throw "Mlp: Error: Invalid input or expected vectors passed (num_rows == 1).";

    //Check if we're not set up.
    if(this->_num_layers == 0)
        throw "Mlp: Error: The network was not setup properly.";

    //Place a shallow copy of input at the first element of the layers.
    this->_layers[0] = input;

    //Initialize the operations class (will be changed based on the mode in the future).
    Ops_cpu ops_cpu;
    Ops* ops = &ops_cpu;
    

    //Go through the layers and compute the layers (up to the output).
    for(size_t i = 0; i < this->_num_layers - 1; i++) {
        Mat layer;
        layer = ops->mult(this->_layers[i], this->_weights[i]);     //Multiply the weights.
        layer = ops->add(this->_layers[i], this->_biases[i]);       //Add the biases.
        ops->sigmoid(layer);                                        //Apply sigmoid.

        //Assign the layer build to the correct index.
        this->_layers[i + 1] = layer;
    }


    /** Back prop Not working yet. *******************************************************

    //Go through the layers in reverse order and compute the gradients (up to the output).
    //It also updates the weights based on the calculations (back-prop).
    for(size_t i = this->_num_layers - 1; i >= 0 ; i--) {
        Mat weight_gradient, bias_gradient;


        //Set it to multiplication of last layer, and it's weights.
        bias_gradient = ops->mult(this->_layers[i - 1], this->_weights[i - 1]);

        //Add the biases to it.
        bias_gradient = ops->add(bias_gradient, this->_biases[i - 1]);

        //Apply the derivative of sigmoid to it.
        ops->deriv_sigmoid(bias_gradient);

        //Find the difference between the layers.
        Mat difference = ops->sub(this->_layers[i], expected);

        layer = ops->mult(this->_layers[i], this->_weights[i]);     //Multiply the weights.
        layer = ops->add(this->_layers[i], this->_biases[i]);       //Add the biases.
        ops->sigmoid(layer);                                        //Apply sigmoid.

        //Assign the layer build to the correct index.
        this->_layers[i + 1] = layer;
    }

    *************************************************************************************/
}

}