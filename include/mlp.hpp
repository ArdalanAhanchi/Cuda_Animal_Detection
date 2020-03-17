#ifndef MLP_HPP
#define MLP_HPP

#include "mat.hpp"

#include <vector>

namespace anr {

/**
 *  The definition for a class which implements a basic multi-layer perceptron. There is 
 *  one input layer, one output layer, and the number of hidden layers can be variable.
 *  (However, that feature is not yet fully implemented, right now it's only 1 layer).
 *
 *  @author Ardalan Ahanchi
 *  @date March 2020
 */
class Mlp {
    /**
     *  Constructor which initializes the network with the proper values. The
     *  initialization depends on the sizes passed to this constructor. The layer sizes
     *  should at least have 3 values (one input, one hidden, and one output layer).
     *
     *  @param layer_sizes A vector which holds the sizes for each layer (in/hiddens/out).
     *  @param rate The rate the learning for this specific network.
     */
    Mlp(const std::vector<size_t>& layer_sizes, const type& rate);

    
    /**
     *  Destructor which deallocates the layers array.
     */
    ~Mlp();
    
    
    /**
     *  A method which trains the initialized model based on the given input, and the
     *  expected output. This is the starting point of all calculations on this mlp.
     *
     *  @param input The input matrix (vector in this case) of size input_size. 
     *  @param expected The expected output (perfect case) for training the model.
     */
    void train(Mat& input, Mat& expected);


    /** 
     *  A method which predicts the classification value based on the given input.
     *  it should only be run when the network is fully trained.
     *
     *  @param input The data input matrix.
     *  @return The classification result.
     */
    Mat predict(Mat& input);
    

    
private:

    Mat* _layers;                           /**< Array of All layers (current values). */
    size_t _num_layers;                     /**< The total number of layers. */

    Mat* _weights;                          /**< Array of Weights for each neuron. */
    Mat* _biases;                           /**< Array of Biases for each neuron. */
    Mat* _weight_gradients;                 /**< Array of Gradients for the weights. */
    Mat* _bias_gradients;                   /**< Arrau of Gradients for the biases. */

    type _rate;                             /**< Network's current learning rate. */
};

}

#endif
