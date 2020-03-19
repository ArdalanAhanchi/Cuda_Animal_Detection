#ifndef CNN_HPP
#define CNN_HPP

#include "mat.hpp"

#include <vector>

namespace anr {

    /**
     *  The definition for a class which implements a convolutional neural network. There is
     *  one input layer, one output layer, and the number of hidden layers can be variable.
     *  (However, that feature is not yet fully implemented, right now it's only 1 layer).
     *
     *  @author Rahil Mehta
     *  @date March 2020
     */
    class Cnn {
    public:
        /**
         *  Constructor which initializes the network with the proper values. The
         *  initialization depends on the sizes passed to this constructor. The layer sizes
         *  should at least have 3 values (one input, one hidden, and one output layer).
         *
         *  @param layer_sizes A vector which holds the sizes for each layer (in/hiddens/out).
         *  @param rate The rate the learning for this specific network.
         */
        Cnn();


        /**
         *  Destructor which deallocates the layers array.
         */
        ~Cnn();

        /*
            A method for performing the convolution operation on the matrix of pixel values.
        */
        Mat convolution(Mat& images, type* kernel, int kernelSizeX, int kernelSizeY);

        Mat maxpool(Mat& input, int kernelSize, int stride);

    private:
        
    };

}

#endif
