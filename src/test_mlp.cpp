#include <iostream>
#include <vector>

#ifndef TEST_MLP
#define TEST_MLP

#include "mat.hpp"
#include "ops_cpu.hpp"
#include "ops.hpp"
#include "mlp.hpp"

/**
 *  A file for testing the implementation of MLP using two diffferent functions/
 *
 *  @author Ardalan Ahanchi
 *  @date March 2020
 */


/**
 *  A function which trains a mlp model to represent the XOR function. It then tests and 
 *  prints results. It trains for 5000 times to get a decent accuracy.
 */
void test_mlp_xor(anr::Ops* ops) {
    std::cout << "Running the mlp test XOR program." << std::endl;

    std::vector<anr::Mat> training_data;
    std::vector<anr::Mat> expected_data;

    //Initialize data for the XOR operation.
    for(size_t i = 0; i < 4; i++) {
        //Calculate each pair based on index (00, 01, 10, 11)
        float a = i / 2, b = i % 2;

        //Add them to an input matrix.
        anr::Mat input(1, 2);
        input.at(0, 0) = a;
        input.at(0, 1) = b;
        training_data.push_back(input);

        //Calculate the xor and save the expected.
        anr::Mat expected(1, 1);
        expected.at(0, 0) = ((a || b) && (! (a && b)));
        expected_data.push_back(expected);  
    }
        
    //Initialize the layer sizes.
    std::vector<size_t> layer_sizes;
    layer_sizes.push_back(2);
    layer_sizes.push_back(5);
    layer_sizes.push_back(4);
    layer_sizes.push_back(1);

    anr::Mlp nn(layer_sizes, ops, 0.8);
    
    //Traing the mlp.
    for(size_t t = 0; t < 5000; t++)
        for(size_t i = 0; i < training_data.size(); i++)
            nn.train(training_data[i], expected_data[i]);

    std::cerr << "\n* Results XOR *****************************************" << std::endl;

    //Test the predictions, and print data.
    for(anr::Mat curr: training_data) {
        anr::Mat predicted = nn.predict(curr);
        
        curr.print("\nInput");
        predicted.print("\nPrediction");
    }
}


/**
 *  A function which trains a mlp model to represent a linear function. It then tests and 
 *  prints results. It trains for 2000 times to get a decent accuracy.
 */
void test_mlp_lin(anr::Ops* ops) {
std::cout << "Running the mlp test linear program." << std::endl;

    std::vector<anr::Mat> training_data;
    std::vector<anr::Mat> expected_data;

    //Initialize 3000 points of data for the custom function (5x + 2y + z > 4).
    for(size_t i = 0; i < 3000; i++) {
        //Create an input node, and randomize it (values 0-1).
        anr::Mat input(1, 3);
        input.randomize(0.0, 1.0);
        training_data.push_back(input);

        //Calculate the result and save the expected based on the linear function.
        bool is_larger = 
            ((5.0 * input.get(0, 0)) + (2.0 * input.get(0, 1)) + input.get(0, 2)) > 4.0;

        anr::Mat expected(1, 2);
        expected.at(0, 0) = (is_larger ? 1.0 : 0.0);
        expected.at(0, 1) = (is_larger ? 0.0 : 1.0);
        expected_data.push_back(expected);  
    }
        
    //Initialize the layer sizes.
    std::vector<size_t> layer_sizes;
    layer_sizes.push_back(3);
    layer_sizes.push_back(7);
    layer_sizes.push_back(9);
    layer_sizes.push_back(2);

    anr::Mlp nn(layer_sizes, ops, 0.8);
    
    //Traing the mlp for 2000 of the points.
    for(size_t i = 0; i < 2000; i++)
        nn.train(training_data[i], expected_data[i]);

    std::cerr << "\n* Results (5X + 2Y + Z) > 4 ***************************" << std::endl;

    //Test the predictions, and print data.
    for(size_t i = 2000; i < 3000; i++) {
        anr::Mat predicted = nn.predict(training_data[i]);
        
        training_data[i].print("\n\nInput");
        expected_data[i].print("Expected");
        predicted.print("Prediction");
    }
}


/**
 *  A function which is the starting point of the MLP testing program. It trains and tests
 *  two networks, one with the XOR function, and one with a custom linear function.
 *
 *  @return EXIT_SUCCESS at the end of execution.
 */
int test_mlp() {
    //Define the ops class we're gonna use.
    anr::Ops* ops = new anr::Ops_cpu;

    //Test both functions.
    test_mlp_xor(ops);
    test_mlp_lin(ops);

    return EXIT_SUCCESS;
}

#endif
