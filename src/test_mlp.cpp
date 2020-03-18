#include <iostream>
#include <vector>

#ifndef TEST_MLP
#define TEST_MLP

#include "mat.hpp"
#include "ops_cpu.hpp"
#include "ops.hpp"
#include "mlp.hpp"

/**
 *  A function which trains a nn model to represent the XOR function. It then tests and 
 *  prints results. It trains for 5000 times to get a decent accuracy.
 *
 *  @author Ardalan Ahanchi
 *  @return EXIT_SUCCESS at the end of execution.
 */
int test_mlp() {
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

    anr::Mlp nn(layer_sizes, 0.8);
    
    //Traing the mlp.
    for(size_t t = 0; t < 5000; t++)
        for(size_t i = 0; i < training_data.size(); i++)
            nn.train(training_data[i], expected_data[i]);

    std::cerr << "\n* Results *********************************************" << std::endl;

    //Test the predictions, and print data.
    for(anr::Mat curr: training_data) {
        anr::Mat predicted = nn.predict(curr);
        
        curr.print("\nInput");
        predicted.print("\nPrediction");
    }
            

    return EXIT_SUCCESS; 
}

#endif
