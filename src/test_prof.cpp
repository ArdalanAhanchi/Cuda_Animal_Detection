#include <iostream>
#include <vector>

#include "mat.hpp"
#include "ops_gpu.cuh"
#include "ops.hpp"

/**
 *  A program which utilizes the GPU for super large matrix operations.
 *  it is used to examine the usage results from the profiler.  
 *
 *  @author Ardalan Ahanchi
 *  @date March 2020
 */


//A function which runs the test based on the passed rows/cols.
void runner(size_t rows, size_t cols) {
    //For testing the gpu.    
    anr::Ops_gpu ops; 

    //Create two matrices for the add / sub / e_mult / scale functions.
    anr::Mat a(rows, cols);
    a.randomize(-1, 1);
    anr::Mat b(rows, cols);
    b.randomize(-1, 1);

    //Test add, subtraction, element wise multiplication, and scaling.
    ops.add(a, b);
    ops.sub(a, b);
    ops.e_mult(a, b);
    ops.scale(a, 0);

    //Create a third matrix for matrix matrix multiplication
    anr::Mat c(cols, rows);
    c.randomize(-1, 1);

    //Test multiplication.
    ops.mult(a, b);

    //Create a matrix for sigmoid, deriv_sigmoid, relu, deriv_relu.
    anr::Mat d(cols, rows);
    c.randomize(-1, 1);

    //Test sigmoid, deriv_sigmoid, relu, deriv_relu.
    ops.sigmoid(d);
    ops.deriv_sigmoid(d);
    ops.relu(d);
    ops.deriv_relu(d);
}

int main() {
    std::cerr << "Starting the GPU profiler test program" << std::endl;
    
    //Test it with super large matrices (N=6000)
    runner(6000, 6000);

    //Test it with the same sizes vector (N=6000)
    runner(1, 6000);    

    return EXIT_SUCCESS;
}
