#include <iostream>
#include <chrono>

#include "mat.hpp"
#include "ops_cpu.hpp"
#include "ops_gpu.cuh"
#include "ops_hybrid.cuh"
#include "ops.hpp"

/**
 *  A program which compares the cpu, gpu, and hybrid implementations of 
 *  varirous algorithms with various sizes of matrices. 
 *
 *  @author Ardalan Ahanchi
 *  @date March 2020
 */


auto start_time() {
    return std::chrono::high_resolution_clock::now();
}

double calc_time(std::chrono::high_resolution_clock::time_point begin) {
    auto total_time = std::chrono::high_resolution_clock::now() - begin;
    double time = std::chrono::duration<double>(total_time).count();
    return time;
}
/*
void runner(Ops* ops, size_t rows, size_t cols) {
    //Create two matrices for the add / sub / e_mult / scale functions.
    anr::Mat 
    ops->add(a, c);
    ops->sub(a, c);
    ops->scale(a, 9);

    //Create a third matrix for matrix matrix multiplication
    ops->mult(a, b);

    //Create matrices for sigmoid, deriv_sigmoid, relu, deriv_relu.

    ops->sigmoid(i);
    ops->deriv_sigmoid(i);
    ops->relu(i);
    ops->deriv_relu(i);

}
*/
int main() {
    std::cerr << "Starting the operations test program" << std::endl;
    //TODO: call runner();
}
