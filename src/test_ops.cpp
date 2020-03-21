#include <iostream>
#include <chrono>
#include <vector>

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

void print_result(const char* op, const char* ops_mode, 
    double time, size_t rows, size_t cols) {

    std::cout << "test_ops: result: [mode]=" << ops_mode
        << " [rows]=" << rows << " [cols]=" << cols
        << " [op]=" << op << " [time]=" << time << std::endl;
}

void runner(anr::Ops* ops, const char* ops_mode, size_t rows, size_t cols) {
    //Create two matrices for the add / sub / e_mult / scale functions.
    anr::Mat a(rows, cols);
    a.randomize(-1, 1);
    anr::Mat b(rows, cols);
    b.randomize(-1, 1);

    //Test add, subtraction, element wise multiplication, and scaling.
    auto start = start_time();
    ops->add(a, b);
    print_result("add", ops_mode, calc_time(start), rows, cols);
    
    start = start_time();
    ops->sub(a, b);
    print_result("sub", ops_mode, calc_time(start), rows, cols);

    start = start_time();
    ops->e_mult(a, b);
    print_result("e_mult", ops_mode, calc_time(start), rows, cols);

    start = start_time();
    ops->scale(a, 0);
    print_result("scale", ops_mode, calc_time(start), rows, cols);

    //Create a third matrix for matrix matrix multiplication
    anr::Mat c(cols, rows);
    c.randomize(-1, 1);

    //Test multiplication.
    start = start_time();
    ops->mult(a, b);
    print_result("mult", ops_mode, calc_time(start), rows, cols);

    //Create a matrix for sigmoid, deriv_sigmoid, relu, deriv_relu.
    anr::Mat d(cols, rows);
    c.randomize(-1, 1);

    //Test sigmoid, deriv_sigmoid, relu, deriv_relu.
    start = start_time();
    ops->sigmoid(d);
    print_result("sigmoid", ops_mode, calc_time(start), rows, cols);

    start = start_time();
    ops->deriv_sigmoid(d);
    print_result("deriv_sigmoid", ops_mode, calc_time(start), rows, cols);

    start = start_time();
    ops->relu(d);
    print_result("relu", ops_mode, calc_time(start), rows, cols);

    start = start_time();
    ops->deriv_relu(d);
    print_result("deriv_relu", ops_mode, calc_time(start), rows, cols);
}

int main() {
    std::cerr << "Starting the operations test program" << std::endl;

    //Define the operations classes, and their mode names.
    std::vector<anr::Ops*> ops_classes;
    std::vector<const char*> ops_modes;

    ops_classes.push_back(new anr::Ops_cpu);
    ops_modes.push_back("cpu");

    ops_classes.push_back(new anr::Ops_gpu);
    ops_modes.push_back("gpu");

    ops_classes.push_back(new anr::Ops_hybrid);
    ops_modes.push_back("hybrid");
    

    //Call runner with different values.
    for(size_t n = 0; n < 2000; n += (n < 20 ? 2 : 50)) {
        //Run the runner in vector, and matrix modes.
        for(size_t i = 0; i < ops_classes.size(); i++) {
            runner(ops_classes[i], ops_modes[i], n, n);
            runner(ops_classes[i], ops_modes[i], 1, n);
        }
    }

    //Go through each type of operation, and deallocate the objects
    for(anr::Ops* ops: ops_classes)
        delete ops;
    
    
}
