#include <iostream>
#include <chrono>

#include "mat.hpp"
#include "ops_cpu.hpp"
#include "ops_gpu.cuh"
#include "ops.hpp"

/*
auto start_time()
{
    return std::chrono::high_resolution_clock::now();
}

double calc_time(std::chrono::high_resolution_clock::time_point begin)
{
    auto total_time = std::chrono::high_resolution_clock::now() - begin;
    double time = std::chrono::duration<double>(total_time).count();
    return time;
}
*/

void cmp_gpu_cpu() {
    std::cout << "Running the matrix test compare GPU/CPU program." << std::endl;

    //Capture the beginning time before the calculations.
    //auto begin_calcs = start_time();

    anr::Mat a(3, 6);
    a.randomize(-10, 10);

    //double time = calc_time(begin_calcs);
    //std::cout << time << std::endl;
    a.print("Matrix a");

    anr::Mat b(6, 3);
    b.randomize(-10, 10);

    b.print("\nMatrix b");

    anr::Mat c(b, true);
    c.print("\nMatrix c");

    anr::Ops_cpu ops_cpu;
    anr::Ops* ops_c = &ops_cpu;
    
    anr::Ops_gpu ops_gpu;
    anr::Ops* ops_g = &ops_gpu;



    anr::Mat d = ops_c->add(a, c);
    d.print("\nAdded a + c CPU");

    d = ops_g->add(a, c);
    d.print("\nAdded a + c GPU");



    anr::Mat e = ops_c->sub(a, c);
    e.print("\nSubbed a - c CPU");
    
    e = ops_g->sub(a, c);
    e.print("\nSubbed a - c GPU");



    anr::Mat f = ops_c->scale(a, 9);
    f.print("\nScaled a * 9 CPU");

    f = ops_g->scale(a, 9);
    f.print("\nScaled a * 9 GPU");



    anr::Mat g = ops_c->mult(a, b);
    g.print("\nMultiplied a * b CPU");

    g = ops_g->mult(a, b);
    g.print("\nMultiplied a * b GPU");


    anr::Mat h(1, 1);
    h = ops_c->scale(a, 9);
    h.print("\nAssigned");

    
    anr::Mat i(3, 7);
    i.randomize(-3, 3);

    anr::Mat i_cp(i, false);

    ops_c->sigmoid(i);
    i.print("\nAfter Sigmoid CPU");

    ops_g->sigmoid(i_cp);
    i_cp.print("\nAfter Sigmoid GPU");



    anr::Mat j(4, 3);
    j.randomize(-20, 20);

    anr::Mat j_cp(j, false);

    ops_c->deriv_sigmoid(j);
    j.print("\nAfter Sigmoid Prime CPU");

    ops_g->deriv_sigmoid(j_cp);
    j_cp.print("\nAfter Sigmoid Prime GPU");
    

    anr::Mat k(1, 10);
    k.randomize(-10, 10);
    anr::Mat l(10, 3);
    l.randomize(-5, 5);
    
    ops_c->mult(k, l).print("\nAfter multiplication CPU");
    ops_g->mult(k, l).print("\nAfter multiplication GPU");


    anr::Mat m(1, 100);
    m.randomize(-10, 10);
    anr::Mat n(1, 100);
    n.randomize(-5, 5);
    
    ops_c->add(m, n).print("\nAfter addition CPU");
    ops_g->add(m, n).print("\nAfter addition GPU");
}

void test_mat() {
    std::cout << "Running the matrix test program" << std::endl; 

    //Capture the beginning time before the calculations.
    //auto begin_calcs = start_time();

    anr::Mat a(3, 6);
    for(size_t i = 0; i < a.rows() * a.cols(); i++)
        a.data[i] = 2;

    //double time = calc_time(begin_calcs);
    //std::cout << time << std::endl;
    a.print("Matrix a");

    anr::Mat b(6, 3);
    for(size_t i = 0; i < b.rows() * b.cols(); i++)
        b.data[i] = 3;

    b.print("\nMatrix b");

    anr::Mat c(b, true);
    c.print("\nMatrix c");

    anr::Ops_cpu ops_cpu;
    anr::Ops* ops = &ops_cpu;
    //anr::Ops_gpu ops_gpu;
    //anr::Ops* ops = &ops_gpu;

    anr::Mat d = ops->add(a, c);
    d.print("\nAdded a + c");

    anr::Mat e = ops->sub(a, c);
    e.print("\nSubbed a - c");

    anr::Mat f = ops->scale(a, 9);
    f.print("\nScaled a * 9");

    anr::Mat g = ops->mult(a, b);
    g.print("\nMultiplied a * b");

    anr::Mat h(1, 1);
    h = ops->scale(a, 9);
    h.print("\nAssigned");

    
    anr::Mat i(3, 7);
    i.randomize(-30, 30);
    i.print("\nBefore Sigmoid");

    ops->sigmoid(i);
    i.print("\nAfter Sigmoid");
}

int main() {
    std::cerr << "Starting the matrix test program" << std::endl;
    cmp_gpu_cpu();  
    test_mat();
}
