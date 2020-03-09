#include <iostream>

#include "mat.hpp"
#include "ops_cpu.hpp"
#include "ops.hpp"

void test_mat() {
    std::cout << "Running the matrix test program" << std::endl;

    anr::Mat a(3, 6);
    for(size_t i = 0; i < a.rows() * a.cols(); i++)
        a.data[i] = 2;

    a.print("Matrix a");

    anr::Mat b(6, 3);
    for(size_t i = 0; i < b.rows() * b.cols(); i++)
        b.data[i] = 3;

    b.print("Matrix b");

    anr::Mat c(b, true);
    c.print("\nMatrix c");

    anr::Ops_cpu ops_cpu;
    anr::Ops* ops = &ops_cpu;

    anr::Mat d = ops->add(a, c);
    d.print("\nAdded a + c");

    anr::Mat e = ops->sub(a, c);
    e.print("\nSubbed a - c");

    anr::Mat f = ops->scale(a, 9);
    f.print("\nScaled a * 9");

    anr::Mat g = ops->mult(a, b);
    g.print("\nMultiplied a * b");
}
