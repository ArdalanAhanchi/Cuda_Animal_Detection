#include "cnn.hpp"
/**
 *  The implementation for a class which implements the convolution layers. 
 *
 *  @author Rahil Mehta
 *  @date March 2020
 */

#include "mat.hpp"
#include "ops.hpp"
#include "ops_cpu.hpp"

#include <vector>
#include <math.h>         //For fabs
#include <climits>        //For INT_MIN 

namespace anr {

#define MAT_RAND_MIN 0.01
#define MAT_RAND_MAX 1.00

    Cnn::Cnn() {
        
    }

    Cnn::~Cnn() {
        
    }

    /**
     *  A method which performs convolutions on the images by applying a kernel.
     *
     *  @param images The input matrix of pixel values.
     *  @param kernel The filter to apply.
     */
    Mat Cnn::convolution(Mat& images, type* kernel, Mat& output, Mat& bias, int kernelSizeX, int kernelSizeY)
    {
        int i, j, m, n, mm, nn;
        int kCenterX, kCenterY;                         // center index of kernel
        float sum;                                      // temp accumulation buffer
        int rowIndex, colIndex;

        // find center position of kernel (half of kernel size)
        kCenterX = kernelSizeX / 2;
        kCenterY = kernelSizeY / 2;

        const int dx = kernelSizeX / 2;
        const int dy = kernelSizeY / 2;

        for (i = 0; i < images.rows(); ++i)                // rows
        {
            for (j = 0; j < images.cols(); ++j)            // columns
            {
                sum = 0;                            // init to 0 before sum
                for (m = 0; m < kernelSizeY; ++m)      // kernel rows
                {
                    mm = kernelSizeY - 1 - m;       // row index of flipped kernel

                    for (n = 0; n < kernelSizeX; ++n)  // kernel columns
                    {
                        nn = kernelSizeX - 1 - n;   // column index of flipped kernel

                        // index of input signal, used for checking boundary
                        rowIndex = i + (kCenterY - mm);
                        colIndex = j + (kCenterX - nn);

                        // ignore input samples which are out of bound
                        if (rowIndex >= 0 && rowIndex < images.rows() && colIndex >= 0 && colIndex < images.cols())
                            sum += images.data[images.cols() * rowIndex + colIndex] * kernel[kernelSizeX * mm + nn];
                    }
                }

            }
        }
        return output;
    }
    
    /*
        Perform the maxpooling operation for downsampling.
    */
    Mat Cnn::maxpool(Mat& input, int window, int stride)
    {
        // calculate output dimensions after the maxpooling operation.
        int h = int((input.rows() - window) / stride) + 1;
        int w = int((input.cols() - window) / stride) + 1;

        Mat out(h, w);
        for (int i = 0; i < h * w; i++)
        {
            int curY = 0;
            int outY = 0;
            while (curY + window < input.rows())
            {
                int curX = 0;
                int outX = 0;
                while (curX + window < input.cols())
                {
                    int curX = 0;
                    int outX = 0;
                    int max = INT_MIN;
                    for (int y = curY; y < curY + window; y++)
                    {
                        for (int x = curX; x < curX + window; x++)
                        {
                            if (input.data[y * input.rows() + x] > INT_MIN)
                            {
                                
                                max = input.data[y * input.rows() + x];
                            }
                        }
                    }
                    out.data[outY * out.rows() + outX] = max;
                    curX += stride;
                    outX += 1;
                }
                curY += stride;
                outY += 1;
            }
        }
        return out;
    }

}
