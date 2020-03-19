#include <iostream>
#include <string>
#include <vector>

#include <opencv2/core.hpp>
#include <opencv2/imgcodecs.hpp>

#include "image_handler.hpp"
#include "mat.hpp"

//Temporary test for mlp.
#include "test_mlp.cpp"
#include "test_mat_ops.cpp"

#define MLP_TRAINING_RATIO 0.7

std::string type2str(int type) {
	std::string r;

	uchar depth = type & CV_MAT_DEPTH_MASK;
	uchar chans = 1 + (type >> CV_CN_SHIFT);

	switch (depth) {
	case CV_8U:  r = "8U"; break;
	case CV_8S:  r = "8S"; break;
	case CV_16U: r = "16U"; break;
	case CV_16S: r = "16S"; break;
	case CV_32S: r = "32S"; break;
	case CV_32F: r = "32F"; break;
	case CV_64F: r = "64F"; break;
	default:     r = "User"; break;
	}

	r += "C";
	r += (chans + '0');

	return r;
}

int main(int argc, char** argv)
{
    //Temporary testing of the mlp.
    //TODO: Remove and replace with proper testing methods.
    //return test_mlp();
    //test_mat();
    //return EXIT_SUCCESS;

	std::string projectDir = std::getenv("CSS535_PROJ");

	//We can add more references to other resource files if time permitting
#ifdef _WIN32
	std::string oiDogResourceFile = projectDir + "\\images\\open-images\\Dog_oi_resource.windows.txt";
	std::string oiTestResourceFile = projectDir + "\\images\\open-images\\test_oi_resource.windows.txt";
#elif linux
	std::string oiDogResourceFile = "images/open-images/Dog_oi_resource.linux.txt";
	std::string oiTestResourceFile = "images/open-images/test_oi_resource.linux.txt";
#endif

	//Uncommented the below for image loading and testing
	
	ImageHandler dogHandler(projectDir, oiDogResourceFile);
	ImageHandler testHandler(projectDir, oiTestResourceFile);
	std::vector<cv::Mat> transformedImages = dogHandler.applyTransforms();
	std::vector<anr::Mat> preparedImages = dogHandler.convertToInteralMat(transformedImages);

	std::vector<cv::Mat> testCvImages = testHandler.parseRawImagesFromResource();
	std::vector<anr::Mat> preparedTestImages = testHandler.convertToInteralMat(testCvImages);
	
/*
	if (transformedImages.size() > 0)
	{
		for (int i = 0; i < transformedImages.size(); i++)
		{
			if (!transformedImages[i].empty())
			{
				cv::imshow("Dog Image", transformedImages[i]);
				cv::waitKey(0);
			}
		}
	}
*/
	std::vector<anr::Mat> training_data;
    std::vector<anr::Mat> expected_data;

    //TODO: Add the data from test images and expected to the vectors.
    
        
    //Initialize the layer sizes.
    std::vector<size_t> layer_sizes;
    layer_sizes.push_back(training_data[0].rows() * training_data[0].cols());
    layer_sizes.push_back(500);
    layer_sizes.push_back(500);
    layer_sizes.push_back(500);
    layer_sizes.push_back(500);
    layer_sizes.push_back(2);

    //Use CPU ops for now, and build the basic model.
    anr::Ops* ops = new anr::Ops_cpu;
    anr::Mlp nn(layer_sizes, ops, 0.7);

    //Calculate the dividing index (training data vs testing data).
    size_t divide_idx = (size_t)((float) training_data.size() * MLP_TRAINING_RATIO);
    
    //Traing the mlp for as many points as requested (by the training ratio).
    for(size_t i = 0; i < divide_idx; i++)
        nn.train(training_data[i], expected_data[i]);


    //Test the predictions, and print data.
    for(size_t i = divide_idx; i < training_data.size(); i++) {
        anr::Mat predicted = nn.predict(training_data[i]);
        expected_data[i].print("Expected");
        predicted.print("Prediction");
    }
}
