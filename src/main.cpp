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
    return test_mlp();
    //test_mat();
    //return EXIT_SUCCESS;

	std::string projectDir = std::getenv("CSS535_PROJ");

	//We can add more references to other resource files if time permitting
#ifdef _WIN32
	std::string oiDogResourceFile = projectDir + "\\images\\open-images\\Dog_oi_resource.windows.txt";
	std::string oiTestResourceFile = projectDir + "\\images\\open-images\\test_oi_resource.windows.txt";
#elif linux
	std::string oiDogResourceFile = projectDir + "/images/open-images/Dog_oi_resource.linux.txt";
	std::string oiTestResourceFile = projectDir + "/images/open-images/test_oi_resource.linux.txt";
#endif

	//Uncommented the below for image loading and testing
	/*
	ImageHandler dogHandler(projectDir, oiDogResourceFile);
	ImageHandler testHandler(projectDir, oiTestResourceFile);
	std::vector<cv::Mat> transformedImages = dogHandler.applyTransforms();
	std::vector<anr::Mat> preparedImages = dogHandler.convertToInteralMat(transformedImages);

	std::vector<cv::Mat> testCvImages = testHandler.parseRawImagesFromResource();
	std::vector<anr::Mat> preparedTestImages = testHandler.convertToInteralMat(testCvImages);
	

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
}
