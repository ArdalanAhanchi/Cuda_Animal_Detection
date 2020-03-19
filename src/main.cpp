#include <iostream>
#include <string>
#include <vector>
#include <algorithm>
#include <random>

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

	//We can add more references to other resource files if time permitting
#ifdef _WIN32
    std::string projectDir = std::getenv("CSS535_PROJ");
	std::string oiDogResourceFile = projectDir + "\\images\\open-images\\Dog_oi_resource.windows.txt";
	std::string oiCatResourceFile = projectDir + "\\images\\open-images\\Cat_oi_resource.windows.txt";
	std::string oiTestResourceFile = projectDir + "\\images\\open-images\\test_oi_resource.windows.txt";
#elif linux
    std::string projectDir = std::string("");
	std::string oiDogResourceFile = "images/open-images/Dog_oi_resource.linux.txt";
	std::string oiCatResourceFile = "images/open-images/Cat_oi_resource.linux.txt";
	std::string oiTestResourceFile = "images/open-images/test_oi_resource.linux.txt";
#endif

	ImageHandler dogHandler(projectDir, oiDogResourceFile);
	ImageHandler catHandler(projectDir, oiCatResourceFile);
	ImageHandler testHandler(projectDir, oiTestResourceFile);
	int averageWidth = 0;
	int averageHeight = 0;
	std::vector<cv::Mat> transformedDogImages = dogHandler.applyTransforms();
	std::vector<cv::Mat> transformedCatImages = catHandler.applyTransforms();
	std::vector<cv::Mat> testImages = testHandler.parseRawImagesFromResource();
	
	dogHandler.getAverageSizes(transformedDogImages, averageWidth, averageHeight);

	if (transformedDogImages.size() > 0)
	{
		//transform test images to same size as dog images
		cv::Size desiredSize(transformedDogImages[0].size().width, transformedDogImages[0].size().height); //All of the dog images here should be the same size (from applyTransforms())
		std::vector<cv::Mat> resizedCatImages = catHandler.resizeImages(transformedCatImages, desiredSize);
		std::vector<cv::Mat> resizedTestImages = testHandler.resizeImages(testImages, desiredSize);

		std::vector<anr::Mat> dog_images = dogHandler.convertToInteralMat(transformedDogImages);
		std::vector<anr::Mat> misc_images = testHandler.convertToInteralMat(resizedTestImages);

		//Uncomment below for viewing the transform images and testing.
		/*for (int i = 0; i < transformedDogImages.size(); i++)
		{
			if (!transformedDogImages[i].empty())
			{
				cv::imshow("Dog Image", transformedDogImages[i]);
				cv::waitKey(0);
			}
		}*/
		

        //TODO: Add the data from test images and expected to the vectors.
       
        //Find out the minimum number of images.
        size_t min_images = (dog_images.size() < misc_images.size()
                ? dog_images.size() : misc_images.size());

        //Define vectors for training and expected data.
        std::vector<anr::Mat> training_data;
        std::vector<anr::Mat> expected_data;

        //Add the data from the dog and test images (every other one, so we have equal numbers).
        for(size_t i = 0; i < min_images; i++) {
            //Add the dog image and label.
            training_data.push_back(dog_images[i]);

            anr::Mat dog_expected(1, 2);
            dog_expected.at(0, 0) = 1.0;
            dog_expected.at(0, 1) = 0.0;
            expected_data.push_back(dog_expected);


            //Add the test image and label.
            training_data.push_back(misc_images[i]);

            anr::Mat misc_expected(1, 2);
            misc_expected.at(0, 0) = 0.0;
            misc_expected.at(0, 1) = 1.0;
            expected_data.push_back(misc_expected);
        }

        for(anr::Mat a: training_data) {
            std::cerr << a.rows() << " " << a.cols() << std::endl;
        }

        std::cerr << "I DO GET HERE 1" << std::endl;

        //Initialize the layer sizes.
        std::vector<size_t> layer_sizes;
        layer_sizes.push_back(training_data[0].rows() * training_data[0].cols());
        layer_sizes.push_back(10);
        layer_sizes.push_back(10);
        layer_sizes.push_back(10);
        //layer_sizes.push_back(500);
        //layer_sizes.push_back(500);
        //layer_sizes.push_back(500);
        layer_sizes.push_back(2);

        //Use CPU ops for now, and build the basic model.
        anr::Ops* ops = new anr::Ops_cpu;
        anr::Mlp nn(layer_sizes, ops, 0.7);

        std::cerr << "I DO GET HERE 2" << std::endl;

        //Calculate the dividing index (training data vs testing data).
        size_t divide_idx = (size_t)((float)training_data.size() * MLP_TRAINING_RATIO);

        //Traing the mlp for as many points as requested (by the training ratio).
        for(size_t t = 0; t < 1; t++)
            for (size_t i = 0; i < divide_idx; i++)
                nn.train(training_data[i], expected_data[i]);

        std::cerr << "I DO GET HERE 3" << std::endl;


        //Test the predictions, and print data.
        for (size_t i = divide_idx; i < training_data.size(); i++) {
            anr::Mat predicted = nn.predict(training_data[i]);
            expected_data[i].print("\nExpected");
            predicted.print("Prediction");
        }
	}
}
