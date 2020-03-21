#include <iostream>
#include <string>
#include <vector>
#include <algorithm>
#include <random>

#include <opencv2/core.hpp>
#include <opencv2/imgcodecs.hpp>

#include "image_handler.hpp"
#include "mat.hpp"
#include "ops.hpp"
#include "ops_cpu.hpp"
#include "ops_gpu.cuh"
#include "ops_hybrid.cuh"
#include "mlp.hpp"
#include "cnn.hpp"

#define MLP_TRAINING_RATIO 0.8
#define NUM_IMAGES 200

// define 5x5 Gaussian kernel
anr::type kernel[25] = { 1 / 256.0,  4 / 256.0,  6 / 256.0,  4 / 256.0, 1 / 256.0,
                     4 / 256.0, 16 / 256.0, 24 / 256.0, 16 / 256.0, 4 / 256.0,
                     6 / 256.0, 24 / 256.0, 36 / 256.0, 24 / 256.0, 6 / 256.0,
                     4 / 256.0, 16 / 256.0, 24 / 256.0, 16 / 256.0, 4 / 256.0,
                     1 / 256.0,  4 / 256.0,  6 / 256.0,  4 / 256.0, 1 / 256.0 };

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

int main(int argc, char** argv) {
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

	ImageHandler dogHandler(projectDir, oiDogResourceFile, NUM_IMAGES);
	ImageHandler catHandler(projectDir, oiCatResourceFile, NUM_IMAGES);
	ImageHandler testHandler(projectDir, oiTestResourceFile, NUM_IMAGES);

	std::vector<cv::Mat> transformedDogImages = dogHandler.applyTransforms();
	std::vector<cv::Mat> transformedCatImages = catHandler.applyTransforms();
	std::vector<cv::Mat> testImages = testHandler.parseRawImagesFromResource();
    std::vector<cv::Mat> rawDogImages = dogHandler.getRawImages();

    int averageWidth = 0, averageHeight = 0;
	dogHandler.getAverageSizes(transformedDogImages, averageWidth, averageHeight);

	if (transformedDogImages.size() > 0) {
		//transform test images to same size as dog images
        //All of the dog images here should be the same size (from applyTransforms())
		cv::Size desiredSize(transformedDogImages[0].size().width, transformedDogImages[0].size().height);
		std::vector<cv::Mat> resizedCatImages = catHandler.resizeImages(transformedCatImages, desiredSize);
		std::vector<cv::Mat> resizedTestImages = testHandler.resizeImages(testImages, desiredSize);

		std::vector<anr::Mat> dog_images = dogHandler.convertToInteralMat(transformedDogImages);
		std::vector<anr::Mat> misc_images = testHandler.convertToInteralMat(resizedTestImages);

		//Uncomment below for viewing the transform images and testing.
		/*for (int i = 0; i < transformedDogImages.size(); i++) {
			if (!transformedDogImages[i].empty()) {
				cv::imshow("Dog Image", transformedDogImages[i]);
				cv::waitKey(0);
			}
	    }*/

        //Find out the minimum number of images.
        size_t min_images = (dog_images.size() < misc_images.size()
                ? dog_images.size() : misc_images.size());

        //Define vectors for training and expected data.
        std::vector<anr::Mat> training_data;
        std::vector<anr::Mat> expected_data;

        //Represents the training data (for printing).
        std::vector<cv::Mat> training_data_repr;

        std::cerr << "Log: Main: Adding Images to training data." << std::endl;

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

            //Add the representation images.
            training_data_repr.push_back(rawDogImages[i]);
            training_data_repr.push_back(testImages[i]);
        }

        //Initialize the layer sizes.
        std::vector<size_t> layer_sizes;
        layer_sizes.push_back(64 * 64);
        layer_sizes.push_back(100);
        //layer_sizes.push_back(500);
        //layer_sizes.push_back(50);
        layer_sizes.push_back(20);
        layer_sizes.push_back(2);


        //Use CPU ops for now, and build the basic model.
        anr::Ops* ops = new anr::Ops_hybrid;

        //anr::Ops* ops = new anr::Ops_cpu;
        anr::Mlp nn(layer_sizes, ops, 0.6);

        //TODO: Fix the sizing issues.
        //anr::Cnn cn;
        //for (int i = 0; i < training_data.size(); i++)
        //{
        //    //cn.maxpool(training_data[i], 2, 2);
        //    training_data[i] = cn.convolution(training_data[i], kernel, 5, 5);
        //}

        //Calculate the dividing index (training data vs testing data).
        size_t divide_idx = (size_t)((float)training_data.size() * MLP_TRAINING_RATIO);

        std::cerr << "Log: Main: Training the network." << std::endl;

        //Traing the mlp for as many points as requested (by the training ratio).
        for(size_t t = 0; t < 5; t++)
            for (size_t i = 0; i < divide_idx; i++)
                nn.train(training_data[i], expected_data[i]);

        std::cerr << "Log: Main: Predicting using the remaining data." << std::endl;

        size_t correct = 0;

        //Predict the data, and Check how many of the predictions were correct.
        for (size_t i = divide_idx; i < training_data.size(); i++) {
            anr::Mat predicted = nn.predict(training_data[i]);

            //Check if the predicted results were on the same priority bracked than expected.
            if(predicted.get(0, 0) >= 0.5 && expected_data[i].get(0, 0) == 1.0)
                correct++;
            else if(predicted.get(0, 1) >= 0.5 && expected_data[i].get(0, 1) == 1.0)
                correct++;

            //expected_data[i].print("\nExpected");
            //predicted.print("Prediction");

            //Add a seperator.
            //std::cout << "Image " << i << " : " ;

            //Print the prediction (first check if it's a dog, then if it's not).
            if(predicted.get(0, 0) >= 0.5)
                std::cout << "I might be a DOG" << std::endl;
            else if(predicted.get(0, 1) > 0.5)
               std::cout << "I might be a NOT A DOG" << std::endl;

            //Display the image (For the demo).
            /*cv::imshow("Image " + std::to_string(i), training_data_repr[i]);
            cv::waitKey(0);
            cv::destroyAllWindows();*/
        }

        //Print the calculation accuracy.
        std::cout << "\nResults: Correct=" << correct << " Total="
            << (training_data.size() - divide_idx) << " Accuracy="
            << float(correct) / float(training_data.size() - divide_idx) << std::endl;
	}
}
