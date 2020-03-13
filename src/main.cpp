#include <iostream>
#include <string>
#include <vector>

#include <opencv2/core.hpp>
#include <opencv2/imgcodecs.hpp>

#include "image_handler.hpp"


int main(int argc, char** argv)
{
	std::string projectDir = std::getenv("CSS535_PROJ");

	//We can add more references to other resource files if time permitting
#ifdef _WIN32
	std::string oiDogResourceFile = projectDir + "\\images\\open-images\\Dog_oi_resource.windows.txt";
#elif linux
	std::string oiDogResourceFile = projectDir + "/images/open-images/Dog_oi_resource.linux.txt";
#endif

	ImageHandler dogHandler(projectDir, oiDogResourceFile);
	std::vector<cv::Mat> transformedImages = dogHandler.applyTransforms();

	//Can uncomment the content below to verify that the images are being loaded as expected
	/*if (transformedImages.size() > 0)
	{
		for (int i = 0; i < transformedImages.size(); i++)
		{
			if (!transformedImages[i].empty())
			{
				cv::imshow("Dog Image", transformedImages[i]);
				cv::waitKey(0);
			}
		}
	}*/
}
