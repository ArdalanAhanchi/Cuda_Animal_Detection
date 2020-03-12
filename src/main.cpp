#include <iostream>
#include <string>
#include <fstream>
#include <sstream>  
#include <iterator> 
#include <vector>
#include <chrono>

#include <opencv2/core.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/highgui.hpp>

#include "open_image.hpp"

/// <summary>
/// Utility method to parse a string into a double value.
/// </summary>
/// <param name="str">Input string to conver to double</param>
/// <returns>Parsed double value. Defaults to a value of 0.0</returns>
double stringToDouble(std::string str)
{
	double value = 0.0;
	try
	{
		value = std::stod(str);
	}
	catch (int e)
	{
		std::cerr << "Error occurred parsing value to double. Error num: " << e << std::endl;
	}
	return value;
}

/// <summary>
/// Load and create a populate an OpenImage structure. The file being loaded is expected to be
/// an OpenImages label text file. The text file is expected to contain a single line which includes
/// the keyword used to help generate the image and the boundary box positions
/// </summary>
/// <param name="pathToFile">Path the information label text file</param>
/// <param name="error">Reference parameter to help return an error</param>
/// <returns>Populated OpenImage struct</returns>
OpenImage LoadAndCreateObj(std::string pathToFile, std::string& error)
{
	OpenImage parsedImage;
	try
	{
		std::string projectDir = std::getenv("CSS535_PROJ");
		std::string labelTextFile = projectDir + pathToFile;
		std::ifstream fileStream(labelTextFile);

		if (fileStream.is_open())
		{
			std::string line;
			while (getline(fileStream, line))
			{
				std::istringstream stringStream(line);
				std::istream_iterator<std::string> begin(stringStream), end;

				//putting all the tokens in the vector
				std::vector<std::string> imageTokens(begin, end);
				if (imageTokens.size() == 5)
				{
					parsedImage.filterDescription = imageTokens[0];
					parsedImage.left = stringToDouble(imageTokens[1]);
					parsedImage.top = stringToDouble(imageTokens[2]);
					parsedImage.right = stringToDouble(imageTokens[3]);
					parsedImage.bottom = stringToDouble(imageTokens[4]);
				}
				else
				{
					error = "Label text file contains unexpected number of parameters";
				}
			}
		}
		else 
		{
			error = "Unable to open label txt file";
		}
	}
	catch (int e)
	{
		error = "An exception occurred trying to load an OpenImage resource";
	}
	
	return parsedImage;
}

/// <summary>
/// Load a list of OpenImages file given a specified resource file. Resource files are generated using an external tool which
/// iterates over the the directories created from OpenImages to create a single file containing all the information needed to help
/// load images
/// </summary>
/// <param name="resourceFile">Path to the resource file</param>
/// <returns>List of OpenImage structs</returns>
std::vector<OpenImage> getImagesFromResourceFile(std::string resourceFile)
{
	std::vector<OpenImage> images;
	std::ifstream fileStream(resourceFile);
	std::cerr << "Attempting to load images from resource file: " 
        << resourceFile.c_str() << std::endl;

	if (fileStream.is_open())
	{
		std::string line;
		unsigned int lineCount = 0;
		std::string projectDir = std::getenv("CSS535_PROJ");
		while (getline(fileStream, line))
		{
			std::istringstream stringStream(line);
			std::istream_iterator<std::string> begin(stringStream), end;

			//putting all the tokens in the vector
			std::vector<std::string> imageTokens(begin, end);
			if (imageTokens.size() == 2)
			{
				std::string error;
				OpenImage oiImg = LoadAndCreateObj(imageTokens[0], error);

				if (error.size() > 0)
				{
					std::cerr << "Error occurred parsing line " << lineCount << " :"
                        << error.c_str() << std::endl;
				}
				else 
				{
					oiImg.pathToImage = projectDir + imageTokens[1];
					images.push_back(oiImg);
				}
			}
			else
			{
				std::cerr << "Unexpected number of parameters on entry line " 
                    << lineCount << std::endl;
			}
			lineCount++;
		}
	}

	return images;
}

/// <summary>
/// Convert an OpenImg object into a OpenCV Mat object. The OpenCV Mat object is a cropped image based on the original image.
/// </summary>
/// <param name="openImg">OpenImg object to convert to OpenCV Mat object</param>
/// <returns>Cropped image object</returns>
cv::Mat createBoundingImage(OpenImage openImg)
{
	cv::Mat imgOrig = cv::imread(openImg.pathToImage);
	cv::Mat cropImg;

	if (!imgOrig.empty())
	{
		const int imgWidth = imgOrig.size().width;
		const int imgHeight = imgOrig.size().height;

		//Create rect to crop image
		int cropX = int(openImg.left); //Truncating value here to ensure full pixel obtained. (Force rounding down)
		int cropY = int(openImg.top); //Truncating value here to ensure full pixel obtained. (Force rounding down)
		int cropWidth = (int(openImg.right) + 1) - cropX; //Similar to above, but round up
		int cropHeight = (int(openImg.bottom) + 1) - cropY; //Similar to above, but round up
		
		
		//Perform check to ensure that the total crop width doesn't exceed the actual image width
		if (cropX + cropWidth >= imgWidth)
		{
			cropWidth = imgWidth - cropX - 1;
		}
		
		//Perform check to ensure that the total crop height doesn't exceed the actual image height
		if (cropY + cropHeight >= imgHeight)
		{
			cropHeight = imgHeight - cropY - 1;
		}

		cv::Rect cropArea (cropX, cropY, cropWidth, cropHeight);
		cropImg = imgOrig(cropArea);
	}
	else 
	{
		std::cerr << "Error occurred trying to convert OpenImage object to OpenCV Mat" << std::endl;
		openImg.print();
	}
	return cropImg;
}

int main(int argc, char** argv)
{
	std::string projectDir = std::getenv("CSS535_PROJ");
#ifdef _WIN32
	std::string oiDogResourceFile = projectDir + "\\images\\open-images\\Dog_oi_resource.windows.txt";
#elif linux
	std::string oiDogResourceFile = projectDir + "/images/open-images/Dog_oi_resource.linux.txt";
#endif

	std::vector<OpenImage> dogImages = getImagesFromResourceFile(oiDogResourceFile);
	if (dogImages.size() > 0)
	{
		std::cout << "Loaded " << dogImages.size() << " images." << std::endl;
		std::cout << "Converting images to cropped OpenCV images..." << std::endl;

		std::vector<cv::Mat> croppedImages;
		for (int i = 0; i < dogImages.size(); i++) {
			cv::Mat cropImage = createBoundingImage(dogImages[i]);
			
			if (!cropImage.empty())
			{
				croppedImages.push_back(cropImage);
			}
		}

		std::cout << "Conversions complete. Total Number of cropped images: " << croppedImages.size() << std::endl;
	}
	else {
		std::cerr << "No images loaded" << std::endl;
	}
}
