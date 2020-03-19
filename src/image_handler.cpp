/**
 *  The class responsible for handling all of the image manipulation. Public methods available to apply different transforms
 *  on a a set of images. Ultimately, one method could be used to apply all the necessary transforms at once.
 *
 *  @author Drew Nelson
 *  @date March 2020
 */

#include <iostream>
#include <fstream>
#include <sstream>  
#include <iterator>

#include "image_handler.hpp"

/*
*  Utility method to parse a string into a double value.
*  @param str - Input string to convert to double
*  @returns - Parsed double value. Defaults to a value of 0.0
*/
double ImageHandler::stringToDouble(std::string str)
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

/**
*  Constructor which sets the appropriate paths needed to parse and manipulate images
*
*  @param rootSrcPath - Path to the root of the source directory.
*  @param pathToResourceFile - Path to the resource file. The resource file should contain a list of pairs which contain
*                              paths to the image label and image file.
*/
ImageHandler::ImageHandler(std::string rootSrcPath, std::string pathToResourceFile)
{
	_rootSrcPath = rootSrcPath;
	_pathToResourceFile = pathToResourceFile;
}

/*
*  Load and create a populate an OpenImage structure. The file being loaded is expected to be
*  an OpenImages label text file. The text file is expected to contain a single line which includes
*  the keyword used to help generate the image and the boundary box positions
*  @param pathToFile - Path the information label text file
*  @param error - Reference parameter to help return an error
*  @returns - Populated OpenImage struct
*/
OpenImage ImageHandler::parseOpenImage(std::string pathToFile, std::string& error)
{
	OpenImage parsedImage;
	try
	{
		std::string projectDir = _rootSrcPath;
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

/*
*  Given an OpenImage struct, convert the object to OpenCV image (mat).
   Image is read in grayscale which is used for MLP purposes.
*  @param openImg - OpenImage structure to convert
*  @returns - OpenCV image object (cv::Mat)
*/
cv::Mat ImageHandler::convertOpenImage(OpenImage openImg)
{
	cv::Mat convertedMat;
	if (!openImg.pathToImage.empty())
	{
		convertedMat = cv::imread(openImg.pathToImage, 0); //0 indicates flag to read as grayscale
	}
	return convertedMat;
}

/*
*  Transform the provided OpenCV image to a new image which just contains
*  the boundary area containing the desired area of the image
*  @param img - Source image
*  @param imgDetail - OpenImage details containing boundary locations
*/
cv::Mat ImageHandler::applyBoundaryTransform(cv::Mat img, OpenImage imgDetail)
{
	cv::Mat boundaryImg(img);
	if (!img.empty())
	{
		const int imgWidth = img.size().width;
		const int imgHeight = img.size().height;

		//Create rect to crop image
		int cropX = int(imgDetail.left); //Truncating value here to ensure full pixel obtained. (Force rounding down)
		int cropY = int(imgDetail.top); //Truncating value here to ensure full pixel obtained. (Force rounding down)
		int cropWidth = (int(imgDetail.right) + 1) - cropX; //Similar to above, but round up
		int cropHeight = (int(imgDetail.bottom) + 1) - cropY; //Similar to above, but round up


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

		cv::Rect cropArea(cropX, cropY, cropWidth, cropHeight);
		boundaryImg = img(cropArea);
	}
	else
	{
		std::cerr << "Error occurred trying to convert OpenImage object to OpenCV Mat" << std::endl;
		imgDetail.print();
	}

	return boundaryImg;
}

/*
*  Convert a local image into an OpenCV image
*  @param str - Path the image file
*  @returns OpenCV matrix (OpenCV image)
*/
cv::Mat ImageHandler::loadImageFromFile(std::string pathToFile)
{
	cv::Mat convertedMat;
	if (!pathToFile.empty())
	{
		convertedMat = cv::imread(pathToFile);
	}
	return convertedMat;
}

/*
*  Parse the resource file and generate a list of OpenImages
*  @returns List of OpenImage objects containing details of the images available
*/
std::vector<OpenImage> ImageHandler::parseImages()
{
	std::vector<OpenImage> images;
	std::ifstream fileStream(_pathToResourceFile);
	std::cout << "Attempting to load images from resource file: "
		<< _pathToResourceFile.c_str() << std::endl;

	if (fileStream.is_open())
	{
		std::string line;
		unsigned int lineCount = 0;
		std::string projectDir = _rootSrcPath;
		while (getline(fileStream, line))
		{
			std::istringstream stringStream(line);
			std::istream_iterator<std::string> begin(stringStream), end;

			//putting all the tokens in the vector
			std::vector<std::string> imageTokens(begin, end);
			if (imageTokens.size() == 2)
			{
				std::string error;
				OpenImage oiImg = parseOpenImage(imageTokens[0], error);

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

/*
*  Given a list of OpenImage objects, create a list of of OpenCV images that have been
*  shrunken down the desired boundary areas as specified in each OpenImage struct.
*  Addtionally, the images are read in grayscale to help with the MLP algorithm.
*
*  @param openImages - OpenImage structures to convert to OpenCV images
*  @returns - List of OpenCV images that only display the boundary area of the original image
*/
std::vector<cv::Mat> ImageHandler::applyBoundaryTransform(std::vector<OpenImage> openImages)
{
	std::cout << "Attempting to apply boundary transform on images..." << std::endl;
	std::vector<cv::Mat> boundaryImages;
	for (int i = 0; i < openImages.size(); i++) {
		cv::Mat origImg = convertOpenImage(openImages[i]);
		if (!origImg.empty())
		{
			cv::Mat boundaryImg = applyBoundaryTransform(origImg, openImages[i]);
			if (!boundaryImg.empty())
			{
				boundaryImages.push_back(boundaryImg);
			}
		}
		else 
		{
			std::cerr << "Error occurred obtaining boundary image " << std::endl;
			openImages[i].print();
		}
	}
	return boundaryImages;
}


/*
*  Given a set of images, resize each one to the average size amoungst every
*  image.
*  @param images - Original images to resize
*  @returns - List of resized images.
*/
std::vector<cv::Mat> ImageHandler::applyAverageSizeTransform(std::vector<cv::Mat> images)
{
	std::cout << "Attempting to apply resize transform on images..." << std::endl;
	std::vector<cv::Mat> resizedImages;
	unsigned int totalWidth = 0;
	unsigned int totalHeight = 0;
	unsigned int averageWidth = 0;
	unsigned int averageHeight = 0;

	//Calculate the average image size
	for (int i = 0; i < images.size(); i++)
	{
		totalWidth += images[i].size().width;
		totalHeight += images[i].size().height;
	}

	cv::Size desiredSize(int(totalWidth / images.size()), int(totalHeight / images.size()));
	for (int i = 0; i < images.size(); i++)
	{
		cv::Mat resizedImage;
		cv::resize(images[i], resizedImage, desiredSize, 0, 0);
		resizedImages.push_back(resizedImage);
	}

	return resizedImages;
}

/*
*  Wrapper method which parses the resource file and performs all the neccessary
* transforms on the image for other algorithms used in the program.
*  @returns - OpenCV images that have been cropped, grayscaled, and resized
*/
std::vector<cv::Mat> ImageHandler::applyTransforms()
{
	std::vector<cv::Mat> resultImages;
	std::vector<OpenImage> openImages = parseImages();
	if (!openImages.empty())
	{
		std::vector<cv::Mat> boundaryImages = applyBoundaryTransform(openImages);
		if (!boundaryImages.empty())
		{
			std::vector<cv::Mat> resizedImages = applyAverageSizeTransform(boundaryImages);
			if (!resizedImages.empty())
			{
				resultImages = resizedImages;
			}
		}
	}
	return resultImages;
}

/*
*  Convert an OpenCV Mat to the internal matrix type that we'll use for the convolusions and MLP
*  @param images - Original OpenCV images
*  @returns - List of internal matrix objects
*/
std::vector<anr::Mat> ImageHandler::convertToInteralMat(std::vector<cv::Mat> images)
{
	std::vector<anr::Mat> convertedMats;

	for (int i = 0; i < images.size(); i++)
	{
		cv::Mat convertedMat;
		anr::Mat internalMat(images[i].size().height, images[i].size().width);
		images[i].convertTo(convertedMat, CV_32F, 1.0 / 255, 0);
		
		for (int row = 0; row < convertedMat.size().height; row++)
		{
			for (int col = 0; col < convertedMat.size().width; col++)
			{
				internalMat.data[(row * convertedMat.size().width) + col] = float(convertedMat.at<float>(row, col));
			}
		}
		convertedMats.push_back(internalMat);
	}

	return convertedMats;
}

/*
*  Parse images from the resource text file as OpenCV objects.
*  @returns - List of OpenCV mat objects.
*/
std::vector<cv::Mat> ImageHandler::parseRawImagesFromResource()
{
	std::vector<cv::Mat> images;
	std::ifstream fileStream(_pathToResourceFile);
	std::cout << "Attempting to load images from resource file (raw): "
		<< _pathToResourceFile.c_str() << std::endl;

	if (fileStream.is_open())
	{
		std::string line;
		unsigned int lineCount = 0;
		std::string projectDir = _rootSrcPath;
		while (getline(fileStream, line))
		{
			cv::Mat loadedImage = loadImageFromFile(_rootSrcPath + line);
			images.push_back(loadedImage);
		}
	}
	return images;
}