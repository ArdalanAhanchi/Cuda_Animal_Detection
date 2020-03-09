#include <iostream>
#include <string>
#include <fstream>
#include <sstream>  
#include <iterator> 
#include <vector>

#include "OpenImage.h"

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
		printf("Error occurred parsing value to double. Error num: %i\n", e);
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
					parsedImage.xMin = stringToDouble(imageTokens[1]);
					parsedImage.xMax = stringToDouble(imageTokens[2]);
					parsedImage.yMin = stringToDouble(imageTokens[3]);
					parsedImage.yMax = stringToDouble(imageTokens[4]);
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
	printf("Attempting to load images from resource file: %s\n", resourceFile.c_str());
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
					printf("Error occurred parsing line %i: %s\n", lineCount, error.c_str());
				}
				else 
				{
					oiImg.pathToImage = projectDir + imageTokens[1];
					images.push_back(oiImg);
				}
			}
			else
			{
				printf("Unexpected number of parameters on entry line %i\n", lineCount);
			}
			lineCount++;
		}
	}

	return images;
}

bool canOpenFiles(std::string resourceFile)
{
	std::ifstream fileStream(resourceFile);

	return fileStream.is_open();
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
		printf("Loaded %i images\n", dogImages.size());
	}
	else {
		printf("No images loaded");
	}
	
    std::cout << "Hello Animal Detector!" << std::endl;
}
