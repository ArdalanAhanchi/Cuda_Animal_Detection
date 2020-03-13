/**
 *  The class responsible for handling all of the image manipulation. Public methods available to apply different transforms
 *  on a a set of images. Ultimately, one method could be used to apply all the necessary transforms at once.
 *
 *  @author Drew Nelson
 *  @date March 2020
 */

#ifndef IMAGE_HANDLER_HPP
#define IMAGE_HANDLER_HPP

#include <vector>
#include <opencv2/core.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/highgui.hpp>
#include "opencv2/imgproc.hpp"

#include "open_image.hpp"
#include <string>

class ImageHandler
{
private:
	/* Stored reference containing the path to the root of the src directory */
	std::string _rootSrcPath;

	/* Stored reference containing the path to the resource file */
	std::string _pathToResourceFile;

	/*
	*  Load and create a populate an OpenImage structure. The file being loaded is expected to be
	*  an OpenImages label text file. The text file is expected to contain a single line which includes
	*  the keyword used to help generate the image and the boundary box positions
	*  @param pathToFile - Path the information label text file
	*  @param error - Reference parameter to help return an error
	*  @returns - Populated OpenImage struct
	*/
	OpenImage parseOpenImage(std::string pathToFile, std::string& error);

	/*
	*  Given an OpenImage struct, convert the object to OpenCV image (mat)
	*  @param openImg - OpenImage structure to convert
	*  @returns - OpenCV image object (cv::Mat)
	*/
	cv::Mat convertOpenImage(OpenImage openImg);

	/*
	*  Transform the provided OpenCV image to a new image which just contains
	*  the boundary area containing the desired area of the image
	*  @param img - Source image
	*  @param imgDetail - OpenImage details containing boundary locations
	*/
	cv::Mat applyBoundaryTransform(cv::Mat img, OpenImage imgDetail);

	/*
	*  Utility method to parse a string into a double value.
	*  @param str - Input string to convert to double
	*  @returns - Parsed double value. Defaults to a value of 0.0
	*/
	double stringToDouble(std::string str);

public:
	/**
	 *  Constructor which sets the appropriate paths needed to parse and manipulate images
	 *
	 *  @param rootSrcPath - Path to the root of the source directory. 
	 *  @param pathToResourceFile - Path to the resource file. The resource file should contain a list of pairs which contain
	 *                              paths to the image label and image file.
	 */
	ImageHandler(std::string rootSrcPath, std::string pathToResourceFile);

	/*
	*  Parse the resource file and generate a list of OpenImages
	*  @returns List of OpenImage objects containing details of the images available
	*/
	std::vector<OpenImage> parseImages();

	/*
	*  Given a list of OpenImage objects, create a list of of OpenCV images that have been
	*  shrunken down the desired boundary areas as specified in each OpenImage struct.
	*  Addtionally, the images are read in grayscale to help with the MLP algorithm.
	*  
	*  @param openImages - OpenImage structures to convert to OpenCV images
	*  @returns - List of OpenCV images that only display the boundary area of the original image
	*/
	std::vector<cv::Mat> applyBoundaryTransform(std::vector<OpenImage> openImages);

	/*
	*  Given a set of images, resize each one to the average size amoungst every
	*  image.
	*  @param images - Original images to resize
	*  @returns - List of resized images.
	*/
	std::vector<cv::Mat> applyAverageSizeTransform(std::vector<cv::Mat> images);

	/*
	*  Wrapper method which parses the resource file and performs all the neccessary
	* transforms on the image for other algorithms used in the program.
	*  @returns - OpenCV images that have been cropped, grayscaled, and resized
	*/
	std::vector<cv::Mat> applyTransforms();
};

#endif