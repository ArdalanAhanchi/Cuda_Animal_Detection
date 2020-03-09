#ifndef OPEN_IMAGE_HPP
#define OPEN_IMAGE_HPP
#include <string>

///<summary>Simple structure to contain image information for an image file from OpenImages (Google)</summary>
struct OpenImage
{
	///<summary>File path to the image of which this structure contains data for</summary>
	std::string pathToImage;

	///<summary>Keyword used to obtain this image</summary>
	std::string filterDescription;

	///<summary>Minimum X poition of the boundary box</summary>
	double xMin = 0.0;

	///<summary>Maximum X poition of the boundary box</summary>
	double xMax = 0.0;

	///<summary>Minimum Y poition of the boundary box</summary>
	double yMin = 0.0;

	///<summary>Maximum Y poition of the boundary box</summary>
	double yMax = 0.0;

	///<summary>Utility method used to print the contents of this structure</summary>
	void print();
};

#endif // !OPEN_IMAGE_H
