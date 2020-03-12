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

	///<summary>Left X position of the boundary box</summary>
	double left = 0.0;

	///<summary>Top Y poition of the boundary box</summary>
	double top = 0.0;

	///<summary>Right X poition of the boundary box</summary>
	double right = 0.0;

	///<summary>Bottom Y poition of the boundary box</summary>
	double bottom = 0.0;

	///<summary>Utility method used to print the contents of this structure</summary>
	void print();
};

#endif // !OPEN_IMAGE_H
