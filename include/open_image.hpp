/**
 *  Simple structure responsible for containing the data parsed from a resource
 *  file containing information (data and paths) for an OpenImage file
 *
 *  @author Drew Nelson
 *  @date March 2020
 */

#ifndef OPEN_IMAGE_HPP
#define OPEN_IMAGE_HPP
#include <string>

struct OpenImage
{
	/* File path to the image of which this structure contains data for */
	std::string pathToImage;

	/* Keyword used to obtain this image */
	std::string filterDescription;

	/* Left X position of the boundary box */
	double left = 0.0;

	/* Top Y poition of the boundary box */
	double top = 0.0;

	/* Right X poition of the boundary box */
	double right = 0.0;

	/* Bottom Y poition of the boundary box */
	double bottom = 0.0;

	/* Utility method used to print the contents of this structure */
	void print();
};

#endif // !OPEN_IMAGE_H
