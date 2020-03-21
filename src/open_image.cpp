/**
 *  Simple structure responsible for containing the data parsed from a resource
 *  file containing information (data and paths) for an OpenImage file
 *
 *  @author Drew Nelson
 *  @date March 2020
 */
#include "open_image.hpp"

/* Utility method used to print the contents of this structure */
void OpenImage::print()
{
	printf("%s, %s, %f, %f, %f, %f\n", pathToImage.c_str(), filterDescription.c_str(), left, top, right, bottom);
}
