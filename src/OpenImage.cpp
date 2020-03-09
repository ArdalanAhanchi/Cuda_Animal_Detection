#include "OpenImage.h"

void OpenImage::print()
{
	printf("%s, %s, %f, %f, %f, %f\n", pathToImage.c_str(), filterDescription.c_str(), xMin, xMax, yMin, yMax);
}