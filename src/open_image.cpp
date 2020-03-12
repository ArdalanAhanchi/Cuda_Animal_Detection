#include "open_image.hpp"

void OpenImage::print()
{
	printf("%s, %s, %f, %f, %f, %f\n", pathToImage.c_str(), filterDescription.c_str(), left, top, right, bottom);
}
