//////////////////////////////////////////////////////////////////////////



#include "VideoObjSegmentor.h"


int main()
{
	rgbdvision::VideoObjSegmentor vobj_segmentor;

	string lab_dir = "F:\\test\\NewlyAlignedRGB_mug\\NewlyAlignedRGB_mug\\";
	string video_dir = "C:\\Users\\vv\\Dropbox\\Experiment data\\book_processed\\book_processed\\";	//"C:\\Users\\jiefeng\\Documents\\MATLAB\\";

	vobj_segmentor.DoSegmentation(lab_dir, 196, 196);

	return 0;
}