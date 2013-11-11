//////////////////////////////////////////////////////////////////////////



#include "VideoObjSegmentor.h"


int main()
{
	rgbdvision::VideoObjSegmentor vobj_segmentor;

	string lab_dir = "F:\\test\\processed_mug2\\";
	string video_dir = "C:\\Users\\vv\\Dropbox\\Experiment data\\book_processed\\book_processed\\";	//"C:\\Users\\jiefeng\\Documents\\MATLAB\\";

	vobj_segmentor.DoSegmentation(lab_dir, 23, 216);

	return 0;
}