//////////////////////////////////////////////////////////////////////////



#include "VideoObjSegmentor.h"


int main()
{
	rgbdvision::VideoObjSegmentor vobj_segmentor;

	string lab_dir = "C:\\Users\\jiefeng\\Dropbox\\Experiment data\\book_processed\\book_processed\\";
	string video_dir = "C:\\Users\\jiefeng\\Documents\\MATLAB\\";

	vobj_segmentor.DoSegmentation(video_dir, 1, 50);

	return 0;
}