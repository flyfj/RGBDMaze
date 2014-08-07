#include "A9Window.h"


A9Window::A9Window(void)
{
}

//////////////////////////////////////////////////////////////////////////

void A9Window::GenerateBlocks(const cv::Mat& frame, std::vector<ImgWin>& wins)
{
	const unsigned int num_leaves = 1000;
	std::vector<Shape::Rectangle> rectangles(num_leaves);

	int cnt = 0;
	cv::RNG rng;

	// Get integral image.
	cv::Mat integral_image;
	cv::integral(frame, integral_image, CV_32S);

	// Partition image.
	SplitImage(integral_image, num_leaves, rectangles);

	// Get the windows.
	std::vector<Shape::Rectangle> windows;
	float dist_threshold = 1.5;  // Between 0 and 2.
	std::vector<float> thresholds;
	thresholds.push_back(300);
	thresholds.push_back(500);
	thresholds.push_back(700);
	thresholds.push_back(1000);

	GetWindows(integral_image, rectangles, thresholds, dist_threshold, windows);

	wins.resize(windows.size());
	for(size_t i=0; i<windows.size(); i++)
	{
		wins[i].x = windows[i].x1;
		wins[i].y = windows[i].y1;
		wins[i].width = windows[i].x2 - windows[i].x1;
		wins[i].height = windows[i].y2 - windows[i].y1;
	}

}
