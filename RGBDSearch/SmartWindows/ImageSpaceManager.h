//////////////////////////////////////////////////////////////////////////
// manage image space
// jiefeng@2014-03-08
//////////////////////////////////////////////////////////////////////////


#pragma once

#include "common.h"
#include "a9wins\SmartWindowing.h"


enum DivideCriteria
{
	DIV_MEANCOLORDIFF
};


// this class is aimed to give a set of windows as candidate 
// for further ranking to give high precision of object windows
class ImageSpaceManager
{
private:
	std::vector<cv::Mat> colorIntegrals;
	float minWinArea;
	int maxLevel;
	DivideCriteria crit;

	double GetIntegralValue(const cv::Mat& integralImg, cv::Rect box);

public:

	std::vector<ImgWin> wins;

	ImageSpaceManager(void);

	bool Preprocess(cv::Mat& color_img);

	// compute integral images for each color channels
	bool ComputeColorIntegrals(const cv::Mat& color_img);

	bool DivideImage(cv::Mat& color_img);

	// recursively divide image window based separation criteria
	bool Divide(ImgWin& rootWin);

	bool A9Split(const cv::Mat& color_img);

};

