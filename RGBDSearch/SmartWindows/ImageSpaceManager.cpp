#include "ImageSpaceManager.h"


ImageSpaceManager::ImageSpaceManager(void)
{
	minWinArea = 1000;
	maxLevel = 10;
	crit = DIV_MEANCOLORDIFF;
}

double ImageSpaceManager::GetIntegralValue(const cv::Mat& integralImg, cv::Rect box)
{	
	//std::cout<<box.tl().x<<" "<<box.tl().y<<" "<<box.br().x<<" "<<box.br().y<<" "<<integralImg.at<double>(0,0)<<std::endl;
	double val = integralImg.at<double>(box.br().y, box.br().x);
	val += integralImg.at<double>(box.tl().y, box.tl().x);
	val -= integralImg.at<double>(box.tl().y, box.br().x);
	val -= integralImg.at<double>(box.br().y, box.tl().x);
	return val;
}

//////////////////////////////////////////////////////////////////////////

bool ImageSpaceManager::Preprocess(cv::Mat& color_img)
{
	if(color_img.channels() != 3)
	{
		std::cerr<<"Input is not a color image."<<std::endl;
		return false;
	}

	cv::Mat grayimg;
	cv::cvtColor(color_img, grayimg, CV_BGR2GRAY);
	cv::Mat edgemap;
	cv::Canny(grayimg, edgemap, 50, 150);

	cv::imshow("canny", edgemap);
	cv::waitKey(10);

	return true;
}

bool ImageSpaceManager::ComputeColorIntegrals(const cv::Mat& color_img)
{
	if(color_img.channels() != 3)
	{
		std::cerr<<"Input is not a color image."<<std::endl;
		return false;
	}

	colorIntegrals.resize(3);

	// convert to lab
	cv::Mat labImg;
	cv::cvtColor(color_img, labImg, CV_BGR2Lab);

	// split channels
	std::vector<cv::Mat> colorChannels(3);
	cv::split(labImg, colorChannels);
	/*cv::imshow("b", colorChannels[0]);
	cv::imshow("g", colorChannels[1]);
	cv::imshow("r", colorChannels[2]);
	cv::waitKey(0);*/

	// compute integrals
	for(int i=0; i<3; i++) cv::integral(colorChannels[i], colorIntegrals[i], CV_64F);

	return true;
}

bool ImageSpaceManager::Divide(ImgWin& rootWin)
{
	if(rootWin.level == maxLevel)
		return true;

	double maxdist = 0;
	cv::Rect childWin1;
	cv::Rect childWin2;
	// try vertical
	int minHeight = rootWin.box.height/8;
	for(int r=minHeight; r<rootWin.box.height-minHeight; r++)
	{
		// compute mean color in each side
		cv::Rect topRect(rootWin.box.tl().x, rootWin.box.tl().y, rootWin.box.width, r);
		std::vector<double> meanTop(3);
		for(int i=0; i<3; i++) meanTop[i] = GetIntegralValue(colorIntegrals[i], topRect) / topRect.area();

		cv::Rect bottomRect(rootWin.box.tl().x, rootWin.box.tl().y+r, rootWin.box.width, rootWin.box.height-r);
		std::vector<double> meanBottom(3);
		for(int i=0; i<3; i++) meanBottom[i] = GetIntegralValue(colorIntegrals[i], bottomRect) / bottomRect.area();

		double dist = 0;
		for(int i=0; i<3; i++) dist += (meanTop[i]-meanBottom[i])*(meanTop[i]-meanBottom[i]);
		dist = std::sqrt(dist);
		if(dist > maxdist) { maxdist = dist; childWin1 = topRect; childWin2 = bottomRect; }
	}
	// try horizontal
	int minWidth = rootWin.box.width/8;
	for(int c=minWidth; c<rootWin.box.width-minWidth ; c++)
	{
		// compute mean color in each side
		cv::Rect leftRect(rootWin.box.tl().x, rootWin.box.tl().y, c, rootWin.box.height);
		std::vector<double> meanLeft(3);
		for(int i=0; i<3; i++) meanLeft[i] = GetIntegralValue(colorIntegrals[i], leftRect) / leftRect.area();

		cv::Rect rightRect(rootWin.box.tl().x+c, rootWin.box.tl().y, rootWin.box.width-c, rootWin.box.height);
		std::vector<double> meanRight(3);
		for(int i=0; i<3; i++) meanRight[i] = GetIntegralValue(colorIntegrals[i], rightRect) / rightRect.area();

		double dist = 0;
		for(int i=0; i<3; i++) dist += (meanLeft[i]-meanRight[i])*(meanLeft[i]-meanRight[i]);
		dist = std::sqrt(dist);
		if(dist > maxdist) { maxdist = dist; childWin1 = leftRect; childWin2 = rightRect; }
	}

	if(childWin1.area() > minWinArea)
	{
		ImgWin newWin;
		newWin.box = childWin1;
		newWin.id = wins.size();
		newWin.parentId = rootWin.id;
		newWin.level = rootWin.level + 1;
		wins.push_back(newWin);
		Divide(newWin);
	}
	if(childWin2.area() > minWinArea)
	{
		ImgWin newWin;
		newWin.box = childWin2;
		newWin.id = wins.size();
		newWin.parentId = rootWin.id;
		newWin.level = rootWin.level + 1;
		wins.push_back(newWin);
		Divide(newWin);
	}

	return true;
}

bool ImageSpaceManager::DivideImage(cv::Mat& color_img)
{
	ComputeColorIntegrals(color_img);

	wins.clear();
	ImgWin rootWin;
	rootWin.box = cv::Rect(0, 0, color_img.cols-1, color_img.rows-1);
	std::cout<<rootWin.box.br().x<<" "<<rootWin.box.br().y<<std::endl;
	rootWin.id = 0;
	rootWin.parentId = 0;
	rootWin.level = 0;
	wins.push_back(rootWin);

	Divide(rootWin);

	std::cout<<"Total windows: "<<wins.size()<<std::endl;

	return true;
}

bool ImageSpaceManager::A9Split(const cv::Mat& color_img)
{
	cv::Mat grayimg;
	cv::cvtColor(color_img, grayimg, CV_BGR2GRAY);
	cv::Mat integralImg;
	cv::integral(grayimg, integralImg, CV_32S); 

	const unsigned int num_leaves = 100;
	std::vector<Shape::Rectangle> rectangles(num_leaves);

	SplitImage(integralImg, num_leaves, rectangles);

	wins.clear();
	for(size_t i=0; i<rectangles.size(); i++)
	{
		ImgWin curwin;
		curwin.box = cv::Rect(cv::Point(rectangles[i].x1, rectangles[i].y1), cv::Point(rectangles[i].x2, rectangles[i].y2));
		wins.push_back(curwin);
	}

	return true;
}
