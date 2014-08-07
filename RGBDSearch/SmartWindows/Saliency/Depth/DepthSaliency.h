//////////////////////////////////////////////////////////////////////////
// measure saliency of a window using depth
// jiefeng©2014-6-7
//////////////////////////////////////////////////////////////////////////

#pragma once

#include "common.h"
#include "Tools.h"
#include "ImageSegmentor.h"
#include "ImgVisualizer.h"

class DepthSaliency
{
private:

	visualsearch::ImageSegmentor imgSegmentor;
	int depth_bin_num;
	float depth_bin_step;
	Mat depth_dist_mat;

public:
	DepthSaliency(void);

	bool InitQuantization(const Mat& dmap, Mat& dcode);

	double CompDepthVariance(const Mat& dmap, ImgWin win);

	bool CompWinDepthSaliency(const Mat& dmap, ImgWin& win);

	bool CompCenterSurroundDepthDist(const Mat& dcode, ImgWin& win);

	bool CompWin3DSaliency(const Mat& cloud, ImgWin& win);

	bool DepthToCloud(const Mat& dmap, Mat& cloud);

	bool OutputToOBJ(const Mat& cloud, string objfile);

	void RankWins(const Mat& dmap, vector<ImgWin>& wins);
};

