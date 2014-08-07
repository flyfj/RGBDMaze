


#pragma once

#include <vector>
#include "common.h"


// evaluate performance of object proposal
class WindowEvaluator
{
private:


public:

	string gtdir;

	WindowEvaluator(void);

	static float ComputeWinMatchScore(const ImgWin& qwin, const ImgWin& gwin);

	// for each image
	Point2f CompPRForSingleImg(const vector<ImgWin>& det_wins, const vector<ImgWin>& gt_wins, int topK);
	// for all images
	Point2f ComputePR(const vector<vector<ImgWin>>& det_wins, const vector<vector<ImgWin>>& gt_wins, int topK);

	// best matched detection windows with ground truth
	bool FindBestWins(const vector<ImgWin>& det_wins, const vector<ImgWin>& gt_wins, vector<ImgWin>& bestWins);

};

