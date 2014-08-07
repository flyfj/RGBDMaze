#include "WindowEvaluator.h"


WindowEvaluator::WindowEvaluator(void)
{
}

//////////////////////////////////////////////////////////////////////////

float WindowEvaluator::ComputeWinMatchScore(const ImgWin& qwin, const ImgWin& gwin)
{
	Rect box1(qwin.x, qwin.y, qwin.width, qwin.height);
	Rect box2(gwin.x, gwin.y, gwin.width, gwin.height);

	Rect interBox = box1 & box2;
	Rect unionBox = box1 | box2;

	if(unionBox.area() > 0)
		return (float)interBox.area() / unionBox.area();
	else
		return 0;
}


Point2f WindowEvaluator::CompPRForSingleImg(const vector<ImgWin>& det_wins, const vector<ImgWin>& gt_wins, int topK)
{
	Point2f pr_val(0, 0);
	set<int> corr_gtwins;
	int validnum = MIN(topK, det_wins.size());
	for(size_t i=0; i<validnum; i++)
	{
		for(size_t j=0; j<gt_wins.size(); j++)
		{
			if( ComputeWinMatchScore(det_wins[i], gt_wins[j]) > 0.5f )
			{
				pr_val.x++;
				corr_gtwins.insert(j);
				break;
			}
		}
	}

	pr_val.y = corr_gtwins.size();

	return pr_val;
}

Point2f WindowEvaluator::ComputePR(const vector<vector<ImgWin>>& det_wins, const vector<vector<ImgWin>>& gt_wins, int topK)
{
	Point2f pr_val(0, 0);
	int sum_det = 0;
	int sum_gt = 0;
	// loop each image
	for(size_t i=0; i<det_wins.size(); i++)
	{
		int validnum = MIN(topK, det_wins[i].size());
		sum_det += validnum;
		sum_gt += gt_wins[i].size();
		Point2f cur_pr = CompPRForSingleImg(det_wins[i], gt_wins[i], validnum);
		pr_val.x += cur_pr.x;
		pr_val.y += cur_pr.y;
	}

	pr_val.x /= sum_det;
	pr_val.y /= sum_gt;
	return pr_val;
}


bool WindowEvaluator::FindBestWins(const vector<ImgWin>& det_wins, const vector<ImgWin>& gt_wins, vector<ImgWin>& bestWins)
{
	bestWins.clear();

	for(size_t i=0; i<gt_wins.size(); i++)
	{
		// find best matched detection window (>0.5)
		float bestscore = 0;
		ImgWin bestwin;
		for(size_t j=0; j<det_wins.size(); j++)
		{
			float s = ComputeWinMatchScore(gt_wins[i], det_wins[j]);
			if(s > bestscore)
			{
				bestscore = s;
				bestwin = det_wins[j];
			}
		}

		if(bestscore > 0.5)
			bestWins.push_back(bestwin);
	}

	return true;
}