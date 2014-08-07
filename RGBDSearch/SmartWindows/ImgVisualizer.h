//////////////////////////////////////////////////////////////////////////
// a class for visualization functions
// jiefeng©2014-3-26
//////////////////////////////////////////////////////////////////////////


#pragma once

#include "common.h"


namespace visualsearch
{
	class ImgVisualizer
	{
	private:


	public:
		ImgVisualizer(void);

		// draw image windows
		static bool DrawImgWins(string winname, const Mat& img, const vector<ImgWin>& wins);

		// visualize float precision image
		static bool DrawFloatImg(string winname, const Mat& img, Mat& oimg, bool toDraw = true);

		static bool DrawShapes(const Mat& img, const vector<BasicShape>& shapes);

		static bool DrawImgCollection(string winname, const vector<Mat>& imgs, int totalnum, int numperrow, Mat& oimg);

	};
}


