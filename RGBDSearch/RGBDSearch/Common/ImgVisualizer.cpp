//////////////////////////////////////////////////////////////////////////

#include "ImgVisualizer.h"


namespace visualsearch
{
	ImgVisualizer::ImgVisualizer(void)
	{
	}

	//////////////////////////////////////////////////////////////////////////

	bool ImgVisualizer::DrawImgWins(string winname, const Mat& img, const vector<ImgWin>& wins)
	{
		cv::RNG rng(cv::getTickCount());
		cv::Mat dispimg;
		if(img.channels() == 3)
			dispimg = img.clone();
		else
		{
			DrawFloatImg("", img, dispimg, false);
			cvtColor(dispimg, dispimg, CV_GRAY2BGR);
		}

		int num = MIN(wins.size(), 20);
		for(size_t i=0; i<num; i++)
		{
			cv::rectangle(dispimg, wins[i], CV_RGB(rng.next()%255, rng.next()%255, rng.next()%255), 2);
		}
		cv::imshow(winname, dispimg);
		cv::waitKey(10);

		return true;
	}

	bool ImgVisualizer::DrawFloatImg(string winname, const Mat& img, Mat& oimg, bool toDraw)
	{
		// normalize
		cv::normalize(img, oimg, 0, 1, NORM_MINMAX);
		// convert to 8u
		oimg.convertTo(oimg, CV_8U, 255);

		if( toDraw )
		{
			// show image
			imshow(winname, oimg);
			cv::waitKey(10);
		}

		return true;
	}

	bool ImgVisualizer::DrawShapes(const Mat& img, const vector<BasicShape>& shapes)
	{
		if(img.channels() != 1 && img.channels() != 3)
			return false;

		cv::Mat colorimg;
		if(img.channels() == 1)
			cvtColor(img, colorimg, CV_GRAY2BGR);
		else
			colorimg = img.clone();

		cv::RNG rng_gen;
		for (size_t i=0; i<shapes.size(); i++)
		{
			CvScalar cur_color = CV_RGB(rng_gen.uniform(0,255), rng_gen.uniform(0,255), rng_gen.uniform(0,255));
			Contours curves;
			curves.push_back(shapes[i].approx_contour);
			drawContours(colorimg, curves, 0, cur_color);
		}
		imshow("contours", colorimg);
		cv::waitKey(10);

		return true;
	}

	bool ImgVisualizer::DrawImgCollection(string winname, const vector<Mat>& imgs, int totalnum, int numperrow, Mat& oimg)
	{
		int validnum = MIN(imgs.size(), totalnum);
		if(validnum == 0)
		{
			cerr<<"Empty image collection to draw."<<endl;
			return false;
		}

		int itemperrow = numperrow;
		int itempercol = (int)ceil((float)validnum / itemperrow);
		Size itemSize(100, 100);
		oimg.create(itemSize.height*itempercol+10, itemSize.width*itemperrow+10, CV_8UC3);

		char str[20];
		for(int i=0; i<validnum; i++)
		{
			Point pos(i%itemperrow, i/itemperrow);
			Mat curimg;
			resize(imgs[i], curimg, itemSize);
			//sprintf_s(str, "%.3f", db_vals[i].y);
			//putText(img, string(str), Point(10,50), CV_FONT_NORMAL, 0.5f, CV_RGB(0,0,0));
			//sprintf_s(str, "res%d", i);
			curimg.copyTo(oimg(Rect(pos.x*itemSize.width, pos.y*itemSize.height, itemSize.width, itemSize.height)));

			// output image file
			//cout<<i<<" drawn."<<endl;
			//cout<<pos.x<<" "<<pos.y<<": "<<cur_obj.imgfile<<endl;
		}

		return true;
	}
}
