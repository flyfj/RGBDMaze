//////////////////////////////////////////////////////////////////////////
//  detector test interface for opencv
//  jie feng@2012-11-10
//////////////////////////////////////////////////////////////////////////


#include "SalientRegionDetector.h"


float compute_downsample_ratio(Size oldSz, float downSampleFactor, Size& newSz)
{
	int imgWidth = oldSz.width;
	int imgHeight = oldSz.height;
	int newWidth = imgWidth, newHeight = imgHeight;
	float down_ratio;
	if (downSampleFactor < 1)		// downSampleFactor is in percentage
	{
		down_ratio = downSampleFactor;
		newWidth = imgWidth * down_ratio + 0.5;
		newHeight = imgHeight * down_ratio + 0.5;
	}
	else if (max(imgWidth, imgHeight) > downSampleFactor)
		// downsize image such that the longer dimension equals downSampleFactor (in pixel), aspect ratio is preserved
	{		
		if (imgWidth > imgHeight)
		{
			newWidth = (int)downSampleFactor;
			newHeight = (int)((float)(newWidth*imgHeight)/imgWidth);
			down_ratio = (float)newWidth / imgWidth;
		}
		else
		{			
			newHeight = downSampleFactor;
			newWidth = (int)((float)(newHeight*imgWidth)/imgHeight);
			down_ratio = (float)newHeight / imgHeight;
		}
	}
	else	// if smaller than specified dimension, ignore resize
	{
		down_ratio = 1;
	}

	newSz.width = newWidth;
	newSz.height = newHeight;

	return down_ratio;
}


int main()
{
	SalientRegionDetector detector;

	// set params
	string imgname = "E:\\Images\\1_26_26640.jpg";
		
	// read image
	Mat img = imread(imgname.c_str());

	// downsample image
	float down_ratio;
	Size newSz;
	Size oldSz(img.cols, img.rows);
	down_ratio = compute_downsample_ratio(oldSz, detector.g_para.downSampleFactor, newSz);
	// resize image
	Mat newImg(newSz, img.depth());
	resize(img, newImg, Size(newImg.cols, newImg.rows));

	imshow("input", img);
	waitKey(10);

	detector.Init(newImg);

	// run detection
	int num_scales = detector.RunMultiSlidingWindow();


	// save result
	{

		// upsample result size to original image
		if (down_ratio != 1)
			for (unsigned int i = 0; i < detector.all_objs.size(); i++)
			{
				for( size_t j = 0; j < detector.all_objs[i].size(); j++ )
				{
					ScoredRect& rc = detector.all_objs[i][j];
					rc.x = rc.x / down_ratio + 0.5;
					rc.y = rc.y / down_ratio + 0.5;
					rc.width = rc.width / down_ratio + 0.5;
					rc.height = rc.height / down_ratio + 0.5;
				}
			}

			// save result image (original size)
			for( size_t i = 0; i < detector.nms_vals.size(); i++ )
			{
				char str[30];
				sprintf(str, "%.2f", detector.nms_vals[i]);

				detector.DrawResult(img, down_ratio, detector.all_objs[i]);

				imshow("result", img);
				waitKey(0);
			}

			//if(detector.g_para.useBGMap)
			//{
			//	// save bg map image
			//	detector.SaveBGMap(saveimgprefix);
			//}

	}


	return 0;
}