#include "VideoObjSegmentor.h"


namespace rgbdvision
{
	VideoObjSegmentor::VideoObjSegmentor(void)
	{
	}

	bool VideoObjSegmentor::MaskBoundingBox(const cv::Mat& mask, cv::Rect& box)
	{

		// find biggest connected component as object mask
		cv::Mat mask_back = mask.clone();
		cv::imshow("mask", mask_back*255);
		cv::waitKey(0);
		vector<vector<cv::Point> > contours;
		vector<cv::Vec4i> hierarchy;
		/// Find contours
		findContours( mask_back, contours, hierarchy, CV_RETR_TREE, CV_CHAIN_APPROX_SIMPLE, cv::Point(0, 0) );

		int maxarea = 0;
		for(size_t i=0; i<contours.size(); i++)
		{
			cv::Rect bbox = cv::boundingRect(contours[i]);
			int barea = cv::contourArea(contours[i]);
			if(barea > maxarea)
			{
				box = bbox;
				maxarea = barea;
			}
		}

		return true;

		int minx = mask.cols;
		int miny = mask.rows;
		int maxx = 0;
		int maxy = 0;

		for(int r=0; r<mask.rows; r++)
		{
			for(int c=0; c<mask.cols; c++)
			{
				if(mask.at<uchar>(r,c) > 0)
				{
					minx = MIN(minx, c);
					miny = MIN(miny, r);
					maxx = MAX(maxx, c);
					maxy = MAX(maxy, r);
				}
			}
		}

		box.x = minx;
		box.y = miny;
		box.width = maxx - minx;
		box.height = maxy - miny;

		return true;
	}

	bool VideoObjSegmentor::LoadVideoFrames(const string& frame_dir, int start_id, int end_id)
	{
		frames.clear();
		frames.resize(end_id-start_id+1);

		char str[50];
		for(int i=start_id; i<=end_id; i++)
		{
			sprintf_s(str, "%02d_color.png", i);
			string imgfile = frame_dir + string(str);
			frames[i-start_id] = cv::imread(imgfile);
			cv::Size old_sz(frames[i-start_id].cols, frames[i-start_id].rows);
			// resize
			cv::resize(frames[i-start_id], frames[i-start_id], cv::Size(old_sz.width/2, old_sz.height/2));
		}

		return true;
	}

	bool VideoObjSegmentor::DoSegmentation(const string& frame_dir, int start_id, int end_id)
	{
		if( !LoadVideoFrames(frame_dir, start_id, end_id) )
			return false;

		fgMasks.clear();
		fgMasks.resize(frames.size());

		// user helps cut first frame
		obj_segmentor.InteractiveCut(frames[0], fgMasks[0]);

		// propagate to other frames
		cv::Rect box;
		MaskBoundingBox(fgMasks[0], box);

		char str[30];
		for(int i=1; i<frames.size(); i++)
		{
			cv::Mat disp_img = frames[i].clone();
			cv::rectangle(disp_img, box, CV_RGB(0, 0, 255));
			cv::imshow("frame", disp_img);

			// expand box by ratio
			float ratio = 0.2f;
			box.x = box.x - (int)(box.width * ratio / 2);
			box.y = box.y - (int)(box.height * ratio / 2);
			box.width = (int)(box.width * (1+ratio));
			box.height = (int)(box.height * (1+ratio));
			box.x = MAX(box.x, 0);
			box.y = MAX(box.y, 0);
			box.width = MIN(frames[i].cols-box.x-1, box.width);
			box.height = MIN(frames[i].rows-box.y-1, box.height);

			cv::rectangle(disp_img, box, CV_RGB(0, 255, 0));
			cv::imshow("frame", disp_img);
			cv::waitKey(10);

			obj_segmentor.RunGrabCut(frames[i], fgMasks[i], box, true);

			// save to file
			sprintf_s(str, "seg%d.jpg", i);
			string savefile = frame_dir + string(str);
			cv::Mat trimap = frames[i].clone();
			trimap.setTo(cv::Vec3b(0, 0, 255), fgMasks[i]);

			cv::imwrite(savefile, trimap);

			cv::waitKey(0);

			// update box
			MaskBoundingBox(fgMasks[i], box);
		}

		return true;
	}

}



