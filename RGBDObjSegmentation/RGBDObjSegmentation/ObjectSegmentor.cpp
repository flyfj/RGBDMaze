#include "ObjectSegmentor.h"

namespace visualsearch
{
	// init static
	cv::Scalar ObjectSegmentor::BOXDRAWCOLOR = CV_RGB(0, 255, 0);
	cv::Mat ObjectSegmentor::toProcessImg = cv::Mat();
	uchar ObjectSegmentor::grabState = GRAB_NOT_SET;
	cv::Rect ObjectSegmentor::grabBox = cv::Rect();
	cv::Point ObjectSegmentor::grabStartPt = cv::Point(0,0);

	ObjectSegmentor::ObjectSegmentor()
	{
		seg_type = OBJSEG_GRABCUT;
	}

	//////////////////////////////////////////////////////////////////////////

	void ObjectSegmentor::ShowGrabbedImage()
	{
		if( !toProcessImg.empty() && toProcessImg.channels()==3 )
		{
			cv::Mat disp_img = toProcessImg.clone();

			// draw box on a new image
			if(grabState == GRAB_IN_PROCESS || 
				grabState == GRAB_SET)
			{
				cv::rectangle(disp_img, grabBox, BOXDRAWCOLOR, 2);
			}
			cv::imshow("Grab", disp_img);
			cv::waitKey(10);
		}
	}

	void ObjectSegmentor::ResetGrabcut()
	{
		cv::destroyAllWindows();
		grabState = GRAB_NOT_SET;
	}

	void ObjectSegmentor::GrabcutMouseCallback(int event, int x, int y, int, void* params)
	{
		switch( event )
		{
		case CV_EVENT_LBUTTONDOWN: // set rect or GC_BGD(GC_FGD) labels
			{
				if( grabState == GRAB_NOT_SET )
				{
					//cout<<"Start to grab a box"<<endl;
					grabState = GRAB_IN_PROCESS;
					grabStartPt = cv::Point(x, y);
					grabBox = cv::Rect( x, y, 1, 1 );
				}
			}
			break;

		case CV_EVENT_LBUTTONUP:
			if( grabState == GRAB_IN_PROCESS )
			{
				cout<<"Grabbed a box"<<endl;
				grabBox = cv::Rect( 
					cv::Point( MIN(grabStartPt.x, x), MIN(grabStartPt.y, y) ), 
					cv::Point( MAX(grabStartPt.x, x), MAX(grabStartPt.y, y) ) 
					);
				grabState = GRAB_SET;
				ShowGrabbedImage();
			}
			break;

		case CV_EVENT_MOUSEMOVE:
			if( grabState == GRAB_IN_PROCESS )
			{
				//cout<<"Grabbing"<<endl;
				grabBox = cv::Rect( 
					cv::Point( MIN(grabStartPt.x, x), MIN(grabStartPt.y, y) ), 
					cv::Point( MAX(grabStartPt.x, x), MAX(grabStartPt.y, y) ) 
					);
				ShowGrabbedImage();
			}
			break;
		}
	}

	//////////////////////////////////////////////////////////////////////////

	bool ObjectSegmentor::RunGrabCut(const cv::Mat& color_img, cv::Mat& fg_mask, const cv::Rect& box, 
		bool ifcont)
	{
		// run grabcut
		cout<<"Running grabcut..."<<endl;
		double start_t = cv::getTickCount();
		cv::Mat cut_mask;
		if ( !ifcont )
		{
			// new iteration
			segmentor.RunGrabCut(color_img, fg_mask, box, grabcutModel.bgModel, grabcutModel.fgModel, 2, cv::GC_INIT_WITH_RECT);
		}
		else
		{
			// continuous
			// use box to create a initial mask without re-initialize the model
			/*cut_mask.create(color_img.rows, color_img.cols, CV_8U);
			cut_mask.setTo(cv::GC_BGD);
			cut_mask(box).setTo(cv::GC_PR_FGD);*/
			segmentor.RunGrabCut(color_img, fg_mask, box, grabcutModel.bgModel, grabcutModel.fgModel, 2, cv::GC_INIT_WITH_MASK);
		}
		
		cout<<"Grabcut time: "<<(double)(cv::getTickCount() - start_t) / cv::getTickFrequency()<<"s"<<endl;

		// visualize mask
		fg_mask = fg_mask & 1;
		cv::Mat trimap = color_img.clone();
		// convert to rgba for transparent drawing
		//cv::cvtColor(trimap, trimap, CV_BGR2BGRA);
		trimap.setTo(cv::Vec3b(0, 0, 255), fg_mask);

		cv::imshow("Segment", trimap);
		cv::waitKey(10);

		return true;
	}

	bool ObjectSegmentor::RunGrabCut(const cv::Mat& color_img, const cv::Mat& dmap, cv::Mat& fg_mask, const cv::Rect& box, bool ifcont)
	{
		// run grabcut
		cout<<"Running grabcut..."<<endl;
		double start_t = cv::getTickCount();
		cv::Mat cut_mask;
		if ( !ifcont )
		{
			// new iteration
			segmentor.RunGrabCut(color_img, dmap, fg_mask, box, grabcutModel.bgModel, grabcutModel.fgModel, 2, cv::GC_INIT_WITH_RECT);
		}
		else
		{
			// continuous
			segmentor.RunGrabCut(color_img, dmap, fg_mask, box, grabcutModel.bgModel, grabcutModel.fgModel, 2, cv::GC_INIT_WITH_MASK);
		}
		
		cout<<"Grabcut time: "<<(double)(cv::getTickCount() - start_t) / cv::getTickFrequency()<<"s"<<endl;

		// visualize mask
		fg_mask = fg_mask & 1;
		cv::Mat trimap = color_img.clone();
		// convert to rgba for transparent drawing
		//cv::cvtColor(trimap, trimap, CV_BGR2BGRA);
		trimap.setTo(cv::Vec3b(0, 0, 255), fg_mask);

		cv::imshow("Segment", trimap);
		cv::waitKey(10);

		return true;
	}

	//////////////////////////////////////////////////////////////////////////

	bool ObjectSegmentor::InteractiveCut(const cv::Mat& img, cv::Mat& fg_mask)
	{
		// reset
		ResetGrabcut();

		img.copyTo(toProcessImg);

		// set up mouse callback
		ShowGrabbedImage();	// must show window first
		cv::setMouseCallback("Grab", GrabcutMouseCallback);

		bool ifCut = false;

		while(1)
		{
			int res = cv::waitKey(0);
			if(res == 'r')
			{
				grabState = GRAB_NOT_SET;
				ShowGrabbedImage();
				ifCut = false;
			}
			if(res == 'n')
			{
				if(ObjectSegmentor::grabState == GRAB_SET)
				{
					// do grabcut
					RunGrabCut(toProcessImg, fg_mask, grabBox);
					ifCut = true;
				}
			}
			if(res == 'q')
			{
				break;
			}
		}

		return true;
	}

	bool ObjectSegmentor::InteractiveCut(const cv::Mat& img, const cv::Mat& dmap, cv::Mat& fg_mask)
	{
		// reset
		ResetGrabcut();

		img.copyTo(toProcessImg);

		// set up mouse callback
		ShowGrabbedImage();	// must show window first
		cv::setMouseCallback("Grab", GrabcutMouseCallback);

		bool ifCut = false;

		while(1)
		{
			int res = cv::waitKey(0);
			if(res == 'r')
			{
				grabState = GRAB_NOT_SET;
				ShowGrabbedImage();
				ifCut = false;
			}
			if(res == 'n')
			{
				if(ObjectSegmentor::grabState == GRAB_SET)
				{
					// do grabcut
					RunGrabCut(toProcessImg, dmap, fg_mask, grabBox);
					ifCut = true;
				}
			}
			if(res == 'q')
			{
				break;
			}
		}

		return true;
	}
}

