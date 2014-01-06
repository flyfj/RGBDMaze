#include "ObjectSegmentor.h"

namespace visualsearch
{
	// init static
	cv::Scalar ObjectSegmentor::BOXDRAWCOLOR = CV_RGB(0, 255, 0);
	cv::Mat ObjectSegmentor::toProcessImg = cv::Mat();
	cv::Mat ObjectSegmentor::toProcessDmap = cv::Mat();
	uchar ObjectSegmentor::grabState = GRAB_NOT_SET;
	cv::Rect ObjectSegmentor::grabBox = cv::Rect();
	cv::Point ObjectSegmentor::grabStartPt = cv::Point(0,0);

	cv::Mat ObjectSegmentor::rgbd_idx_img = cv::Mat();
	std::vector<visualsearch::SuperPixel> ObjectSegmentor::rgbd_superpixels = std::vector<visualsearch::SuperPixel>();


	ObjectSegmentor::ObjectSegmentor()
	{
		seg_type = OBJSEG_GRABCUT;

		grabcutter.DATA_CONFIG = GC_DATA_RGB;
		grabcutter.SMOOTH_CONFIG = GC_SMOOTH_RGB;
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
			if(!toProcessDmap.empty())
				cv::imshow("dmap", toProcessDmap);
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

	bool ObjectSegmentor::RunRGBDGrabCut(const cv::Mat& color_img, const cv::Mat& dmap, const cv::Mat& dmask, cv::Mat& fg_mask, const cv::Rect& box, bool ifcont)
	{
		// run grabcut
		cout<<"Running grabcut..."<<endl;
		double start_t = cv::getTickCount();
		cv::Mat cut_mask;

		grabcutter.RunRGBDGrabCut(color_img, dmap, dmask, cut_mask, box, 2, visualsearch::GC_MODE_NEW);
		
		cout<<"Grabcut time: "<<(double)(cv::getTickCount() - start_t) / cv::getTickFrequency()<<"s"<<endl;

		// visualize mask
		fg_mask = cut_mask & 1;
		cv::Mat trimap = color_img.clone();
		// convert to rgba for transparent drawing
		//cv::cvtColor(trimap, trimap, CV_BGR2BGRA);
		trimap.setTo(cv::Vec3b(0, 0, 255), fg_mask);

		//cv::resize(trimap, trimap, cv::Size(trimap.cols*2, trimap.rows*2));
		//cv::imwrite("g:\\seg_73.jpg", trimap);

		cv::imshow("Segment", trimap);
		cv::waitKey(10);

		return true;
	}

	//////////////////////////////////////////////////////////////////////////

	bool ObjectSegmentor::InteractiveCut(const cv::Mat& img, const cv::Mat& dmap, const cv::Mat& dmask, cv::Mat& fg_mask)
	{
		// reset
		ResetGrabcut();

		img.copyTo(toProcessImg);
		dmap.copyTo(toProcessDmap);

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
					RunRGBDGrabCut(toProcessImg, dmap, dmask, fg_mask, grabBox);
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

	//////////////////////////////////////////////////////////////////////////

	void ObjectSegmentor::FixationMouseCallback(int event, int x, int y, int, void* params)
	{
		switch( event )
		{
		case CV_EVENT_LBUTTONDOWN:
			{
				// show selected seed superpixel
				cv::imshow("sel_mask", rgbd_superpixels[rgbd_idx_img.at<int>(y,x)].mask);
				cv::waitKey(0);
			}
			break;

		default: break;
		}
	}

	bool ObjectSegmentor::ExtractRGBDSuperpixel(const cv::Mat& color_img, const cv::Mat& depth_map, std::vector<visualsearch::SuperPixel>& rgbd_sps)
	{
		float intersect_overlap_th = 0.9f;
		bool verbose = false;

		std::cout<<endl;

		double start_t = cv::getTickCount();
		// do oversegmentation on color image
		img_segmentor.m_dThresholdK = 100;
		int img_seg_num = img_segmentor.DoSegmentation(color_img);
		std::cout<<"image segments: "<<img_seg_num<<endl;
		cv::Mat color_segimg = img_segmentor.m_segImg.clone();
		std::vector<visualsearch::SuperPixel> img_sps = img_segmentor.superPixels;
		cv::imshow("color_img", color_img);
		cv::imshow("color_seg", img_segmentor.m_segImg);
		cv::waitKey(10);

		// oversegmentation on depth map
		img_segmentor.m_dThresholdK = 15;
		cv::Mat depth_img;
		visualsearch::RGBDTools::ConvertDmapForDisplay(depth_map, depth_img);
		cv::cvtColor(depth_img, depth_img, CV_GRAY2BGR);
		int dmap_seg_num = img_segmentor.DoSegmentation(depth_img);
		std::vector<visualsearch::SuperPixel> dmap_sps = img_segmentor.superPixels;
		std::cout<<"depth segments: "<<dmap_seg_num<<endl;
		cv::Mat depth_segimg = img_segmentor.m_segImg.clone();
		cv::imshow("depth_img", depth_img);
		cv::imshow("depth_seg", img_segmentor.m_segImg);
		cv::waitKey(10);

		std::cout<<"Time cost for two oversegmentations: "<<(cv::getTickCount()-start_t) / cv::getTickFrequency()<<"s"<<endl;

		start_t = cv::getTickCount();
		// compute intersection of the superpixel maps
		rgbd_sps.clear();
		for(size_t i=0; i<img_sps.size(); i++)
		{
			cv::Mat img_sp_mask = img_sps[i].mask.clone();
			for(size_t j=0; j<dmap_sps.size(); j++)
			{
				cv::Mat dmap_sp_mask = dmap_sps[j].mask.clone();
				if( verbose )
				{
					cv::imshow("img_sp", img_sp_mask*255);
					cv::imshow("dmap_sp", dmap_sp_mask*255);
					cv::waitKey(10);
				}

				cv::Mat intersect_mask = img_sp_mask & dmap_sp_mask;
				if(verbose)
					cv::imshow("intersect", intersect_mask*255);

				cv::Mat union_mask = img_sp_mask | dmap_sp_mask;
				float intersect_area = cv::countNonZero(intersect_mask);
				if(intersect_area == 0)
					continue;

				float union_area = cv::countNonZero(union_mask);
				float cur_sp_area = cv::countNonZero(img_sp_mask);
				if( intersect_area / cur_sp_area > intersect_overlap_th )
				{
					// directly add to collection
					visualsearch::SuperPixel new_sp;
					img_sp_mask.copyTo(new_sp.mask);
					rgbd_sps.push_back(new_sp);

					if(verbose)
					{
						std::cout<<"add superpixel as raw sp."<<endl;
						cv::waitKey(0);
					}
					break;
				}
				else
				{
					visualsearch::SuperPixel new_sp;
					intersect_mask.copyTo(new_sp.mask);
					rgbd_sps.push_back(new_sp);

					if(verbose)
					{
						std::cout<<"added intersect superpixel"<<endl;
						cv::waitKey(0);
					}
					img_sp_mask.setTo(0, intersect_mask);
					if(cv::countNonZero(img_sp_mask) == 0)
						break;
				}
			}
		}
		std::cout<<"Time cost for rgbd oversegmentation: "<<(cv::getTickCount()-start_t) / cv::getTickFrequency()<<"s"<<endl;

		if(verbose)
		{
			// compose a segment map to visualize rgbd sps
			cv::RNG rng(30000);
			cv::Mat rgbd_sp_map(color_img.rows, color_img.cols, CV_8UC3);
			rgbd_idx_img.create(color_img.rows, color_img.cols, CV_32S);
			for(size_t i=0; i<rgbd_sps.size(); i++)
			{
				cv::Vec3b cur_color(rng.next() % 256, rng.next() % 256, rng.next() % 256);
				rgbd_sp_map.setTo(cur_color, rgbd_sps[i].mask);
				rgbd_idx_img.setTo(i, rgbd_sps[i].mask);
			}
			std::cout<<"rgbd sp num: "<<rgbd_sps.size()<<endl;
			cv::imshow("rgbd_sps", rgbd_sp_map);
			cv::waitKey(0);

			// show fixated superpixel
			cv::Point fixationPt(150,150);
			cv::Vec3b sel_color(255,255,255);
			int sel_id = rgbd_idx_img.at<int>(fixationPt.y, fixationPt.x);
			rgbd_sp_map.setTo(sel_color, rgbd_sps[sel_id].mask);
			cv::circle(rgbd_sp_map, fixationPt, 5, CV_RGB(0,0,0));
			cv::imshow("sel_seg", rgbd_sp_map);
			cv::waitKey(0);
		}

		return true;
	}

	bool ObjectSegmentor::FixationCut(const cv::Mat& color_img, const cv::Mat& dmap, cv::Mat& fg_mask)
	{


		return true;
	}

}

