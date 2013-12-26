#include "VideoObjSegmentor.h"


namespace rgbdvision
{
	VideoObjSegmentor::VideoObjSegmentor(void)
	{
		invF = (cv::Mat_<float>(3,3) << 594.21, 0, 0.5385, 0, 591.04, 0.4039, 0, 0, 1);
	}

	bool VideoObjSegmentor::ExpandBox(const cv::Rect oldBox, cv::Rect& newBox, float ratio, int imgWidth, int imgHeight)
	{
		newBox.x = oldBox.x - (int)(oldBox.width * ratio / 2);
		newBox.y = oldBox.y - (int)(oldBox.height * ratio / 2);
		newBox.width = (int)(oldBox.width * (1+ratio));
		newBox.height = (int)(oldBox.height * (1+ratio));
		newBox.x = MAX(newBox.x, 0);
		newBox.y = MAX(newBox.y, 0);
		newBox.width = MIN(imgWidth-newBox.x-1, newBox.width);
		newBox.height = MIN(imgHeight-newBox.y-1, newBox.height);

		return true;
	}

	bool VideoObjSegmentor::MaskBoundingBox(const cv::Mat& mask, cv::Rect& box)
	{

		// find biggest connected component as object mask
		cv::Mat mask_back, disp_mask;
		mask_back = mask.clone() * 255;
		cv::cvtColor(mask_back, disp_mask, CV_GRAY2BGR);

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

		// draw box
		/*cv::rectangle(disp_mask, box, CV_RGB(255,0,0));
		cv::imshow("mask", disp_mask);
		cv::waitKey(10);*/

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

	bool VideoObjSegmentor::OutputMaskToFile(ofstream& out, const cv::Mat& color_img, const cv::Mat& mask, bool hasProb)
	{
		if(mask.empty())
		{
			std::cerr<<"Empty mask."<<std::endl;
			return false;
		}

		for(int r=0; r<mask.rows; r++)
		{
			for(int c=0; c<mask.cols; c++)
			{
				if(hasProb)
				{
					// TODO: implement this
					cv::Vec3b cur_color = color_img.at<cv::Vec3b>(r,c);
					//out<<(c==0? "": " ")<<(mask.at<uchar>(r,c)>0? )
				}
				else
				{
					out<<(c==0? "": " ")<<(int)mask.at<uchar>(r,c);
				}
			}
			out<<std::endl;
		}

		return true;
	}

	bool VideoObjSegmentor::LoadBinaryDepthmap(const string& filename, cv::Mat& dmap, int w, int h)
	{
		dmap.create(h, w, CV_32F);

		ifstream in(filename, ios::binary);
		if( !in.is_open() )
			return false;

		// get file size
		in.seekg (0, in.end);
		int length = in.tellg();
		in.seekg (0, in.beg);

		// verify
		assert( length == w*h*sizeof(float) );

		// read data
		vector<float> data(length / sizeof(float) + 1);
		in.read((char*)(&data[0]), length);

		for(int r=0; r<h; r++)
		{
			for(int c=0; c<w; c++)
				dmap.at<float>(r,c) = data[r*w+c];
		}

		return true;
	}

	bool VideoObjSegmentor::LoadMat(const string& filename, cv::Mat& rmat, int w, int h)
	{
		ifstream in(filename);
		if( !in.is_open() )
			return false;

		rmat.create(h, w, CV_32F);
		for(int r=0; r<h; r++)
		{
			for(int c=0; c<w; c++)
			{
				in>>rmat.at<float>(r,c);
			}
		}

		return true;
	}

	bool VideoObjSegmentor::ConvertDmapForDisplay(const cv::Mat& dmap, cv::Mat& dmap_disp)
	{
		dmap_disp = dmap.clone();
		cv::normalize(dmap_disp, dmap_disp, 1, 0, cv::NORM_MINMAX);
		dmap.convertTo(dmap_disp, CV_8U, 255);

		//cv::cvtColor(dmap_disp, dmap_disp, CV_GRAY2BGR);

		return true;
	}

	bool VideoObjSegmentor::Proj2Dto3D(const cv::Mat& fg_mask, const cv::Mat& dmap, const cv::Mat& w2c_mat, std::vector<cv::Vec3f>& pts3d)
	{
		// homogeneous coordinates: (x, y, d)
		cv::Mat homo_coords(0, 0, CV_32F);
		cv::Mat dvalmap(0, 0, CV_32F);
		for(int r=0; r<fg_mask.rows; r++)
		{
			for(int c=0; c<fg_mask.cols; c++)
			{
				if(fg_mask.at<uchar>(r,c) > 0)
				{
					float dval = dmap.at<float>(r,c);
					cv::Vec3f cur_pt(c, r, dval);
					homo_coords.push_back( cv::Mat(cur_pt).t() );
					cv::Vec3f cur_dval(dval, dval, dval);
					dvalmap.push_back( cv::Mat(cur_dval).t() );
				}
			}
		}

		// convert to local coordinates (x, y, z)
		homo_coords = homo_coords / dvalmap;



		return true;
	}

	//////////////////////////////////////////////////////////////////////////

	bool VideoObjSegmentor::LoadVideoFrames(const string& frame_dir, int start_id, int end_id, SegmentInput seg_input)
	{
		frames.clear();
		frames.resize(end_id-start_id+1);

		dmaps.clear();
		dmaps.resize(end_id-start_id+1);

		dmasks.clear();
		dmasks.resize(end_id-start_id+1);

		w2c.clear();
		w2c.resize(end_id-start_id+1);

		char str[50];
		for(int i=start_id; i<=end_id; i++)
		{
			if(seg_input == SEG_RGB || seg_input == SEG_RGBD)
			{
				// load color frame
				sprintf_s(str, "%d_color.png", i);
				string imgfile = frame_dir + string(str);
				frames[i-start_id] = cv::imread(imgfile);
				if( frames[i-start_id].empty() )
					return false;

				// resize
				cv::Size old_sz(frames[i-start_id].cols, frames[i-start_id].rows);
				cv::resize(frames[i-start_id], frames[i-start_id], cv::Size(old_sz.width/2, old_sz.height/2));

				// visual inspection
				cv::imshow("color frame", frames[i-start_id]);
				cv::waitKey(10);
			}
			
			if(seg_input == SEG_RGBD)
			{
				// load depth map
				sprintf_s(str, "%d_depth.bin", i);
				string dmapfile = frame_dir + string(str);
				if( !LoadBinaryDepthmap(dmapfile, dmaps[i-start_id], 640, 480) )
				{
					std::cerr<<"Fail to load depth map."<<std::endl;
					return false;
				}

				cv::compare(dmaps[i-start_id], 0, dmasks[i-start_id], cv::CMP_GT);

				cv::Size old_sz(dmaps[i-start_id].cols, dmaps[i-start_id].rows);
				cv::resize(dmaps[i-start_id], dmaps[i-start_id], cv::Size(old_sz.width/2, old_sz.height/2));
				cv::resize(dmasks[i-start_id], dmasks[i-start_id], cv::Size(old_sz.width/2, old_sz.height/2));

				cv::Mat dmap_disp;
				ConvertDmapForDisplay(dmaps[i-start_id], dmap_disp);
				cv::imshow("depth frame", dmap_disp);
				cv::waitKey(10);

				// load w2c matrix
				sprintf_s(str, "%d_w2c.txt", i);
				string cmatfile = frame_dir + string(str);
				if( !LoadMat(cmatfile, w2c[i-start_id], 4, 4) )
				{
					std::cerr<<"Fail to load w2c matrix."<<std::endl;
					return false;
				}
			}

			cout<<"Loaded "<<i<<endl;
		}

		cv::destroyAllWindows();

		return true;
	}

	bool VideoObjSegmentor::DoSegmentation(const string& frame_dir, int start_id, int end_id, SegmentInput seg_input)
	{
		if(start_id > end_id)
		{
			cerr<<"Invalid start and end frame id."<<endl;
			return false;
		}

		if(seg_input != SEG_RGB && seg_input != SEG_RGBD)
		{
			cerr<<"Only support RGB and RGBD segmentation."<<endl;
			return false;
		}

		if( !LoadVideoFrames(frame_dir, start_id, end_id, seg_input) )
			return false;

		if(seg_input == SEG_RGB)
		{
			obj_segmentor.grabcutter.DATA_CONFIG = visualsearch::GC_DATA_RGB;
			obj_segmentor.grabcutter.SMOOTH_CONFIG = visualsearch::GC_SMOOTH_RGB;
		}
		if(seg_input == SEG_RGBD)
		{
			obj_segmentor.grabcutter.DATA_CONFIG = visualsearch::GC_DATA_RGB;
			obj_segmentor.grabcutter.SMOOTH_CONFIG = visualsearch::GC_SMOOTH_DEPTH;
		}

		fgMasks.clear();
		fgMasks.resize(frames.size());

		// user helps cut first frame
		obj_segmentor.InteractiveCut(frames[0], dmaps[0], dmasks[0], fgMasks[0]);

		// test projection
		//vector<cv::Vec3f> pts;
		//Proj2Dto3D(fgMasks[0], dmaps[0], w2c[0], pts);

		// propagate to other frames
		cv::Rect box;
		MaskBoundingBox(fgMasks[0], box);

		char str[30];
		for(int i=1; i<frames.size(); i++)
		{
			cv::Mat disp_img = frames[i].clone();
			cv::rectangle(disp_img, box, CV_RGB(0, 0, 255));
			cv::imshow("cur_frame", disp_img);
			if(seg_input == SEG_RGBD)
			{
				cv::Mat dmap_disp;
				ConvertDmapForDisplay(dmaps[i], dmap_disp);
				cv::imshow("cur_dmap", dmap_disp);
			}
			
			// expand box by ratio
			float ratio = 0.3f;
			cv::Rect newBox;
			ExpandBox(box, newBox, ratio, frames[i].cols, frames[i].rows);

			box = newBox;
			cv::rectangle(disp_img, box, CV_RGB(0, 255, 0));
			cv::imshow("cur_frame", disp_img);

			obj_segmentor.PredictSegmentMask(frames[i], dmaps[i], dmasks[i], fgMasks[i], box, true);
			cv::waitKey(10);

			obj_segmentor.RunRGBDGrabCut(frames[i], dmaps[i], dmasks[i], fgMasks[i], box, false);

			// update box for bg initialization on next frame
			MaskBoundingBox(fgMasks[i], box);

			// save segment image
			sprintf_s(str, "seg_%d", i+start_id);
			string savefile = frame_dir + string(str) + ".jpg";
			cv::Mat trimap = frames[i].clone();
			trimap.setTo(cv::Vec3b(0, 0, 255), fgMasks[i]);
			// scale back for verification
			cv::resize(trimap, trimap, cv::Size(trimap.cols*2, trimap.rows*2));
			cv::imwrite(savefile, trimap);

			// save segment result
			// resize mask back
			cv::resize(fgMasks[i], fgMasks[i], cv::Size(fgMasks[i].cols*2, fgMasks[i].rows*2));

			savefile = frame_dir + string(str) + ".txt";
			std::ofstream out(savefile);
			OutputMaskToFile(out, frames[i], fgMasks[i]);

			if( cv::waitKey(10) == 'q' )
				break;

			std::cout<<"Finish frame "<<i<<std::endl<<std::endl;
		}

		return true;
	}

}



