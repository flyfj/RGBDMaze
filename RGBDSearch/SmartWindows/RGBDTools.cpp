//////////////////////////////////////////////////////////////////////////
// implementation
// jiefeng@2014-1-4
//////////////////////////////////////////////////////////////////////////


#include "RGBDTools.h"

namespace visualsearch
{
	bool RGBDTools::LoadMat(const std::string& filename, cv::Mat& rmat, int w, int h)
	{
		std::ifstream in(filename);
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

	bool RGBDTools::LoadBinaryDepthmap(const std::string& filename, cv::Mat& dmap, int w, int h)
	{
		dmap.create(h, w, CV_32F);

		std::ifstream in(filename, std::ios::binary);
		if( !in.is_open() )
			return false;

		// get file size
		in.seekg (0, in.end);
		int length = in.tellg();
		in.seekg (0, in.beg);

		// verify
		assert( length == w*h*sizeof(float) );

		// read data
		std::vector<float> data(length / sizeof(float) + 1);
		in.read((char*)(&data[0]), length);

		for(int r=0; r<h; r++)
		{
			for(int c=0; c<w; c++)
				dmap.at<float>(r,c) = data[r*w+c];
		}

		return true;
	}

	bool RGBDTools::OutputMaskToFile(std::ofstream& out, const cv::Mat& color_img, const cv::Mat& mask, bool hasProb /* = false */)
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
					out<<(c==0? "": " ")<<(int)mask.at<uchar>(r,c);
			}
			out<<std::endl;
		}

		return true;
	}

	//////////////////////////////////////////////////////////////////////////

	bool RGBDTools::ConvertDmapForDisplay(const cv::Mat& dmap, cv::Mat& dmap_disp)
	{
		dmap_disp = dmap.clone();
		cv::normalize(dmap_disp, dmap_disp, 1, 0, cv::NORM_MINMAX);
		dmap.convertTo(dmap_disp, CV_8U, 255);

		//cv::cvtColor(dmap_disp, dmap_disp, CV_GRAY2BGR);
		return true;
	}

	//////////////////////////////////////////////////////////////////////////

	bool RGBDTools::Proj2Dto3D(const cv::Mat& fg_mask, const cv::Mat& dmap, const cv::Mat& w2c_mat, std::vector<cv::Vec3f>& pts3d)
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
}

