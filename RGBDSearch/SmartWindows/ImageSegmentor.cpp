//////////////////////////////////////////////////////////////////////////


#include "ImageSegmentor.h"
#include "GraphBasedSegmentor/segment-image.h"


namespace visualsearch
{
	ImageSegmentor::ImageSegmentor(void)
	{
		m_dSmoothSigma = 1.5f;
		m_dThresholdK = 100.f;
		m_dMinArea = 200;
	}


	int ImageSegmentor::DoSegmentation(const cv::Mat& img)
	{
		int height = img.rows;
		int width = img.cols;

		image<rgb> input(width, height);
		rgb val;
		for(int y=0; y<height; y++)
		{
			const uchar* row_pt = img.ptr(y);
			for(int x=0; x<width; x++)
			{
				val.b = *row_pt++; //img.at<cv::Vec3b>(y, x).val[0];
				val.g = *row_pt++; //img.at<cv::Vec3b>(y, x).val[1];
				val.r = *row_pt++; //img.at<cv::Vec3b>(y, x).val[2];
				input.access[y][x] = val;
			}
		}

		double start_t = cv::getTickCount();
		
		image<int> index(width, height);	//index matrix, each pixel value is its object id (0~object_num)
		int num_ccs;
		image<rgb> *seg = segment_image(&input, m_dSmoothSigma, m_dThresholdK, m_dMinArea, &num_ccs, &index);	
		
		cout<<"Segmentation time: "<<(getTickCount()-start_t) / getTickFrequency()<<"s."<<endl;

		// set up segment color image for visualization
		m_segImg.create(img.size(), CV_8UC3);
		for(int y = 0; y < height; y++)	for(int x = 0; x < width; x++)
		{
			int loc = y*width*3 + 3*x;
			m_segImg.at<cv::Vec3b>(y,x).val[0] = imRef(seg,x,y).b;
			m_segImg.at<cv::Vec3b>(y,x).val[1] = imRef(seg,x,y).g;
			m_segImg.at<cv::Vec3b>(y,x).val[2] = imRef(seg,x,y).r;
		}

		delete seg;

		// set up index image and create superpixels
		superPixels.clear();
		superPixels.resize(num_ccs);
		vector<cv::Point> minpts(num_ccs);
		vector<cv::Point> maxpts(num_ccs);
		for(size_t i=0; i<superPixels.size(); i++)
		{
			superPixels[i].mask.create(m_segImg.rows, m_segImg.cols, CV_8U);
			superPixels[i].mask.setTo(0);
			minpts[i].x = m_segImg.cols;
			minpts[i].y = m_segImg.rows;
			maxpts[i].x = 0;
			maxpts[i].y = 0;
		}

		m_idxImg.create(height, width, CV_32S);
		for(int y = 0; y < height; y++) for(int x = 0; x < width; x++)
		{
			int cur_id = index.access[y][x];
			m_idxImg.at<int>(y, x) = cur_id;
			superPixels[cur_id].mask.at<uchar>(y, x) = 1;

			if(x < minpts[cur_id].x)
				minpts[cur_id].x = x;
			if(x > maxpts[cur_id].x)
				maxpts[cur_id].x = x;
			if(y < minpts[cur_id].y)
				minpts[cur_id].y = y;
			if(y > maxpts[cur_id].y)
				maxpts[cur_id].y = y;
		}
		for(size_t i=0; i<superPixels.size(); i++)
		{
			superPixels[i].box.x = minpts[i].x;
			superPixels[i].box.y = minpts[i].y;
			superPixels[i].box.width = maxpts[i].x - minpts[i].x;
			superPixels[i].box.height = maxpts[i].y - minpts[i].y;
		}

		// compute mean image
		m_mean_img = img.clone();
		double maxval, minval;
		cv::minMaxLoc(m_idxImg, &minval, &maxval);
		for(int i=minval; i<=maxval; i++)
		{
			cv::Mat cur_mask;
			cv::compare(m_idxImg, cv::Scalar(i), cur_mask, cv::CMP_EQ);
			cv::Scalar cur_mean = cv::mean(img, cur_mask);
			superPixels[i].meancolor = cur_mean;
			m_mean_img.setTo(cur_mean, cur_mask);
		}

		return num_ccs;
	}

	bool ImageSegmentor::ComputeAdjacencyMat(const std::vector<SuperPixel>& sps, cv::Mat& adjacencyMat)
	{
		if(sps.empty())
		{
			std::cerr<<"Empty superpixels."<<std::endl;
			return false;
		}
		
		adjacencyMat.create(sps.size(), sps.size(), CV_8U);
		adjacencyMat.setTo(0);

		// map all sp index to single image
		cv::Mat idx_map(sps[0].mask.rows, sps[0].mask.cols, CV_32S);
		idx_map.setTo(-1);
		int mask_sum = 0;
		for(size_t i=0; i<sps.size(); i++)
		{
			idx_map.setTo(i, sps[i].mask);
			mask_sum += cv::countNonZero(sps[i].mask);
		}
		// check if masks are complete
		if(mask_sum != idx_map.rows*idx_map.cols)
		{
			std::cerr<<"Not valid superpixel masks."<<std::endl;
			return false;
		}

		// set up adjacency mat
		for(int r=0; r<idx_map.rows; r++)
		{
			for(int c=0; c<idx_map.cols; c++)
			{
				int cur_sp_id = idx_map.at<int>(r, c);
				if(r>0)
				{
					int top_sp_id = idx_map.at<int>(r-1, c);
					if(cur_sp_id != top_sp_id)
					{
						//cout<<cur_sp_id<<" "<<top_sp_id<<endl;
						adjacencyMat.at<uchar>(cur_sp_id, top_sp_id) = adjacencyMat.at<uchar>(top_sp_id, cur_sp_id) = 1;
					}
				}
				if(r<idx_map.rows-1)
				{
					int bottom_sp_id = idx_map.at<int>(r+1, c);
					if(cur_sp_id != bottom_sp_id)
					{
						//cout<<cur_sp_id<<" "<<bottom_sp_id<<endl;
						adjacencyMat.at<uchar>(cur_sp_id, bottom_sp_id) = adjacencyMat.at<uchar>(bottom_sp_id, cur_sp_id) = 1;
					}
				}
				if(c>0)
				{
					int left_sp_id = idx_map.at<int>(r, c-1);
					if(cur_sp_id != left_sp_id)
					{
						//cout<<cur_sp_id<<" "<<left_sp_id<<endl;
						adjacencyMat.at<uchar>(cur_sp_id, left_sp_id) = adjacencyMat.at<uchar>(left_sp_id, cur_sp_id) = 1;
					}
				}
				if(c<idx_map.cols-1)
				{
					int right_sp_id = idx_map.at<int>(r, c+1);
					if(cur_sp_id != right_sp_id)
					{
						//cout<<cur_sp_id<<" "<<right_sp_id<<endl;
						adjacencyMat.at<uchar>(cur_sp_id, right_sp_id) = adjacencyMat.at<uchar>(right_sp_id, cur_sp_id) = 1;
					}
				}
			}
		}

		return true;
	}
}
