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
			for(int x=0; x<width; x++)
			{
				val.b = img.at<cv::Vec3b>(y, x).val[0];
				val.g = img.at<cv::Vec3b>(y, x).val[1];
				val.r = img.at<cv::Vec3b>(y, x).val[2];
				input.access[y][x] = val;
			}
		}

		image<int> index(width, height);	//index matrix, each pixel value is its object id (0~object_num)
		int num_ccs;
		image<rgb> *seg = segment_image(&input, m_dSmoothSigma, m_dThresholdK, m_dMinArea, &num_ccs, &index);	

		// set up segment image
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
		cv::Mat mean_img = img.clone();
		double maxval, minval;
		cv::minMaxLoc(m_idxImg, &minval, &maxval);
		for(int i=minval; i<=maxval; i++)
		{
			cv::Mat cur_mask;
			cv::compare(m_idxImg, cv::Scalar(i), cur_mask, cv::CMP_EQ);
			cv::Scalar cur_mean = cv::mean(img, cur_mask);
			mean_img.setTo(cur_mean, cur_mask);
		}


		return num_ccs;

	}
}
