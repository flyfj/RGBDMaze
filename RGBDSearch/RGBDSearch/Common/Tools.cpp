

#include "Tools.h"

namespace tools
{
	//////////////////////////////////////////////////////////////////////////
	double ToolFactory::hist_l2_dist(const Feature& a, const Feature& b)
	{
		double sum = 0;
		for(size_t i=0; i<a.size(); i++)
		{
			double min_dist = (a[i]-b[i])*(a[i]-b[i]);
			sum += min_dist;
		}

		return sqrt(sum);
	}

	double ToolFactory::hist_intersection_dist(const Feature& a, const Feature& b)
	{
		double sum = 0;
		for(size_t i=0; i<a.size(); i++)
			sum += min(a[i], b[i]);

		return 1 - sum;
	}

	double ToolFactory::L2_DIST(const Point2f& a, const Point2f& b)
	{
		return sqrt( (a.x-b.x)*(a.x-b.x)+(a.y-b.y)*(a.y-b.y) );
	}


	//////////////////////////////////////////////////////////////////////////
#ifndef IOS

	void ToolFactory::GetFilesFromDir(const string& dir, const string& type, FileInfos& fileInfos)
	{
		fileInfos.clear();
		FileInfo fileinfo;

		struct _finddata_t ffblk;
		int done = 1;
		string filelist = dir + type;	// TODO: is there a way to find multiple ext files without doing several times?
		int handle = _findfirst(filelist.c_str(), &ffblk);
		if(handle != -1)
		{
			while( done != -1 )
			{
				string fname(ffblk.name);
				//fname = fname.substr(0, fname.length()-4);	// image name without extension
				string path = dir + fname;
				fileinfo.filename = fname;
				fileinfo.filepath = path;
				fileInfos.push_back(fileinfo);

				done = _findnext(handle, &ffblk);
			}
		}

	}

	void ToolFactory::GetDirsFromDir(const string& dir, DirInfos& dirInfos, const string& outputFile)
	{
		dirInfos.clear();
		DirInfo dirInfo;

		struct _finddata_t ffblk;
		int done = 1;
		string files = dir + "*.*";	// TODO: is there a way to find multiple ext files without doing several times?
		int handle = _findfirst(files.c_str(), &ffblk);
		if(handle != -1)
		{
			while( done != -1 )
			{
				// _A_SUBDIR is not right
				if(ffblk.attrib == 2064)
				{
					string fname(ffblk.name);
					if(fname != "." && fname != "..")
					{
						dirInfo.dirname = fname;
						dirInfo.dirpath = dir + fname + "\\";
						dirInfos.push_back(dirInfo);
					}
				}
				
				done = _findnext(handle, &ffblk);
			}
		}

		if(!dirInfos.empty())
		{
			ofstream out(outputFile.c_str());
			if(out.is_open())
			{
				for(size_t i=0; i<dirInfos.size(); i++)
					out<<dirInfos[i].dirname<<endl;
			}
			
		}
	}

	void ToolFactory::RemoveEmptyDir(const string& dir, const string& type)
	{
		DirInfos dir_infos;
		GetDirsFromDir(dir, dir_infos);
		for(size_t i=0; i<dir_infos.size(); i++)
		{
			FileInfos info;
			GetFilesFromDir(dir_infos[i].dirpath, type, info);
			if(info.empty())
			{
				_rmdir(dir_infos[i].dirpath.c_str());
				cout<<"Remove "<<dir_infos[i].dirpath<<endl;
			}

			cout<<"Finish "<<dir_infos[i].dirpath<<endl;
		}
	}

#endif

	//////////////////////////////////////////////////////////////////////////
	bool ToolFactory::compFileNums(const DirInfo& a, const DirInfo& b)
	{
		return a.filenum > b.filenum;
	}

	bool ToolFactory::compValuePairsAsce(const Point2f& a, const Point2f& b)
	{
		if(a.y < b.y)
			return true;
		else if(a.y == b.y)
			return a.x < b.x;
		else
			return false;
	}

	bool ToolFactory::compValuePairsDesc(const Point2f& a, const Point2f& b)
	{
		if(a.y > b.y)
			return true;
		else if(a.y == b.y)
			return a.x < b.x;
		else
			return false;
	}

	bool ToolFactory::compValueTriplesAsce(const Point3f& a, const Point3f& b)
	{
		return a.z < b.z;
	}

	bool ToolFactory::compValueTriplesDesc(const Point3f& a, const Point3f& b)
	{
		return a.z > b.z;
	}

	bool ToolFactory::compNameValuePairsAsce(const NameValuePair& a, const NameValuePair& b)
	{
		return a.value < b.value;
	}

	bool ToolFactory::compNameValuePairsDesc(const NameValuePair& a, const NameValuePair& b)
	{
		return a.value > b.value;
	}

	bool ToolFactory::compScoredRectAsce(const ScoredRect& a, const ScoredRect& b)
	{
			return a.score < b.score;
	}

	bool ToolFactory::compScoredRectDesc(const ScoredRect& a, const ScoredRect& b)
	{
			return a.score > b.score;
	}

	//////////////////////////////////////////////////////////////////////////
	bool ToolFactory::DrawHist(cv::Mat& canvas, cv::Size canvas_size, int max_val, const cv::Mat& hist)
	{
		if( hist.empty() || hist.rows != 1 || hist.depth() != CV_32F )
		{
			std::cerr<<"Wrong format of histogram to draw."<<std::endl;
			return false;
		}

		/* show histogram */
		// Get scale so the histogram fit the canvas height
		double maxv = max_val;	// upper bound of bin value

		canvas.create(canvas_size.height, canvas_size.width, CV_8UC3);
		canvas.setTo(255);
		double binWidth = (double)canvas.cols / hist.cols;
		double scale = maxv > canvas.rows ? (double)canvas.rows / maxv : 1.;  

		// Draw histogram
		for ( int i = 0; i < hist.cols; i++) 
		{    
			cv::Point pt1(i*binWidth, canvas.rows - (hist.at<float>(0, i) * canvas.rows));
			cv::Point pt2((i+1)*binWidth, canvas.rows);
			cv::rectangle(canvas, pt1, pt2, CV_RGB(0, 255, 0), CV_FILLED);
		}

		return true;

	}


	//////////////////////////////////////////////////////////////////////////
	float ToolFactory::compute_downsample_ratio(Size oldSz, float downSampleFactor, Size& newSz)
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

	void ToolFactory::generateSpaitalGrids(cv::Size imgsz, SpatialPyramidLayout spm_layout, vector<cv::Rect>& grids)
	{
		// grid size
		int gridWidth, gridHeight;
		// grid dimension
		int gridDimX, gridDimY;
		switch (spm_layout)
		{
		case visualsearch::SPM1X1:
			gridWidth = imgsz.width;
			gridHeight = imgsz.height;
			gridDimX = 1;
			gridDimY = 1;
			break;
		case visualsearch::SPM3X3:
			gridWidth = imgsz.width / 3;
			gridHeight = imgsz.height / 3;
			gridDimX = 3;
			gridDimY = 3;
			break;
		case visualsearch::SPM4X4:
			gridWidth = imgsz.width / 4;
			gridHeight = imgsz.height / 4;
			gridDimX = 4;
			gridDimY = 4;
			break;
		case visualsearch::SPM2X2:
			gridWidth = imgsz.width / 2;
			gridHeight = imgsz.height / 2;
			gridDimX = 2;
			gridDimY = 2;
			break;
		case visualsearch::SPM1X3:
			gridWidth = imgsz.width / 3;
			gridHeight = imgsz.height / 1;
			gridDimX = 3;
			gridDimY = 1;
			break;
		case visualsearch::SPM8X8:
			gridWidth = imgsz.width / 8;
			gridHeight = imgsz.height / 8;
			gridDimX = 8;
			gridDimY = 8;
			break;
		default:
			break;
		}

		grids.clear();
		for(int r=0; r<gridDimY; r++)
		{
			for(int c=0; c<gridDimX; c++)
			{
				cv::Rect grid(c*gridWidth, r*gridHeight, gridWidth, gridHeight);
				grids.push_back(grid);
			}
		}

	}

	void ToolFactory::computeTriangleAngles(const cv_KeyPoints& pts, vector<float>& angles)
	{
		angles.clear();
		if(pts.size() != 3)
			return;

		// angles
		angles.resize(3);
		for(int i=0; i<3; i++)
		{
			vector<int> neighbors;
			if(i==0)
			{
				neighbors.push_back(1);
				neighbors.push_back(2);
			}
			if(i==1)
			{
				neighbors.push_back(0);
				neighbors.push_back(2);
			}
			if(i==2)
			{
				neighbors.push_back(1);
				neighbors.push_back(2);
			}

			vector<Point2f> lines(neighbors.size());
			vector<float> line_lens(neighbors.size());
			for(size_t j=0; j<neighbors.size(); j++)
			{
				lines[j] = Point2f(pts[neighbors[j]].pt.x - pts[i].pt.x, pts[neighbors[j]].pt.y - pts[0].pt.y);
				line_lens[j] = sqrt(lines[j].x*lines[j].x + lines[j].y*lines[j].y);
			}

			float dotprodcut = lines[0].x*lines[1].x + lines[0].y*lines[1].y;
			angles[i] = acos(dotprodcut / (line_lens[0]*line_lens[1]) );
		}

		sort(angles.begin(), angles.end());

	}

	double ToolFactory::ComputeNDSampleStd(const cv::Mat& samps)
	{
		// compute mean
		Mat mean_samp;
		meanStdDev(samps, mean_samp, Mat());

		double samps_std = 0;
		for(int r=0; r<samps.rows; r++)
		{
			double tnorm = norm(samps.row(r), mean_samp, NORM_L2);
			samps_std += tnorm * tnorm;
		}
		samps_std /= (samps.rows-1);
		samps_std = sqrt(samps_std);

		return samps_std;
	}

	double ToolFactory::computeEntropy(const vector<double>& distri)
	{
		double result = 0.0;
		for (int b = 0; b < distri.size(); b++)
		{
			double p = (double)distri[b];
			result -= p == 0.0 ? 0.0 : p * log(p)/log(2.0);
		}

		return result;
	}

	void ToolFactory::ComputeSampleWeights(const std::vector<double>& input_val, std::vector<double>& weights)
	{
			weights.clear();
			weights.resize(input_val.size());

			for(size_t i=0; i<input_val.size(); i++)
			{
					weights[i] = 1 - input_val[i];//MEstimateWeight<MEstimator::L1>(input_val[i]);
				/*if(weights[i] < 0)
						cout<<input_val[i]<<endl;*/
			}
	}

	void ToolFactory::SplitClassSamples(const cv::Mat& all_labels, std::vector<std::vector<int>>& class_labels, bool shuffle)
	{
			double maxval, minval;
			cv::minMaxLoc(all_labels, &minval, &maxval);

			class_labels.clear();
			class_labels.resize((int)maxval+1);

			for(int r=0; r<all_labels.rows; r++)
			{
					class_labels[all_labels.at<int>(r)].push_back(r);
			}

			if(shuffle)
			{
					for(size_t i=0; i<class_labels.size(); i++)
							std::random_shuffle(class_labels[i].begin(), class_labels[i].end());
			}
	}

	cv::Rect ToolFactory::RectIntersection(const cv::Rect& a, const cv::Rect& b)
	{
		// check if intersect (one of the point of one rect should be contained in another rect)
		if(a.contains(b.tl()) || a.contains(b.br()) || a.contains(cv::Point(b.x, b.br().y)) || a.contains(cv::Point(b.br().x, b.y)) ||
			b.contains(a.tl()) || b.contains(a.br()) || b.contains(cv::Point(a.x, a.br().y)) || b.contains(cv::Point(a.br().x, a.y)) )
		{
			int minx = MAX(a.x, b.x);
			int miny = MAX(a.y, b.y);
			int maxx = MIN(a.br().x, b.br().x);
			int maxy = MIN(a.br().y, b.br().y);

			cv::Rect res_box(minx, miny, maxx-minx, maxy-miny);
			return res_box;
		}
		else
			return cv::Rect(0, 0, 0, 0);
	}

	cv::Rect ToolFactory::RectUnion(const cv::Rect& a, const cv::Rect& b)
	{
		int minx = MIN(a.x, b.x);
		int miny = MIN(a.y, b.y);
		int maxx = MAX(a.br().x, b.br().x);
		int maxy = MAX(a.br().y, b.br().y);

		cv::Rect res_box(minx, miny, maxx-minx, maxy-miny);
		return res_box;
	}

}