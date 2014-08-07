#include "FixationSegmentor.h"

namespace visualsearch
{
	FixationSegmentor::FixationSegmentor(void)
	{
	}

	//////////////////////////////////////////////////////////////////////////

	float FixationSegmentor::SPDist(const SuperPixel& a, const SuperPixel& b)
	{
		float dist = 0;
		dist += fabs(a.feats[0].at<float>(0) - b.feats[0].at<float>(0));
		/*for(int i=0; i<3; i++)
		dist += (a.meancolor.val[i]/255-b.meancolor.val[i]/255)*(a.meancolor.val[i]/255-b.meancolor.val[i]/255);*/

		dist = sqrt(dist);
		return dist;
	}

	float FixationSegmentor::SPCenterDist(const SuperPixel& a, const SuperPixel& b)
	{
		float dist = 0;
		Point acenter(a.box.tl().x+a.box.width/2, a.box.tl().y+a.box.height/2);
		Point bcenter(b.box.tl().x+b.box.width/2, b.box.tl().x+b.box.height/2);
		dist = (acenter.x-bcenter.x)*(acenter.x-bcenter.x) + (bcenter.x-bcenter.y)*(bcenter.x-bcenter.y);
		dist = sqrt(dist);

		return dist;
	}


	bool FixationSegmentor::DoSegmentation(Point fixpt, const Mat& cimg, const Mat& dmap, Mat& objmask)
	{
		ImageSegmentor imgsegmentor;
		imgsegmentor.m_dThresholdK = 50;
		imgsegmentor.DoSegmentation(cimg);
		imshow("seg", imgsegmentor.m_segImg);
		int sel_spid = imgsegmentor.m_idxImg.at<int>(fixpt);
		imshow("segmask", imgsegmentor.superPixels[sel_spid].mask*255);
		imshow("input", cimg);
		imshow("dmap", dmap);
		//ImgVisualizer::DrawFloatImg("dmap", dmap, Mat());
		waitKey(0);

		vector<SuperPixel>& sps = imgsegmentor.superPixels;

		// compute meancolor in lab
		Mat labimg;
		cvtColor(cimg, labimg, CV_BGR2Lab);
		for (size_t i=0; i<sps.size(); i++)
		{
			Mat meand(1, 1, CV_32F);
			meand.at<float>(0,0) = mean(dmap, sps[i].mask).val[0];
			sps[i].feats.push_back(meand);
			sps[i].meancolor = mean(labimg, sps[i].mask);
		}

		// compute distance map for selected sp
		Mat distmap(cimg.rows, cimg.cols, CV_32F);
		vector<float> rawdists(sps.size());
		for (size_t i=0; i<sps.size(); i++)
		{
			rawdists[i] = SPDist(sps[sel_spid], sps[i]);
			distmap.setTo(rawdists[i], sps[i].mask);
		}
		ImgVisualizer::DrawFloatImg("distmap", distmap, Mat());
		waitKey(0);

		Mat adjacencyMat;
		imgsegmentor.ComputeAdjacencyMat(sps, adjacencyMat);
		
		// compute neighbor (dis)similarity
		vector<vector<PII> > edges(sps.size());
		for (int r=0; r<adjacencyMat.rows; r++)
			for(int c=0; c<adjacencyMat.cols; c++)
			{
				if(adjacencyMat.at<uchar>(r,c) > 0)
				{
					edges[r].push_back(make_pair(SPDist(sps[r], sps[c]), c));
				}
			}

		// compute shortest path
		int srcsp = imgsegmentor.m_idxImg.at<int>(fixpt);
		Dijkstra sspath;
		vector<float> dists;
		sspath.ComputeSShortestPath(edges, srcsp, dists);

		// draw saliency map
		Mat salmap(cimg.rows, cimg.cols, CV_32F);
		for (size_t i=0; i<dists.size(); i++)
			salmap.setTo(dists[i]/SPCenterDist(sps[sel_spid], sps[i]), sps[i].mask);
		normalize(salmap, salmap, 0, 1, NORM_MINMAX);
		salmap = 1 - salmap;
		ImgVisualizer::DrawFloatImg("salmap", salmap, Mat());
		waitKey(0);

		objmask.create(salmap.rows, salmap.cols, CV_8U);
		objmask.setTo(GC_PR_FGD);
		objmask.setTo(GC_PR_BGD, salmap<0.5);
		objmask.setTo(GC_BGD, salmap<0.3);	// could change this threshold to generate multiple segments
		objmask.setTo(GC_FGD, salmap==1);
		imshow("inputmask", objmask*50);
		waitKey(0);

		// do graph-cut
 		cv::Mat bgModel, fgModel;
		Rect box(0, 0, cimg.cols, cimg.rows);
		cv::grabCut(cimg, objmask, box, bgModel, fgModel, 3, cv::GC_INIT_WITH_MASK);

		cv::Mat trimap = cimg.clone();
		objmask = objmask & GC_FGD;
		trimap.setTo(cv::Vec3b(0, 0, 255), objmask);
		// show fg boundary
		Mat boundaryMask;
		Canny(objmask, boundaryMask, 0.2, 0.5);
		trimap.setTo(cv::Vec3b(255, 0, 0), boundaryMask);
		imshow("objmask", trimap);
		waitKey(0);

		return true;
	}
}
