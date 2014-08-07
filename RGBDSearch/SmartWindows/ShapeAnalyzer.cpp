

#include "ShapeAnalyzer.h"


ShapeAnalyzer::ShapeAnalyzer(void)
{
}

//////////////////////////////////////////////////////////////////////////

bool ShapeAnalyzer::ExtractShapes(const cv::Mat& img, double edgeTh, int contour_mode, vector<BasicShape>& shapes)
{
	cv::Mat grayimg;
	if(img.channels() ==3 )
		cvtColor(img, grayimg, CV_BGR2GRAY);
	else
		grayimg = img.clone();	// should optimize

	cv::Mat edgemap;
	Mat Gx, Gy, Gmag;
	cv::Sobel(grayimg, Gx, CV_32F, 1, 0);
	cv::Sobel(grayimg, Gy, CV_32F, 0, 1);
	magnitude(Gx, Gy, Gmag);
	normalize(Gmag, Gmag, 1, 0, NORM_MINMAX);
	threshold(Gmag, edgemap, edgeTh, 255, CV_THRESH_BINARY);
	/*cv::dilate(edgemap, edgemap, cv::Mat());
	cv::dilate(edgemap, edgemap, cv::Mat());
	cv::erode(edgemap, edgemap, cv::Mat());
	cv::erode(edgemap, edgemap, cv::Mat());*/
	//cv::erode(edgemap, edgemap, cv::Mat());
	edgemap.convertTo(edgemap, CV_8U);
	cv::imshow("edge", edgemap);
	cv::waitKey(10);

	// connect broken lines
	//dilate(edgemap, edgemap, Mat(), Point(-1,-1));
	//erode(edgemap, edgemap, Mat(), Point(-1,-1));

	// detect contours and draw
	cv::Mat edge_copy;
	edgemap.copyTo(edge_copy);
	Contours curves;
	std::vector<cv::Vec4i> hierarchy;
	findContours( edge_copy, curves, hierarchy, contour_mode, CV_CHAIN_APPROX_SIMPLE );

	shapes.clear();
	shapes.reserve(2*curves.size());
	for(size_t i=0; i<curves.size(); i++)
	{
		BasicShape cur_shape;
		cur_shape.original_contour = curves[i];
		approxPolyDP(cur_shape.original_contour, cur_shape.approx_contour, cv::arcLength(cv::Mat(cur_shape.original_contour), true)*0.02, true);
		cur_shape.minRect = minAreaRect( cur_shape.approx_contour );
		cur_shape.bbox = boundingRect(cur_shape.approx_contour);
		cur_shape.area = contourArea(curves[i]);
		cur_shape.perimeter = arcLength(curves[i], true);
		cur_shape.isConvex = isContourConvex(cur_shape.approx_contour);
		shapes.push_back( cur_shape );
	}

	return true;
}