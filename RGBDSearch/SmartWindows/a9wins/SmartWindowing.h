// Confidential, Copyright 2013 A9.com, Inc.

#ifndef LOGO_RECOGNITION_SMART_WINDOW_H_
#define LOGO_RECOGNITION_SMART_WINDOW_H_

#include <vector>

#include "Rectangle.h"
#include <opencv2\opencv.hpp>

namespace
{

typedef std::vector<Shape::Rectangle>::const_iterator neighbor_type;
//typedef const Shape::Rectangle* neighbor_type;
typedef std::vector<neighbor_type > neighbors_type;

} // Anonymous namespace.

void SplitImage(const cv::Mat& integral_image, size_t num_rectangles, std::vector<Shape::Rectangle>& rectangles);
size_t GetNeighborRectangles(const cv::Size image_size, const std::vector<Shape::Rectangle>& rectangles, std::vector<neighbors_type >& neighbors);
void MergeRectangles(const cv::Mat& integral_image, std::vector<Shape::Rectangle>& rectangles, const float threshold, std::vector<std::vector<std::vector<Shape::Rectangle>::iterator> >& segments);
void GetWindows(const cv::Mat& integral_image, std::vector<Shape::Rectangle>& rectangles, const float threshold, const float dist_threshold, std::vector<Shape::Rectangle>& windows, bool append = false);
void GetWindows(const cv::Mat& integral_image, std::vector<Shape::Rectangle>& rectangles, std::vector<float>& thresholds, const float dist_threshold, std::vector<Shape::Rectangle>& windows);
inline void GetWindows(const cv::Mat& integral_image, std::vector<Shape::Rectangle>& rectangles, const float threshold, std::vector<Shape::Rectangle>& windows, bool append = false)
{
  GetWindows(integral_image, rectangles, threshold, 0, windows, append);
}
inline void GetWindows(const cv::Mat& integral_image, std::vector<Shape::Rectangle>& rectangles, std::vector<float>& thresholds, std::vector<Shape::Rectangle>& windows)
{
  GetWindows(integral_image, rectangles, thresholds, 0, windows);
}

#endif  // LOGO_RECOGNITION_SMART_WINDOW_H_