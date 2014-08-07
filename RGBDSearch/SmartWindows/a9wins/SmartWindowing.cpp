// Confidential, Copyright 2013 A9.com, Inc.

#include "SmartWindowing.h"

#include <stdint.h>
#include <algorithm>
#include <limits>
#include <queue>

#include "DisjointSet.h"
#include <opencv2\opencv.hpp>

namespace
{

template<typename S, typename G = S>
struct Split
{
  typedef S total_squared_sum_type;
  typedef G gain_type;

  Shape::Rectangle r, r1, r2;
  total_squared_sum_type s, s1, s2;
  gain_type gain;

  // Comparators.
  inline bool operator<(const Split& split) const {return gain < split.gain;}
  inline bool operator<=(const Split& split) const {return gain <= split.gain;}
  inline bool operator>(const Split& split) const {return gain > split.gain;}
  inline bool operator>=(const Split& split) const {return gain >= split.gain;}
};

struct Edge
{
  typedef DisjointSetVector<Shape::Rectangle>::iterator node_type;
  typedef DisjointSetNode<Shape::Rectangle>::weight_type weight_type;

  // Constructors.
  inline Edge() : weight(0) {}
  inline Edge(const node_type& n1, const node_type& n2, weight_type w = 0) : node1(n1), node2(n2), weight(w) {}

  // Comparators.
  inline bool operator>(const Edge& e) const {return weight > e.weight;}
  inline bool operator>=(const Edge& e) const {return weight >= e.weight;}
  inline bool operator<(const Edge& e) const {return weight < e.weight;}
  inline bool operator<=(const Edge& e) const {return weight <= e.weight;}

  // Union.
  inline bool Union(const weight_type& threshold) {return node1->Union(&*node2, weight, threshold) > 0;}

  // Nodes.
  node_type node1, node2;

  // Edge weight.
  weight_type weight;
};

template<typename integral_type, typename sum_type>
inline void GetSums(const cv::Mat& integral_image, const Shape::Rectangle& patch, sum_type* sums)
{
  const unsigned char channels = (unsigned char)integral_image.channels();
  const int x1 = channels * patch.x1;
  const int x2 = channels * (patch.x2 + 1);
  const int y2 = patch.y2 + 1;
  const integral_type* row1 = (const integral_type*)integral_image.ptr(patch.y1);
  const integral_type* row2 = (const integral_type*)integral_image.ptr(y2);

  const integral_type* p1 = row1 + x1;
  const integral_type* p2 = row1 + x2;
  const integral_type* p3 = row2 + x1;
  const integral_type* p4 = row2 + x2;

  const sum_type* sums_end = sums + channels;
  for (sum_type* sum = sums; sum != sums_end; sum++, p1++, p2++, p3++, p4++)
  {
    *sum = (sum_type)(*p1 - *p2 - *p3 + *p4);
  }
}

template<typename integral_type, typename sum_type>
inline void GetSums(const cv::Mat& integral_image, const Shape::Rectangle& patch, std::vector<sum_type>& sums)
{
  sums.resize(integral_image.channels());
  GetSums<integral_type>(integral_image, patch, sums.data());
}

template<typename integral_type, typename mean_type>
inline void GetMeans(const cv::Mat& integral_image, const Shape::Rectangle& patch, mean_type* means)
{
  const unsigned char channels = (unsigned char)integral_image.channels();
  const int x1 = channels * patch.x1;
  const int x2 = channels * (patch.x2 + 1);
  const int y2 = patch.y2 + 1;
  const integral_type* row1 = (const integral_type*)integral_image.ptr(patch.y1);
  const integral_type* row2 = (const integral_type*)integral_image.ptr(y2);

  const integral_type* p1 = row1 + x1;
  const integral_type* p2 = row1 + x2;
  const integral_type* p3 = row2 + x1;
  const integral_type* p4 = row2 + x2;

  const unsigned int area = RectangleArea(patch);
  const mean_type* means_end = means + channels;
  for (mean_type* mean = means; mean != means_end; mean++, p1++, p2++, p3++, p4++)
  {
    *mean = (mean_type)(*p1 - *p2 - *p3 + *p4) / (mean_type)area;
  }
}

template<typename integral_type, typename mean_type>
inline void GetMeans(const cv::Mat& integral_image, const Shape::Rectangle& patch, std::vector<mean_type>& means)
{
  means.resize(integral_image.channels());
  GetMeans<integral_type>(integral_image, patch, means.data());
}

template<typename integral_type, typename mean_type>
inline void GetMeans(const cv::Mat& integral_image, const std::vector<Shape::Rectangle>& rectangles, std::vector<std::vector<mean_type> >& means)
{
  means.resize(rectangles.size());
  std::vector<std::vector<mean_type> >::iterator mean = means.begin();
  for (std::vector<Shape::Rectangle>::const_iterator rectangle = rectangles.begin(); rectangle != rectangles.end(); rectangle++, mean++)
  {
    GetMeans<integral_type, mean_type>(integral_image, *rectangle, *mean);
  }
}

template<typename integral_type, typename mean_type>
inline void GetMeans(const cv::Mat& integral_image, const std::vector<Shape::Rectangle>& rectangles, std::vector<mean_type>& means)
{
  const unsigned char channels = (unsigned char)integral_image.channels();
  means.resize(rectangles.size() * channels);
  std::vector<mean_type>::iterator mean = means.begin();
  for (std::vector<Shape::Rectangle>::const_iterator rectangle = rectangles.begin(); rectangle != rectangles.end(); rectangle++, mean += channels)
  {
    GetMeans<integral_type, mean_type>(integral_image, *rectangle, &*mean);
  }
}

template<typename integral_type, typename mean_type>
inline void GetMeans(const cv::Mat& integral_image, const DisjointSetVector<Shape::Rectangle>& disjoint_set, std::vector<std::vector<mean_type> >& means)
{
  means.resize(disjoint_set.size());
  std::vector<std::vector<mean_type> >::iterator mean = means.begin();
  for (DisjointSetVector<Shape::Rectangle>::const_iterator node = disjoint_set.begin(); node != disjoint_set.end(); node++, mean++)
  {
    GetMeans<integral_type, mean_type>(integral_image, *node->value(), *mean);
  }
}

template<typename integral_type, typename mean_type>
inline void GetMeans(const cv::Mat& integral_image, const DisjointSetVector<Shape::Rectangle>& disjoint_set, std::vector<mean_type>& means)
{
  const unsigned char channels = (unsigned char)integral_image.channels();
  means.resize(disjoint_set.size() * channels);
  std::vector<mean_type>::iterator mean = means.begin();
  for (DisjointSetVector<Shape::Rectangle>::const_iterator node = disjoint_set.begin(); node != disjoint_set.end(); node++, mean += channels)
  {
    GetMeans<integral_type, mean_type>(integral_image, *node->value(), &*mean);
  }
}

template<typename sum_type, typename total_squared_sum_type>
inline total_squared_sum_type GetTotalSquaredSums(const std::vector<sum_type>& sums, total_squared_sum_type& total_squared_sum)
{
  total_squared_sum = 0;
  for (std::vector<sum_type>::const_iterator sum = sums.begin(); sum != sums.end(); sum++)
  {
    total_squared_sum += (total_squared_sum_type)*sum * (total_squared_sum_type)*sum;
  }
  return total_squared_sum;
}

template<typename integral_type, typename sum_type, typename total_squared_sum_type, typename gain_type>
bool FindBestSplit(const cv::Mat& integral_image, Split<total_squared_sum_type, gain_type>& split)
{
  // We assume that split.s and split.r are already computed.
  const Shape::Rectangle& r = split.r;
  const int width = RectangleWidth(r);
  const int height = RectangleHeight(r);
  const int margin = std::max(std::max(width, height) / 16, 3);

  // Check if we can split within a margin.
  if (height <= margin && width <= margin)
  {
    split.gain = 0;
    return false;
  }

  Shape::Rectangle r1, r2;
  std::vector<sum_type> sums(integral_image.channels());
  total_squared_sum_type s1, s2;
  gain_type max_gain = -std::numeric_limits<gain_type>::infinity();

  // Horizontal split.
  if (height > margin)
	{
		for (r1.x1 = r2.x1 = r.x1,
			   r1.x2 = r2.x2 = r.x2,
			   r1.y1 = r.y1,
		     r1.y2 = r.y1 + margin,
			   r2.y1 = r.y1 + margin + 1,	
			   r2.y2 = r.y2;
			   r2.y1 < r.y2 - margin;
			   r1.y2++, r2.y1++)
		{
      // Compute rectangle areas.
      const size_t area1 = RectangleArea(r1);
	  const size_t area2 = RectangleArea(r2);

      // Update sums.
      GetSums<integral_type>(integral_image, r1, sums.data());
      GetTotalSquaredSums(sums, s1);
      GetSums<integral_type>(integral_image, r2, sums.data());
      GetTotalSquaredSums(sums, s2);

      // Get splitting gain.
      const gain_type gain = s1 / (gain_type)area1 + s2 / (gain_type)area2;

      // Record maximum gain.
			if(max_gain < gain)
			{
        max_gain = gain;
        split.r1 = r1;
        split.r2 = r2;
        split.s1 = s1;
        split.s2 = s2;
      }
    }
  }

  // Vertical split.
  if (width > margin)
	{
		for (r1.y1 = r2.y1 = r.y1,
			   r1.y2 = r2.y2 = r.y2,
			   r1.x1 = r.x1,
			   r1.x2 = r.x1 + margin,
			   r2.x1 = r.x1 + margin + 1,	
			   r2.x2 = r.x2;
			   r2.x1 < r.x2 - margin;
			   r1.x2++, r2.x1++)
		{
      // Compute rectangle areas.
      const size_t area1 = RectangleArea(r1);
			const size_t area2 = RectangleArea(r2);

      // Update sums.
      GetSums<integral_type>(integral_image, r1, sums.data());
      GetTotalSquaredSums(sums, s1);
      GetSums<integral_type>(integral_image, r2, sums.data());
      GetTotalSquaredSums(sums, s2);

      // Get splitting gain.
      const gain_type gain = s1 / (gain_type)area1 + s2 / (gain_type)area2;

      // Record maximum gain.
			if(max_gain < gain)
			{
        max_gain = gain;
        split.r1 = r1;
        split.r2 = r2;
        split.s1 = s1;
        split.s2 = s2;
      }
    }
  }

  // Compute final gain.
  const size_t area = RectangleArea(r);
  split.gain = max_gain - split.s / (gain_type)area;
  if (split.gain < std::numeric_limits<gain_type>::epsilon())
  {
    split.gain = 0;
    return false;
  }

  return true;
}

template<typename integral_type, typename mean_type>
void GetEdges(const cv::Mat& integral_image, DisjointSetVector<Shape::Rectangle>& disjoint_set, std::vector<Edge >& edges)
{
  // Preallocate memory.
  const unsigned int avg_branching_factor = 50;
  const unsigned char channels = (unsigned char)integral_image.channels();
  if (edges.capacity() < disjoint_set.size() * avg_branching_factor) edges.reserve(disjoint_set.size() * avg_branching_factor);
  edges.clear();

  // Get means.
  std::vector<mean_type> means(disjoint_set.size() * channels);
  GetMeans<integral_type>(integral_image, disjoint_set, means);
  
  typedef DisjointSetVector<Shape::Rectangle>::iterator neighbor_type;
  typedef std::vector<neighbor_type > neighbors_type;
  typedef DisjointSetVector<Shape::Rectangle>::value_type::value_type value_type;
  std::vector<neighbors_type > x2_list(integral_image.cols);
  std::vector<neighbors_type > y2_list(integral_image.rows);
  
  // Organize rectangles by coordinates.
  for (neighbor_type node = disjoint_set.begin(); node != disjoint_set.end(); node++)
  {
    neighbors_type& x2_bin = x2_list[node->value()->x2 + 1];
    if (x2_bin.capacity() < avg_branching_factor) x2_bin.reserve(avg_branching_factor);
    x2_bin.push_back(node);
    neighbors_type& y2_bin = y2_list[node->value()->y2 + 1];
    if (y2_bin.capacity() < avg_branching_factor) y2_bin.reserve(avg_branching_factor);
    y2_bin.push_back(node);
  }
  
  // Get edges.
  std::vector<mean_type>::iterator rectangle_color_means = means.begin();
  for (neighbor_type node = disjoint_set.begin(); node != disjoint_set.end(); node++, rectangle_color_means += channels)
  {
    const value_type& rectangle = node->value();

    // Check left neighbors.
    neighbors_type& potential_x_neighbors = x2_list[rectangle->x1];
    for (neighbors_type::iterator potential_neighbor = potential_x_neighbors.begin(); potential_neighbor != potential_x_neighbors.end(); potential_neighbor++)
    {
      const value_type& n_rectangle = (*potential_neighbor)->value();
      if (n_rectangle->x1 <= rectangle->x2 + 1 && n_rectangle->y1 <= rectangle->y2 + 1 && n_rectangle->y2 + 1 >= rectangle->y1)
      {
        // Link nodes.
        std::vector<mean_type>::iterator n_mean = means.begin() + (size_t)(*potential_neighbor - disjoint_set.begin()) * channels;
        Edge edge(node, *potential_neighbor);

        // Compute weight.
        for (std::vector<mean_type>::iterator mean = rectangle_color_means; mean != rectangle_color_means + channels; mean++, n_mean++)
        {
          mean_type diff = *mean - *n_mean;
          edge.weight += diff * diff;
        }
        edge.weight = std::sqrt(edge.weight);
        edges.push_back(edge);
      }
    }

    // Check up neighbors.
    neighbors_type& potential_y_neighbors = y2_list[rectangle->y1];
    for (neighbors_type::iterator potential_neighbor = potential_y_neighbors.begin(); potential_neighbor != potential_y_neighbors.end(); potential_neighbor++)
    {
      const value_type& n_rectangle = (*potential_neighbor)->value();
      if (n_rectangle->x2 + 1 >= rectangle->x1 && n_rectangle->x1 <= rectangle->x2 + 1 && n_rectangle->y1 <= rectangle->y2 + 1 && n_rectangle->x2 + 1 != rectangle->x1)
      {
        // Link nodes.
        std::vector<mean_type>::iterator n_mean = means.begin() + (size_t)(*potential_neighbor - disjoint_set.begin()) * channels;
        Edge edge(node, *potential_neighbor);

        // Compute weight.
        for (std::vector<mean_type>::iterator mean = rectangle_color_means; mean != rectangle_color_means + channels; mean++, n_mean++)
        {
          mean_type diff = *mean - *n_mean;
          edge.weight += diff * diff;
        }
        edge.weight = std::sqrt(edge.weight);
        edges.push_back(edge);
      }
    }
  }
}

typedef Split<uint64_t, float> split_type;
typedef int integral_type;
typedef integral_type sum_type;
typedef float mean_type;
typedef std::vector<Shape::Rectangle>::const_iterator neighbor_type;
typedef std::vector<neighbor_type > neighbors_type;

} // Anonymous namespace.

void SplitImage(const cv::Mat& integral_image, size_t num_rectangles, std::vector<Shape::Rectangle>& rectangles)
{
  // Compute initial split.
  split_type split;
  split.r.x1 = split.r.y1 = 0;
  split.r.x2 = integral_image.cols - 2;
  split.r.y2 = integral_image.rows - 2;

  std::vector<sum_type> sums(integral_image.channels());
  GetSums<integral_type>(integral_image, split.r, sums);
  GetTotalSquaredSums(sums, split.s);
  bool available_split = FindBestSplit<integral_type, sum_type>(integral_image, split);

  if (!available_split) return;

  // Add initial split int the queue.
  std::priority_queue<split_type > priority_queue;
  priority_queue.push(split);

  while(priority_queue.size() < num_rectangles)
  {
    // Pop best split.
    split = priority_queue.top();
    available_split = split.gain > 0;
    if (!available_split) break;

    priority_queue.pop();

	// first half
    split_type new_split;
    new_split.r = split.r1;
    new_split.s = split.s1;
    FindBestSplit<integral_type, sum_type>(integral_image, new_split);
    priority_queue.push(new_split);

	// second half
    new_split.r = split.r2;
    new_split.s = split.s2;
    FindBestSplit<integral_type, sum_type>(integral_image, new_split);
    priority_queue.push(new_split);
  }

  rectangles.resize(priority_queue.size());
  std::vector<Shape::Rectangle>::iterator r = rectangles.begin();
  for (std::vector<Shape::Rectangle>::iterator r = rectangles.begin(); r != rectangles.end(); r++)
  {
    *r = priority_queue.top().r;
    priority_queue.pop();
  }
}

size_t GetNeighborRectangles(const cv::Size image_size, const std::vector<Shape::Rectangle>& rectangles, std::vector<neighbors_type >& neighbors)
{
  // Preallocate memory.
  bool reserved = true;
  const unsigned int min_capacity = 50;
  neighbors.resize(rectangles.size());
  for (std::vector<neighbors_type >::iterator neighbor_rectangles = neighbors.begin(); neighbor_rectangles != neighbors.end(); neighbor_rectangles++)
  {
    neighbor_rectangles->clear();
    if (neighbor_rectangles->capacity() < min_capacity)
    {
      neighbor_rectangles->reserve(min_capacity);
      reserved = false;
    }
  }

  std::vector<neighbors_type > x2_list(image_size.width + 1);
  std::vector<neighbors_type > y2_list(image_size.height + 1);

  // Organize rectangles by coordinates.
  for (std::vector<Shape::Rectangle>::const_iterator rectangle = rectangles.begin(); rectangle != rectangles.end(); rectangle++)
  {
    x2_list[rectangle->x2 + 1].push_back(rectangle);
    y2_list[rectangle->y2 + 1].push_back(rectangle);
  }

  // Get neighbors.
  size_t cnt = 0;
  std::vector<neighbors_type >::iterator neighbor_rectangles = neighbors.begin();
  for (std::vector<Shape::Rectangle>::const_iterator rectangle = rectangles.begin(); rectangle != rectangles.end(); rectangle++, neighbor_rectangles++)
  {
    // Check left neighbors.
    const neighbors_type& potential_x_neighbors = x2_list[rectangle->x1];
    for (neighbors_type::const_iterator potential_neighbor = potential_x_neighbors.begin(); potential_neighbor != potential_x_neighbors.end(); potential_neighbor++)
    {
      const neighbor_type& n_rectangle = *potential_neighbor;
      if (n_rectangle->x1 <= rectangle->x2 + 1 && n_rectangle->y1 <= rectangle->y2 + 1 && n_rectangle->y2 + 1 >= rectangle->y1)
      {
        neighbor_rectangles->push_back(n_rectangle);
        neighbors[n_rectangle - rectangles.begin()].push_back(rectangle);
        cnt += 2;
      }
    }

    // Check up neighbors.
    const neighbors_type& potential_y_neighbors = y2_list[rectangle->y1];
    for (neighbors_type::const_iterator potential_neighbor = potential_y_neighbors.begin(); potential_neighbor != potential_y_neighbors.end(); potential_neighbor++)
    {
      const neighbor_type& n_rectangle = *potential_neighbor;
      if (n_rectangle->x2 + 1 >= rectangle->x1 && n_rectangle->x1 <= rectangle->x2 + 1 && n_rectangle->y1 <= rectangle->y2 + 1 && n_rectangle->x2 + 1 != rectangle->x1)
      {
        neighbor_rectangles->push_back(n_rectangle);
        neighbors[n_rectangle - rectangles.begin()].push_back(rectangle);
        cnt += 2;
      }
    }
  }

  return cnt;
}

void MergeRectangles(const cv::Mat& integral_image, std::vector<Shape::Rectangle>& rectangles, const float threshold, std::vector<std::vector<std::vector<Shape::Rectangle>::iterator> >& segments)
{
  // Create disjoint set vector.
  DisjointSetVector<Shape::Rectangle> disjoint_set(rectangles);
  /*DisjointSetVector<Shape::Rectangle>::iterator node = disjoint_set.begin();
  for (std::vector<Shape::Rectangle>::iterator rectangle = rectangles.begin(); rectangle != rectangles.end(); rectangle++, node++)
  {
    node->component_size() = RectangleArea(*rectangle);
  }*/

  // Compute edges.
  std::vector<Edge > edges;
  GetEdges<integral_type, mean_type>(integral_image, disjoint_set, edges);

  // Sort edges.
  std::sort(edges.begin(), edges.end());

  // Merge nodes.
  for (std::vector<Edge >::iterator edge = edges.begin(); edge != edges.end(); edge++)
  {
    edge->Union((Edge::weight_type)threshold);
  }

  // Get segments.
  std::vector<std::vector<std::vector<Shape::Rectangle>::iterator> > sparse_segments(disjoint_set.size());
  for (DisjointSetVector<Shape::Rectangle>::iterator node = disjoint_set.begin(); node != disjoint_set.end(); node++)
  {
    DisjointSetVector<Shape::Rectangle>::pointer root = node->FindRoot();
    const size_t id = root - &*disjoint_set.begin();
    sparse_segments[id].push_back(node->value());
  }

  if (segments.capacity() < disjoint_set.size()) segments.reserve(disjoint_set.size());
  segments.clear();

  for (std::vector<std::vector<std::vector<Shape::Rectangle>::iterator> >::iterator sparse_segment = sparse_segments.begin(); sparse_segment != sparse_segments.end(); sparse_segment++)
  {
    if (!sparse_segment->size()) continue;
    segments.push_back(*sparse_segment);
  }

  //std::cout << edges.size() << " | " << segments.size() << std::endl;
}

void GetWindows(const cv::Mat& integral_image, std::vector<Shape::Rectangle>& rectangles, const float threshold, const float dist_threshold, std::vector<Shape::Rectangle>& windows, bool append)
{
  // Init windows.
  if (windows.capacity() < rectangles.size()) windows.reserve(rectangles.size());
  if (!append) windows.clear();

  // Create disjoint set vector.
  DisjointSetVector<Shape::Rectangle> disjoint_set(rectangles);

  // Compute edges.
  std::vector<Edge > edges;
  GetEdges<integral_type, mean_type>(integral_image, disjoint_set, edges);

  // Sort edges.
  std::sort(edges.begin(), edges.end());

  // Merge nodes.
  for (std::vector<Edge >::iterator edge = edges.begin(); edge != edges.end(); edge++)
  {
    edge->Union((Edge::weight_type)threshold);
  }

  // Get segments.
  std::vector<std::vector<std::vector<Shape::Rectangle>::iterator> > sparse_segments(disjoint_set.size());
  for (DisjointSetVector<Shape::Rectangle>::iterator node = disjoint_set.begin(); node != disjoint_set.end(); node++)
  {
    DisjointSetVector<Shape::Rectangle>::pointer root = node->FindRoot();
    const size_t id = root - &*disjoint_set.begin();
    sparse_segments[id].push_back(node->value());
  }

  size_t min_area = (integral_image.rows - 1) * (integral_image.cols - 1) / 400;
  for (std::vector<std::vector<std::vector<Shape::Rectangle>::iterator> >::iterator sparse_segment = sparse_segments.begin(); sparse_segment != sparse_segments.end(); sparse_segment++)
  {
    if (!sparse_segment->size() || (sparse_segment->size() == 1 && RectangleArea(*(*sparse_segment)[0]) < min_area)) continue;
    Shape::Rectangle window = {integral_image.cols, integral_image.rows, 0, 0};
    for (std::vector<std::vector<Shape::Rectangle>::iterator>::iterator rectangle = sparse_segment->begin(); rectangle != sparse_segment->end(); rectangle++)
    {
      const Shape::Rectangle& r = **rectangle;
      window.x1 = std::min(window.x1, r.x1);
      window.y1 = std::min(window.y1, r.y1);
      window.x2 = std::max(window.x2, r.x2);
      window.y2 = std::max(window.y2, r.y2);
    }
    windows.push_back(window);
  }

  // Also include nearby windows.
  if (dist_threshold <= 0) return;
  const int end = windows.size();
  for (int i1 = 0; i1 < end; i1++)
  {
    for (int i2 = i1 + 1; i2 < end; i2++)
    {
      const Shape::Rectangle& window1 = windows[i1];
      const Shape::Rectangle& window2 = windows[i2];

      // No need to merge inside windows.
      if (IsRectangleInside(window1, window2) || IsRectangleInside(window2, window1) || NormalizedRectangleDist(window2, window1) > 0.5) continue;
      Shape::Rectangle window = {std::min(window1.x1, window2.x1), std::min(window1.y1, window2.y1), std::max(window1.x2, window2.x2), std::max(window1.y2, window2.y2)};
      windows.push_back(window);
    }
  }
}

void GetWindows(const cv::Mat& integral_image, std::vector<Shape::Rectangle>& rectangles, std::vector<float>& thresholds, const float dist_threshold, std::vector<Shape::Rectangle>& windows)
{
  // Init windows.
  if (windows.capacity() < rectangles.size()) windows.reserve(rectangles.size());
  windows.clear();

  // Create disjoint set vector.
  DisjointSetVector<Shape::Rectangle> disjoint_set(rectangles);

  // Compute edges.
  std::vector<Edge > edges;
  GetEdges<integral_type, mean_type>(integral_image, disjoint_set, edges);

  // Sort edges.
  std::sort(edges.begin(), edges.end());

  // Sort threshold.
  std::sort(thresholds.begin(), thresholds.end());

  // Merge nodes.
  std::vector<std::vector<std::vector<Shape::Rectangle>::iterator> > sparse_segments(disjoint_set.size());
  std::vector<size_t> sparse_segment_sizes(disjoint_set.size(), 0);
  std::vector<Edge > unmerged_edges;
  unmerged_edges.reserve(edges.size());
  const size_t min_area = (integral_image.rows - 1) * (integral_image.cols - 1) / 400;

  std::vector<Shape::Rectangle> small_windows;
  if (dist_threshold > 0) small_windows.reserve(rectangles.size());
  for (int k = 0; k < thresholds.size(); k++)
  {
    const float threshold = thresholds[k];
    std::vector<Edge >& merging = k & 1 ? unmerged_edges : edges;
    std::vector<Edge >& unmerged = k & 1 ? edges : unmerged_edges;
    unmerged.clear();
    for (std::vector<Edge >::iterator edge = merging.begin(); edge != merging.end(); edge++)
    {
      if (!edge->Union((Edge::weight_type)threshold))
      {
        unmerged.push_back(*edge);
      }
    }

    // Get segments.
    for (DisjointSetVector<Shape::Rectangle>::iterator node = disjoint_set.begin(); node != disjoint_set.end(); node++)
    {
      DisjointSetVector<Shape::Rectangle>::pointer root = node->FindRoot();
      const size_t id = root - &*disjoint_set.begin();
      sparse_segments[id].push_back(node->value());
    }

    // Get windows.
    std::vector<size_t>::iterator sparse_segment_size = sparse_segment_sizes.begin();
    for (std::vector<std::vector<std::vector<Shape::Rectangle>::iterator> >::iterator sparse_segment = sparse_segments.begin(); sparse_segment != sparse_segments.end(); sparse_segment++, sparse_segment_size++)
    {
      if (sparse_segment->size() && sparse_segment->size() != *sparse_segment_size)
      {
        if (sparse_segment->size() != 1 || RectangleArea(*(*sparse_segment)[0]) > min_area)
        {
          Shape::Rectangle window = {integral_image.cols, integral_image.rows, 0, 0};
          for (std::vector<std::vector<Shape::Rectangle>::iterator>::iterator rectangle = sparse_segment->begin(); rectangle != sparse_segment->end(); rectangle++)
          {
            const Shape::Rectangle& r = **rectangle;
            window.x1 = std::min(window.x1, r.x1);
            window.y1 = std::min(window.y1, r.y1);
            window.x2 = std::max(window.x2, r.x2);
            window.y2 = std::max(window.y2, r.y2);
          }
          windows.push_back(window);
        }
        else if (!k && dist_threshold > 0)
        {
          Shape::Rectangle window = {integral_image.cols, integral_image.rows, 0, 0};
          for (std::vector<std::vector<Shape::Rectangle>::iterator>::iterator rectangle = sparse_segment->begin(); rectangle != sparse_segment->end(); rectangle++)
          {
            const Shape::Rectangle& r = **rectangle;
            window.x1 = std::min(window.x1, r.x1);
            window.y1 = std::min(window.y1, r.y1);
            window.x2 = std::max(window.x2, r.x2);
            window.y2 = std::max(window.y2, r.y2);
          }
          small_windows.push_back(window);
        }
      }
      *sparse_segment_size = sparse_segment->size();
      sparse_segment->clear();
    }
  }

  // Also include nearby windows.
  if (dist_threshold <= 0) return;
  const int end = windows.size();
  for (int i1 = 0; i1 < end; i1++)
  {
    for (int i2 = i1 + 1; i2 < end; i2++)
    {
      const Shape::Rectangle& window1 = windows[i1];
      const Shape::Rectangle& window2 = windows[i2];

      // No need to merge inside windows.
      if (IsRectangleInside(window1, window2) || IsRectangleInside(window2, window1) || NormalizedRectangleDist(window2, window1) > dist_threshold) continue;
      Shape::Rectangle window = {std::min(window1.x1, window2.x1), std::min(window1.y1, window2.y1), std::max(window1.x2, window2.x2), std::max(window1.y2, window2.y2)};
      windows.push_back(window);
    }

    for (std::vector<Shape::Rectangle>::iterator small_window = small_windows.begin(); small_window != small_windows.end(); small_window++)
    {
      const Shape::Rectangle& window1 = windows[i1];
      const Shape::Rectangle& window2 = *small_window;

      // No need to merge inside windows.
      if (IsRectangleInside(window1, window2) || IsRectangleInside(window2, window1) || NormalizedRectangleDist(window2, window1) > dist_threshold) continue;
      Shape::Rectangle window = {std::min(window1.x1, window2.x1), std::min(window1.y1, window2.y1), std::max(window1.x2, window2.x2), std::max(window1.y2, window2.y2)};
      windows.push_back(window);
    }
  }
}