// Confidential, Copyright 2013 A9.com, Inc.

#include "Rectangle.h"

#include <algorithm>

Shape::Rectangle::value_type RectangleIntersection(const Shape::Rectangle& r1, const Shape::Rectangle& r2,
                                                   Shape::Rectangle* inter)
{
  inter->x1 = std::max(r1.x1, r2.x1);
	inter->y1 = std::max(r1.y1, r2.y1);
	inter->x2 = std::min(r1.x2, r2.x2);
	inter->y2 = std::min(r1.y2, r2.y2);
    
  bool intersect = true;
	if (inter->x1 > inter->x2)
  {
    inter->x2 = inter->x1;
    intersect = false;
  }
	if (inter->y1 > inter->y2)
  {
    inter->y2 = inter->y1;
    intersect = false;
  }
    
	return intersect ? RectangleArea(*inter) : 0;
}

Shape::Rectangle::value_type RectangleIntersectionArea(const Shape::Rectangle& r1, const Shape::Rectangle& r2)
{
  Shape::Rectangle inter;
	return RectangleIntersection(r1, r2, &inter);
}

float RectangleIntersectionOverUnionArea(const Shape::Rectangle& r1, const Shape::Rectangle& r2)
{
  Shape::Rectangle::value_type i = RectangleIntersectionArea(r1, r2);
	return static_cast<float>(i) / static_cast<float>(RectangleArea(r1) + RectangleArea(r2) - i);
}

float RectangleShapeDist(const Shape::Rectangle& r1, const Shape::Rectangle& r2)
{
  Shape::Rectangle::value_type w1 = RectangleWidth(r1);
	Shape::Rectangle::value_type h1 = RectangleHeight(r1);
	Shape::Rectangle::value_type v1 = w1 * h1;
    
	Shape::Rectangle::value_type w2 = RectangleWidth(r2);
	Shape::Rectangle::value_type h2 = RectangleHeight(r2);
	Shape::Rectangle::value_type v2 = w2 * h2;
    
	Shape::Rectangle::value_type w = std::min(w1, w2);
	Shape::Rectangle::value_type h = std::min(h1, h2);
	Shape::Rectangle::value_type v = w * h;
    
	return static_cast<float>(std::max(v1, v2)) / static_cast<float>(v);
}

float RectangleDist(const Shape::Rectangle& r1, const Shape::Rectangle& r2)
{
  int dx = 0, dy = 0;
  if (r1.x2 < r2.x1) dx += r2.x1 - r1.x2;
  if (r2.x2 < r1.x1) dx += r1.x1 - r2.x2;
  if (r1.y2 < r2.y1) dx += r2.y1 - r1.y2;
  if (r2.y2 < r1.y1) dx += r1.y1 - r2.y2;

  if (dx || dy) return std::sqrt((float)dx * dx + dy * dy);
  else return 1 - RectangleIntersectionOverUnionArea(r1, r2);
}

float NormalizedRectangleDist(const Shape::Rectangle& r1, const Shape::Rectangle& r2)
{
  int dx = 0, dy = 0;
  if (r1.x2 < r2.x1) dx += r2.x1 - r1.x2;
  if (r2.x2 < r1.x1) dx += r1.x1 - r2.x2;
  if (r1.y2 < r2.y1) dx += r2.y1 - r1.y2;
  if (r2.y2 < r1.y1) dx += r1.y1 - r2.y2;

  if (dx || dy)
  {
    float n_dx = dx / (float)(std::max(r1.x2, r2.x2) - std::min(r1.x1, r2.x1));
    float n_dy = dy / (float)(std::max(r1.y2, r2.y2) - std::min(r1.y1, r2.y1));
    return 1 + std::sqrt(n_dx * n_dx + n_dy * n_dy);
  }
  else return 1 - RectangleIntersectionOverUnionArea(r1, r2);
}

std::ostream& operator<<(std::ostream& os, const Shape::Rectangle& rectangle)
{
  return os << "(" << rectangle.x1 << ", " << rectangle.y1 << ") "
  << "(" << rectangle.x2 << ", " << rectangle.y2 << ")";
}

