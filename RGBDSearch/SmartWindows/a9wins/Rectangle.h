// Confidential, Copyright 2013 A9.com, Inc.

#ifndef LOGO_RECOGNITION_RECTANGLE_H_
#define LOGO_RECOGNITION_RECTANGLE_H_

#include <iostream>

// Rectangle structure.
#define RECTANGLE_STRUCT                                                 \
union                                                                    \
{                                                                        \
  struct{Shape::Rectangle::value_type x1, y1, x2, y2;};                  \
  struct{Shape::Rectangle::value_type top_left[2], bottom_right[2];};    \
  Shape::Rectangle::value_type corners[4];                               \
}

namespace Shape
{

struct Rectangle
{
  #pragma warning (push, 3)  // Temporarily setting warning level 3. It complains about nameless structure.
  typedef int value_type;
  RECTANGLE_STRUCT;
  #pragma warning (pop)
};

} // namespace Shape.

// Find if two rectangles are neighbors.
#define IsRectangleNeighbor(r1, r2)      \
(IsRectangleNeighborOrIntersect(r1, r2) &\
((r2).x1     == (r1).x2 + 1 ||           \
 (r2).x2 + 1 == (r1).x1     ||           \
 (r2).y1     == (r1).y2 + 1 ||           \
 (r2).y2 + 1 == (r1).y1) )

// Find if two rectangles intersect.
#define IsRectangleIntersect(r1, r2)  \
((r2).x1 <= (r1).x2 &&                \
 (r2).x2 >= (r1).x1 &&                \
 (r2).y1 <= (r1).y2 &&                \
 (r2).y2 >= (r1).y1   )

// 	Find if two rectangles are neighbors or intersect.
#define IsRectangleNeighborOrIntersect(r1, r2)  \
((r2).x1     <= (r1).x2 + 1 &&                  \
 (r2).x2 + 1 >= (r1).x1     &&                  \
 (r2).y1     <= (r1).y2 + 1 &&                  \
 (r2).y2 + 1 >= (r1).y1  )

// Find if one rectangle contains the other one.
#define IsRectangleInside(r1, r2)  \
((r2).x1 <= (r1).x1 &&             \
 (r2).x2 >= (r1).x2 &&             \
 (r2).y1 <= (r1).y1 &&             \
 (r2).y2 >= (r1).y2   )

// Find rectangle width.
#define RectangleWidth(r) ((r).x2 - (r).x1 + 1)

// Find rectangle height.
#define RectangleHeight(r) ((r).y2 - (r).y1 + 1)

// Find rectangle area.
#define RectangleArea(r) ( RectangleWidth(r) * RectangleHeight(r) )

// Find area of the union of two rectangle.
#define RectangleUnionArea(r1, r2) \
(RectangleArea(r1) + RectangleArea(r2) - RectangleIntersectionArea(r1, r2))

// Find rectangle intersection.
Shape::Rectangle::value_type RectangleIntersection(const Shape::Rectangle& r1, const Shape::Rectangle& r2,
                                                   Shape::Rectangle* inter);

// Find area intersection of two rectangle.
Shape::Rectangle::value_type RectangleIntersectionArea(const Shape::Rectangle& r1, const Shape::Rectangle& r2);

// Find area of the union over area of the intersection of two rectangle.
float RectangleIntersectionOverUnionArea(const Shape::Rectangle& r1, const Shape::Rectangle& r2);

// Compute the distance between two rectangle shapes.
float RectangleShapeDist(const Shape::Rectangle& r1, const Shape::Rectangle& r2);

// Compute distance between two rectangles
float RectangleDist(const Shape::Rectangle& r1, const Shape::Rectangle& r2);
float NormalizedRectangleDist(const Shape::Rectangle& r1, const Shape::Rectangle& r2);

// Convert rectangle to cv::Rect and vice versa.
template <class C>
inline C& ConvertRectangleToCvRect(const Shape::Rectangle& src, C* dst)
{
  dst->x = src.x1;
	dst->y = src.y1;
	dst->width = RectangleWidth(src);
	dst->height = RectangleHeight(src);
  return *dst;
}

#define ConvertRectangleToCvRect(r) cv::Rect(r.x1, r.y1, RectangleWidth(r), RectangleHeight(r))

template<typename T>
inline Shape::Rectangle& InitRectangleWithCvRect(const T& cv_r, Shape::Rectangle& r)
{
  r.x1 = cv_r.x;
  r.y1 = cv_r.y;
  r.x2 = cv_r.x + cv_r.width - 1;
  r.y2 = cv_r.y + cv_r.height - 1;
  return r;
}

template<typename T>
inline Shape::Rectangle& GetRectangleFromCvRect(const T& cv_r)
{
  Shape::Rectangle r;
  r.x1 = cv_r.x;
  r.y1 = cv_r.y;
  r.x2 = cv_r.x + cv_r.width - 1;
  r.y2 = cv_r.y + cv_r.height - 1;
  return r;
}

// Displaying rectangle properties.
std::ostream& operator<<(std::ostream& os, const Shape::Rectangle& rectangle);

#endif  // LOGO_RECOGNITION_RECTANGLE_H_
