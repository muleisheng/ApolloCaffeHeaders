#ifndef INCLUDE_DETECT2D_NEWRECTMAP_HPP_
#define INCLUDE_DETECT2D_NEWRECTMAP_HPP_

#include <vector>
#include <map>
#include <string>
#include <set>
#include <opencv2/opencv.hpp>
#include "caffe/blob.hpp"

using namespace std;
namespace caffe {

class NewRectPoint {
public:
    NewRectPoint();
    ~NewRectPoint();
    NewRectPoint(int y_, int x_);
    bool operator <(const NewRectPoint& other) const;
    NewRectPoint add(int dy, int dx) const;
    friend ostream& operator <<(ostream& stream, NewRectPoint& point);
    int _x;
    int _y;
};

class NewRect {
public:
    NewRect();
    ~NewRect();
    friend ostream& operator <<(ostream& stream, NewRect& rect);
    NewRect(NewRectPoint left_top_, int height_ = 0, int width_ = 0);
    bool contain(const NewRectPoint& point) const;
    int overlap(const NewRect& other);
    NewRectPoint _left_top;
    int _height;
    int _width;
};

class NewRectMap {
public:
    NewRectMap();
    ~NewRectMap();
    bool occupied(const NewRectPoint& point);
    int mapheight();
    int mapwidth();
    int getarea();
    const vector<NewRect>& getplacedrects();
    bool placerect(const NewRect& rect);
    void clear();

private:
    // Return the total area if place rectangle at this point. If cannot place the rectange here, return INT_MAX; //
    int trytoplacerectat(const NewRect& rect, NewRectPoint point);

    // return the area if the added_rect is placed //
    int getarea(const NewRect& added_rect);

    NewRectPoint greedyfindbestpointtoplace(const NewRect& rect);

    // place a corner point in map, which updates the horizontal_line, vertical_line, candidateLeftTopPoint; //
    void placecornerpoint(const NewRectPoint point);
    void pruneinvalidcandidatepoint();

    vector<NewRect> _placed_rects;
    vector<int> _placed_rect_ids;

    set<int, greater<int> > _horizontal_line; // line horizontal, contain y axis， largest first
    set<int, greater<int> > _vertical_line; // line vertical, contain x axis， largest first
    map<NewRectPoint, int> _candidatellefttoppoint;      // point, id
};

template<typename Dtype>
class NewRoiRect {
public:
    NewRoiRect(Dtype scale = 0, Dtype start_y = 0, Dtype start_x = 0,
            Dtype height = 0, Dtype width = 0);      //use
    ~NewRoiRect() {
    }
    ;
    inline Dtype getarea() const {
        return _height * _width;
    }

    inline Dtype getscaledx(Dtype dx) const {
        return (_start_x + dx) * _scale;
    }
    inline Dtype getscaledy(Dtype dy) const {
        return (_start_y + dy) * _scale;
    }

    inline Dtype getorix(Dtype scaled_dx) const {
        return _start_x + scaled_dx / _scale;
    }
    inline Dtype getoriy(Dtype scaled_dy) const {
        return _start_y + scaled_dy / _scale;
    }

    static bool greaterscale(const NewRoiRect& a, const NewRoiRect& b) {
        return a._scale > b._scale;
    }

    inline Dtype getscaledarea() const {
        return _scale * _height * _scale * _width;
    }
    static bool greaterscaledarea(const NewRoiRect& a, const NewRoiRect&b) {
        return a.getscaledarea() > b.getscaledarea();
    }      //use

    inline Dtype getscaledheight() const {
        return _scale * _height;
    }
    inline Dtype getscaledwidth() const {
        return _scale * _width;
    }
    static bool greatermaxscalededge(const NewRoiRect& a, const NewRoiRect&b) {
        return MAX(a.getscaledheight(), a.getscaledwidth())
                > MAX(b.getscaledheight(), b.getscaledwidth());
    }

    Dtype _scale;
    Dtype _start_y, _start_x;
    Dtype _height, _width;
};

}
#endif /* INCLUDE_CAFFE_UTIL_RECTMAP_HPP_ */
