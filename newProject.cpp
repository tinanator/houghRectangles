

#include "opencv2/imgcodecs.hpp"
#include "opencv2/imgproc.hpp"
#include "opencv2/highgui.hpp"
#include <opencv2/core/utility.hpp>
#include <vector>
#include <algorithm>
#include <set>
#include <queue>
#include <iostream>

using namespace cv;
using namespace std;

/****************************************************************************************\
*                                     Rect Detection                                     *
\****************************************************************************************/
struct Cell
{
    int row;
    int col;
    Cell(
        int row = -1,
        int col = -1
    )
        : row(row)
        , col(col)
    {};
};

struct AccumPoint
{
    int value;
    int angle;
    Cell cell;
    int height;
    int width;
    int ratio;
    AccumPoint(
        int value = 0,
        int angle = 0,
        Cell cell = Cell(-1, -1),
        int height = 0,
        int width = 0,
        int ratio = 0
    )
        : value(value)
        , angle(angle)
        , cell(cell)
        , height(height)
        , width(width)
        , ratio(ratio)
    {};
    bool operator < (const AccumPoint& r) const { return value > r.value; }
};

struct RectSize
{
    int height;
    int width;
    AccumPoint accum;
    RectSize(
        int height,
        int width,
        AccumPoint accum
    )
        : height(height)
        , width(width)
        , accum(accum)
    {};
};

int get_distance(Cell& c1, Cell& c2)
{
    return (c2.row - c1.row) * (c2.row - c1.row) + (c2.col - c1.col) * (c2.col - c1.col);
}

Cell get_real_cell(Cell& c, int hough_scale)
{
    return Cell(c.row * hough_scale + hough_scale / 2, c.col * hough_scale + hough_scale / 2);
}

struct Accum
{
    int counter;
    int angle_scaled;
    cv::Mat accum;
    std::vector<AccumPoint> local_max;
    Accum(
        int counter = 0
    )
        : counter(counter)
    {};
    bool operator < (const Accum& r) const { return counter > r.counter; }
};

struct AccumRatio
{
    std::vector<Accum> accums;
    int aspectRatio;
    int counter;
    AccumRatio(
        int aspectRatio = 0,
        int counter = 0
    )
        : aspectRatio(aspectRatio)
        , counter(counter)
    {};
    bool operator < (const AccumRatio& r) const { return counter > r.counter; }
};

class HoughRectsAccumInvoker : public ParallelLoopBody
{
private:
    const cv::Mat& image;
    const cv::Mat& edges;
    const cv::Mat& angles;
    AccumPoint& globalMaxAccum;
    std::vector<AccumRatio>& accums;
    // int width;
    // int height;
    int minAspectRatio;
    int maxAspectRatio;
    int accumScale;
    int angleStep;
    int sideSize;
    // int aspectRatio;


public:
    HoughRectsAccumInvoker(
        const cv::Mat& image,
        const cv::Mat& edges,
        const cv::Mat& angles,
        AccumPoint& globalMaxAccum,
        std::vector<AccumRatio>& accums,
        int minAspectRatio,
        int maxAspectRatio,
        // int width,
        // int height,
        int accumScale,
        int angleStep,
        int sideSize
        // int aspectRatio
    )
        : image(image)
        , edges(edges)
        , angles(angles)
        , globalMaxAccum(globalMaxAccum)
        , accums(accums)
        // , width(width)
        // , height(height)
        , minAspectRatio(minAspectRatio)
        , maxAspectRatio(maxAspectRatio)
        , accumScale(accumScale)
        , angleStep(angleStep)
        , sideSize(sideSize)
        // , aspectRatio(aspectRatio)
    {}

    ~HoughRectsAccumInvoker() { }

    std::vector <Cell> rotate_rect(Cell rect_coords[4], Cell centr, int angle) const
    {
        std::vector<Cell> res(4);
        for (int i = 0; i < 4; i++) {
            rect_coords[i].row -= centr.row;
            rect_coords[i].col -= centr.col;
        }

        for (size_t i = 0; i < res.size(); i++) {
            double theta = (angle * CV_PI) / 180.0;
            res[i].row = rect_coords[i].col * sin(theta) + rect_coords[i].row * cos(theta);
            res[i].col = rect_coords[i].col * cos(theta) - rect_coords[i].row * sin(theta);
        }

        for (size_t i = 0; i < res.size(); i++) {
            res[i].row += centr.row;
            res[i].col += centr.col;
        }

        return res;
    }

    void run_along_line(
        const cv::Mat& image,
        std::vector <Accum>& accum,
        Cell start,
        Cell finish,
        int scale,
        int angleStep,
        int angle,
        AccumRatio& accumRatio) const
    {
        if (angle < 0) angle += 180; else if (angle > 180) angle -= 180;

        int angleScaled = angle / angleStep;
        double _norm = sqrt((finish.row - start.row) * (finish.row - start.row) +
            (finish.col - start.col) * (finish.col - start.col));
        for (double i = 0; i <= 1; i += scale / (double)(_norm)) {
            Cell p = Cell(floor(start.row + (finish.row - start.row) * i),
                floor(start.col + (finish.col - start.col) * i));
            if (p.row < image.rows && p.row > 0 && p.col < image.cols && p.col > 0) {
                accum[angleScaled].accum.at<int>(p.row / scale, p.col / scale)++;
                accum[angleScaled].counter++;
                accumRatio.counter++;
                if (accum[angleScaled].accum.at<int>(p.row / scale, p.col / scale) > globalMaxAccum.value) {
                    globalMaxAccum.value = accum[angleScaled].accum.at<int>(p.row / scale, p.col / scale);
                    globalMaxAccum.cell.row = p.row / scale;
                    globalMaxAccum.cell.col = p.col / scale;
                    globalMaxAccum.angle = angle;//?
                    globalMaxAccum.ratio = accumRatio.aspectRatio;
                }
            }
        }
    }

    void run_rectangle(
        const cv::Mat& image,
        std::vector<Accum>& accum,
        int accumScale,
        int angleStep,
        int angle,
        int rad,
        double k,
        int row,
        int col,
        AccumRatio& accumRatio) const
    {
        for (int r = rad - 2; r <= rad + 2; r++) {
            int bound = 15;

            int start = angle - bound;
            int finish = angle + bound;
            for (int angle = start; angle <= finish; angle += angleStep) {
                int cur_height = r;
                int cur_width = k * r;

                Cell ptl = Cell(row - cur_height, col - cur_width);
                Cell ptr = Cell(row - cur_height, col + cur_width);
                Cell pbr = Cell(row + cur_height, col + cur_width);
                Cell pbl = Cell(row + cur_height, col - cur_width);


                Cell rect_coords[4] = { ptl, ptr, pbr, pbl };
                std::vector <Cell> rotate_coords = rotate_rect(rect_coords, Cell(row, col), angle);

                run_along_line(image, accum, rotate_coords[0], rotate_coords[1], accumScale, angleStep, angle, accumRatio);
                run_along_line(image, accum, rotate_coords[1], rotate_coords[2], accumScale, angleStep, angle, accumRatio);
                run_along_line(image, accum, rotate_coords[2], rotate_coords[3], accumScale, angleStep, angle, accumRatio);
                run_along_line(image, accum, rotate_coords[3], rotate_coords[0], accumScale, angleStep, angle, accumRatio);
            }
        }
    }

    void operator()(const cv::Range& range) const CV_OVERRIDE
    {
        int start = range.start;
        int end = range.end;
        // double k = ((double)width / (double)height);
        for (int row = start; row < end; row++) {
            for (int col = 0; col < edges.cols; col++) {
                if (edges.at<uchar>(row, col) == 0) {
                    continue;
                }
                for (int ratio = minAspectRatio, i = 0; ratio <= maxAspectRatio && i < accums.size(); ratio++, i++) {
                    int height = sideSize / 2;
                    int width = sideSize * ratio / 2;
                    double k = ((double)width / (double)height);
                    run_rectangle(image, accums[i].accums, accumScale, angleStep,
                        angles.at<int>(row, col), height, k, row, col, accums[i]);
                    run_rectangle(image, accums[i].accums, accumScale, angleStep,
                        angles.at<int>(row, col) - 90, height, k, row, col, accums[i]);
                }
            }
        }
    }

};



class HoughRectRecognizer
{
public:

    int sideSize;
    int minAspectRatio;
    int maxAspectRatio;
    int angleStep;
    int accumScale;

    HoughRectRecognizer(
        int sideSize,
        int minAspectRatio,
        int maxAspectRatio,
        int angleStep,
        int accumScale
    )
        : sideSize(sideSize)
        , minAspectRatio(minAspectRatio)
        , maxAspectRatio(maxAspectRatio)
        , angleStep(angleStep)
        , accumScale(accumScale)
    {};


    std::vector <Cell> rotate_rect(Cell rect_coords[4], Cell centr, int angle) const
    {
        std::vector<Cell> res(4);
        for (int i = 0; i < 4; i++) {
            rect_coords[i].row -= centr.row;
            rect_coords[i].col -= centr.col;
        }

        for (size_t i = 0; i < res.size(); i++) {
            double theta = (angle * CV_PI) / 180.0;
            res[i].row = rect_coords[i].col * sin(theta) + rect_coords[i].row * cos(theta);
            res[i].col = rect_coords[i].col * cos(theta) - rect_coords[i].row * sin(theta);
        }

        for (size_t i = 0; i < res.size(); i++) {
            res[i].row += centr.row;
            res[i].col += centr.col;
        }

        return res;
    }

    cv::Mat normalize_mat(const cv::Mat& mat, int _max)
    {
        cv::Mat res = cv::Mat::zeros(mat.rows, mat.cols, CV_8UC1);
        if (_max <= 0) {
            return res;
        }
        for (int i = 0; i < mat.rows; i++) {
            for (int j = 0; j < mat.cols; j++) {
                res.at<uchar>(i, j) = (255 * mat.at<int>(i, j)) / (int)_max;
            }
        }
        return res;
    }

    cv::Mat int_to_char(const cv::Mat& mat)
    {
        double _min = 0, _max = 0;
        cv::minMaxLoc(mat, &_min, &_max);
        return normalize_mat(mat, _max);
    }

    cv::Mat int_to_char_global_max(const cv::Mat& mat, int _max)
    {
        cv::Mat res = cv::Mat::zeros(mat.rows, mat.cols, CV_8UC1);
        return normalize_mat(mat, _max);
    }

    cv::Mat find_edges(const cv::Mat& mat, cv::Mat& angles, bool& empty)
    {
        cv::Mat delta_i = cv::Mat::zeros(mat.rows - 1, mat.cols - 1, CV_32SC1);
        std::vector<AccumPoint> filtering;

        angles = cv::Mat::zeros(delta_i.rows, delta_i.cols, CV_32SC1);

        for (int i = 0; i < mat.rows - 1; i++) {
            for (int j = 0; j < mat.cols - 1; j++) {

                int dx = (int)mat.at<uchar>(i, j) - (int)mat.at<uchar>(i, j + 1);
                int dy = (int)mat.at<uchar>(i, j) - (int)mat.at<uchar>(i + 1, j);
                int laplas = dx * dx + dy * dy;
                double alpha = atan2(dy, dx);

                if (alpha < 0) {
                    alpha += CV_PI;
                }

                int alpha_grad = (alpha * 180.0) / CV_PI;
                angles.at<int>(i, j) = alpha_grad;
                delta_i.at<int>(i, j) = laplas;
                filtering.push_back(AccumPoint(laplas, 0, Cell(i, j)));
                //  std::cout << i << ' ' << j << std::endl;
            }
        }

        sort(filtering.begin(), filtering.end());

        std::vector<AccumPoint> filtered(filtering.begin(), filtering.begin() + filtering.size() * 5 / 10);


        if (filtered.size() == 0) {
            empty = true;
        }

        cv::Mat edges = cv::Mat::zeros(delta_i.rows, delta_i.cols, CV_32SC1);

        for (size_t i = 0; i < filtered.size(); i++) {
            edges.at<int>(filtered[i].cell.row, filtered[i].cell.col) = filtered[i].value;
        }

        cv::Mat edges_char = int_to_char(edges);
        imshow("gg", edges_char);
        return edges_char;
    }

    std::vector <Cell> rotate_rect(Cell rect_coords[4], Cell centr, int angle)
    {
        std::vector<Cell> res(4);
        for (int i = 0; i < 4; i++) {
            rect_coords[i].row -= centr.row;
            rect_coords[i].col -= centr.col;
        }

        for (size_t i = 0; i < res.size(); i++) {
            double theta = (angle * CV_PI) / 180.0;
            res[i].row = rect_coords[i].col * sin(theta) + rect_coords[i].row * cos(theta);
            res[i].col = rect_coords[i].col * cos(theta) - rect_coords[i].row * sin(theta);
        }

        for (size_t i = 0; i < res.size(); i++) {
            res[i].row += centr.row;
            res[i].col += centr.col;
        }

        return res;
    }

    void find_local_max(const cv::Mat& image, const std::vector<AccumRatio>& accums, int windowSize, std::vector<AccumPoint>& localMax, AccumPoint& globalMaxAccum)
    {

        localMax.push_back(globalMaxAccum);
    
    }



    void houghRectParallel(
        const cv::Mat& image,
        const cv::Mat& edges,
        const cv::Mat& angles,
        AccumPoint& globalMaxAccum,
        std::vector<AccumPoint>& rects)
    {
        int max_angle = (180 + 1) / angleStep;
        int difRatio = (maxAspectRatio - minAspectRatio);
        std::vector<AccumRatio> accums(difRatio + 1);

        for (int i = 0, aspectRatio = minAspectRatio; i < accums.size() && aspectRatio <= maxAspectRatio; i++, aspectRatio++) {
            accums[i].accums.resize(max_angle + 1);
            accums[i].aspectRatio = aspectRatio;
            for (int j = 0, angle = 0; j < accums[i].accums.size() && angle <= 180; j++, angle += angleStep) {
                accums[i].accums[j].accum = cv::Mat::zeros(image.rows / accumScale,
                    image.cols / accumScale, CV_32SC1);
                accums[i].accums[j].angle_scaled = angle;
            }
        }
        int numThreads = std::max(1, getNumThreads());
        cv::parallel_for_(Range(0, edges.rows),
            HoughRectsAccumInvoker(image, edges, angles, globalMaxAccum,
                accums, minAspectRatio, maxAspectRatio, accumScale,
                angleStep, sideSize),
            numThreads);


        std::sort(accums.begin(), accums.end());
        for (int i = 0; i < accums.size(); i++) {
            std::sort(accums[i].accums.begin(), accums[i].accums.end());
        }

        find_local_max(image, accums, sideSize / accumScale, rects, globalMaxAccum);
    }

    void recognize(cv::Mat& src, std::vector<cv::Vec6f>& _rects, cv::Mat& edges, cv::Mat& angles)
    {
        AccumPoint globalMaxAccum;
        std::vector<AccumPoint> rects;
        houghRectParallel(src, edges, angles, globalMaxAccum, rects);

        for (int i = 0; i < rects.size(); i++) {
            cv::Vec6f rRect(
                rects[i].cell.col * accumScale + accumScale / 2,
                rects[i].cell.row * accumScale + accumScale / 2,
                sideSize * rects[i].ratio, sideSize, rects[0].angle, rects[0].value);
            _rects.push_back(rRect);
        }
    }
};

void HoughRects(cv::InputArray image, cv::OutputArray rects, int sideSize, int minAspectRatio,
    int maxAspectRatio, int accumScale, int angleStep)
{

    HoughRectRecognizer hr(sideSize, minAspectRatio, minAspectRatio, accumScale, angleStep);
    std::vector<cv::Vec6f> _rects;

    cv::Mat src;
    cv::Mat angles;
    image.copyTo(src);

    bool empty = false;

    cv::Mat edges = hr.find_edges(src, angles, empty);

    if (empty) {
        return;
    }
    //imshow("angles", angles);
    hr.recognize(src, _rects, edges, angles);

    int rows = (int)_rects.size();
    cv::Mat resRects(rows, 6, CV_32FC1);
    for (int i = 0; i < _rects.size(); i++) {
        for (int j = 0; j < 6; j++) {
            resRects.at<float>(i, j) = _rects[i][j];
        }
    }

    if (rows > 0) {
        rects.create(rows, 6, CV_32FC1);
        rects.assign(resRects);
    }
    return;
}

void drawTestRect(Mat& image, Point2f center, Size2f size, int angle) {
    RotatedRect rRect = RotatedRect(center, size, angle);
    Point2f vertices[4];
    rRect.points(vertices);
    for (int i = 0; i < 4; i++)
        line(image, vertices[i], vertices[(i + 1) % 4], Scalar(0, 0, 255), 3);
}
cv::Mat hsv_filter(const cv::Mat& img, const cv::Mat& hsv_distr) {
    cv::Mat hsv_img;
    cvtColor(img, hsv_img, COLOR_BGR2HSV);
    //imshow("i", hsv_img);
    cv::Mat res = cv::Mat::zeros(hsv_img.rows, hsv_img.cols, CV_8U);
    for (int i = 0; i < hsv_img.rows; i++) {
        for (int j = 0; j < hsv_img.cols; j++) {
            auto hsv = hsv_img.at<cv::Vec3b>(i, j);
            res.at<uchar>(i, j) = hsv_distr.at<uchar>(hsv[0], hsv[1]);
        }
    }
    //imshow("r", res);
    return res;
}

cv::Mat binarize_hsv(const cv::Mat& frame)
{

    cv::Mat hsv;
    cvtColor(frame, hsv, COLOR_BGR2HSV);
    std::vector<cv::Mat> channels(3);
    split(hsv, channels);

    cv::Mat bin_h, bin_s, bin_v, bin_img;
    

    int hmin = 1;
    int hmax = 30;
    int smin = 50;
    int smax = 10000;
    int vmin = 0;
    int vmax = 10000;


    if (hmin >= 0) {
        inRange(channels[0], hmin, hmax, bin_h);
    }
    else {
        cv::Mat range1, range2;
        inRange(channels[0], 180 + hmin, 180, range1);
        inRange(channels[0], 0, hmax, range2);
        bitwise_or(range1, range2, bin_h);
    }  

    inRange(channels[1], smin, smax, bin_s);
    inRange(channels[2], vmin, vmax, bin_v);

    bitwise_and(bin_s, bin_h, bin_img);
    bitwise_and(bin_v, bin_img, bin_img);
   // showDebugImg("binary", bin_img);
    return bin_img;
}

int main(int argc, char** argv) {
    unsigned int start_time = clock();
    Mat img;
    img = imread("test3.png");
   
    cv::Mat gate_hs_distr = imread("new_distr.png");
    GaussianBlur(img, img, Size(9, 9), 2, 2);
    imshow("t", gate_hs_distr);
    cv::Mat element = getStructuringElement(cv::MORPH_ELLIPSE, cv::Size(9, 9));
    cv::Mat dilated, blured;
    dilate(gate_hs_distr, dilated, element);
    blur(dilated, blured, cv::Size(9, 9));
    blured = dilated;
    gate_hs_distr = blured;
    imshow("t2", gate_hs_distr);

    auto gray = binarize_hsv(img);
    imshow("img", gray);
    imwrite("img_out.png", gray);

    
    imshow("img2", gray);
    imshow("g", gray);
    imshow("a", img);
    Mat rects;

    HoughRects(gray, rects, 30, 6, 7, 5, 1);
    unsigned int end_time = clock();
    std::cout << end_time - start_time << std::endl;

    for (int i = 0; i < rects.rows; i++) {
        drawTestRect(img, Point2f(rects.at<float>(i, 0), rects.at<float>(i, 1)), Size2f(rects.at<float>(i, 2), rects.at<float>(i, 3)), rects.at<float>(i, 4));
    }

    imshow("image", img);


    waitKey(0);
    return 0;
}

