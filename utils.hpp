#ifndef UTILS_H
#define UTILS_H

#include <opencv2/core/core.hpp>

#include <chrono>
#include <iostream>
#include <vector>
#include <algorithm>

class Timer
{
public:
    Timer() : beg_(clock_::now()) {}
    void reset() { beg_ = clock_::now(); }
    double elapsed() const {
        return std::chrono::duration_cast<second_>
            (clock_::now() - beg_).count(); }
    void out(std::string message = ""){
        double t = elapsed();
        // std::cout << message << "\nelasped time:" << t << "s" << std::endl;
        std::cout << message << " >>> elapsed time: " << t << " ms" << std::endl;
        reset();
    }
private:
    typedef std::chrono::high_resolution_clock clock_;
    // typedef std::chrono::duration<double, std::ratio<1> > second_;
    typedef std::chrono::duration<double, std::milli> second_;
    std::chrono::time_point<clock_> beg_;
};

cv::Mat displayQuantized(const cv::Mat& quantized);
cv::Mat extractFiducialImg(
    const std::map<std::string,cv::Mat>& matched_fiducials,
    const line2Dup::Template& templ
);

int showQuantization(const cv::Mat& img, line2Dup::Detector detector, std::string windowLabel="window");

void showIndividualMatchings(const cv::Mat& img_roi, const cv::Mat& imgfid_roi,
                            float similarity, std::string modelName,
                            std::vector<std::string>& extraInfo, int index_plot = 0);

int showAllMatchings(const cv::Mat& img, const std::vector<line2Dup::Match>& matches, const std::vector<int>& indices,
    const std::map<std::string, cv::Mat>& matched_fiducials,
    line2Dup::Detector detector, std::string windowLabel="window");

std::vector<double> calcHistogram(const cv::Mat1b& img, int histSize = 256);

double compHistogram(const std::vector<double>& h1, const std::vector<double>& h2);

#endif /* UTILS_H */
