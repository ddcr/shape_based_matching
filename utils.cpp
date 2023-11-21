#include "line2Dup.h"
#include <opencv2/core/core.hpp>
#include <iostream>
#include <string>

std::map<std::string,std::string> map_models = {
    {"2_1688586489", "EXAR"},
    {"3_1688650244", "TAIYO YUNDEN"},
    {"4_1688651426", "SAMSUNG"},
    {"5_1688652122", "YAGEO"},
    {"6_1688652831", "HILISIN"},
    {"7_1688653607", "WALSIN SINCERA"},
    {"8_1688654280", "AMAZING"},
    {"9_1688655019", "LRC"},
    {"10_1688655560", "DARFON"},
    {"11_1688656382", "MURATA"},
    {"12_1688659851", "WALSIN"},
    {"13_1688661142", "VIKING"},
    {"14_1688661964", "LITEON"},
    {"15_1688663919", "VISHAY"},
    {"16_1688765210", "DARFON_02"},
    {"19_1689171468", "JOHANSON"},
    {"20_1689179930", "MICROCHIP"},
    {"21_1689693018", "HILSIN_02"}
};

using namespace cv;
using namespace std;

static inline int myGetLabel(int quantized)
{
  switch (quantized)
  {
    case 1:   return 0;
    case 2:   return 1;
    case 4:   return 2;
    case 8:   return 3;
    case 16:  return 4;
    case 32:  return 5;
    case 64:  return 6;
    case 128: return 7;
    default:
      CV_Error(Error::StsBadArg, "Invalid value of quantized parameter");
  }
}

cv::Mat displayQuantized(const Mat& quantized)
{
    std::vector<Vec3b> lut(8);
    lut[0] = Vec3b(  0,   0, 255);
    lut[1] = Vec3b(  0, 170, 255);
    lut[2] = Vec3b(  0, 255, 170);
    lut[3] = Vec3b(  0, 255,   0);
    lut[4] = Vec3b(170, 255,   0);
    lut[5] = Vec3b(255, 170,   0);
    lut[6] = Vec3b(255,   0,   0);
    lut[7] = Vec3b(255,   0, 170);

    cv::Mat dst = Mat::zeros(quantized.size(), CV_8UC3);
    for (int r = 0; r < dst.rows; ++r)
    {
        const uchar* quant_r = quantized.ptr(r);
        Vec3b* dst_r = dst.ptr<Vec3b>(r);
        for (int c = 0; c < dst.cols; ++c)
        {
        uchar q = quant_r[c];
        if (q)
            dst_r[c] = lut[myGetLabel(q)];
        }
    }
    return dst;
}

cv::Mat extractFiducialImg(
    const std::map<std::string,cv::Mat>& matched_fiducials,
    const line2Dup::Template& templ,
    bool addTitle=true
)
{
    cv::Mat dst;

    cv::Mat src = matched_fiducials.at(templ.fiducial_src).clone();
    float angle = templ.orientation;

    if (std::abs(angle - 90.0) < ANGLE_TOLERANCE)
    {
        cv::rotate(src, dst, cv::ROTATE_90_CLOCKWISE);
    }
    else if (std::abs(angle - 180.0) < ANGLE_TOLERANCE)
    {
        cv::rotate(src, dst, cv::ROTATE_180);
    }
    else if (std::abs(angle - 270.0) < ANGLE_TOLERANCE)
    {
        cv::rotate(src, dst, cv::ROTATE_90_COUNTERCLOCKWISE);
    }
    else
    {
        src.copyTo(dst);
    }

    if (std::abs(templ.sscale-1.0) > FLT_EPSILON)
    {
        cv::resize(dst, dst, cv::Size(), templ.sscale, templ.sscale);
    }

    if (addTitle)
    {
        std::string fiducial_src = templ.fiducial_src;
        std::string fiducial_substr = fiducial_src.substr(0, fiducial_src.find('.'));
        std::string modelName = map_models.at(fiducial_substr);
        cv::putText(dst, modelName, cv::Point(0, dst.rows/2), cv::FONT_HERSHEY_PLAIN, 1.0f, {0, 0, 0}, 2);
    }

    return dst;
}

int showQuantization(const cv::Mat& img, line2Dup::Detector detector, std::string windowLabel)
{
    cv::Mat cimg = img.clone();
    cv::resize(cimg, cimg, cv::Size(), 0.5f, 0.5f);

    cv::Mat img_mag_norm, img_nq_norm, img_q_norm, cimg_gray_norm;
    cv::Ptr<line2Dup::ColorGradientPyramid> colorGradientPyramid = detector.getModalities()->process(cimg, cv::Mat());
    cv::Mat img_mag = colorGradientPyramid->magnitude;  // CV_32F
    cv::Mat img_nq = colorGradientPyramid->angle_ori;   // CV_32F
    cv::Mat img_q = colorGradientPyramid->angle;        // CV_8U

    cv::normalize(img_mag, img_mag_norm, 0, 255, cv::NORM_MINMAX, CV_8U);
    cv::normalize(img_nq, img_nq_norm, 0, 255, cv::NORM_MINMAX, CV_8U);

    cv::applyColorMap(img_mag_norm, img_mag_norm, cv::COLORMAP_VIRIDIS);
    cv::applyColorMap(img_nq_norm, img_nq_norm, cv::COLORMAP_VIRIDIS);

    cv::Mat hconcat1, hconcat2, concatenated;
    cv::hconcat(cimg, displayQuantized(img_q), hconcat1);
    cv::hconcat(img_mag_norm, img_nq_norm, hconcat2);
    cv::vconcat(hconcat1, hconcat2, concatenated);

    cv::namedWindow(windowLabel, WINDOW_AUTOSIZE);
    cv::moveWindow(windowLabel, 80, 50);
    cv::imshow(windowLabel, concatenated);

    return cv::waitKey(0);
}

int showMatchings(const cv::Mat& img, const std::vector<line2Dup::Match>& matches, const std::vector<int>& indices,
    const std::map<std::string, cv::Mat>& matched_fiducials,
    line2Dup::Detector detector, std::string windowLabel)
{
    cv::Mat img_show = img.clone();
    int iWindow = 0;
    int jWindow = 0;
    for (auto idx: indices)
    {
        auto match = matches[idx];
        auto templ = detector.getTemplates(match.class_id, match.template_id);

        // 1.1 Extract matched ROI
        cv::Rect templ_roi = cv::Rect(match.x, match.y, templ[0].width, templ[0].height);
        cv::Mat img_roi = img(templ_roi).clone();
        cv::Mat img_roi_gray;
        cv::cvtColor(img_roi, img_roi_gray, cv::COLOR_BGR2GRAY);

        // 1.2 extract fiducial with model name imprinted
        cv::Mat img_fiducial = extractFiducialImg(matched_fiducials, templ[0]);

        // ================================== SHOW ROI AND FIDUCIAL SIDE BY SIDE ==================================
        cv::Mat frame;
        std::vector<cv::Mat> images = {img_roi_gray, img_fiducial};
        int max_height = std::max(img_roi_gray.rows, img_fiducial.rows);
        int padding = 5;

        for (const cv::Mat& img : images)
        {
            int top = 0, bottom = 0;
            if (img.rows < max_height)
            {
                top = (max_height - img.rows) / 2;
                bottom = max_height - img.rows - top;
            }
            cv::Mat padded_img;
            cv::copyMakeBorder(img, padded_img, top + padding, bottom + padding,
                padding, padding, cv::BORDER_CONSTANT);
            if (frame.empty())
                frame = padded_img;
            else cv::hconcat(frame, padded_img, frame);
        }

        std::string windowId = to_string(match.template_id);
        std::string windowIdTitle = "Box" + to_string(match.template_id) + "/" + to_string(int(round(match.similarity)));
        cv::namedWindow(windowId, WINDOW_AUTOSIZE);
        cv::moveWindow(windowId, (iWindow % 5)*350 + 80, (jWindow % 2)*160+50);
        cv::setWindowTitle(windowId, windowIdTitle);
        cv::imshow(windowId, frame);
        // ================================== SHOW ROI AND FIDUCIAL SIDE BY SIDE ==================================

        // templ[0] == base of pyramid
        int x = templ[0].width + match.x;
        int y = templ[0].height + match.y;
        int r = templ[0].width/2;

        // cv::Vec3b randColor = {rand()%155+100, rand()%155+100, rand()%155+100};
        cv::Vec3b randColor = {255, 0, 0};

        for(int i = 0; i < templ[0].features.size(); i++){
            auto feat = templ[0].features[i];
            cv::circle(img_show, {feat.x + match.x, feat.y + match.y}, 2, randColor, -1);
        }

        //  log to stdout
        std::stringstream log_t, sscale_t;
        sscale_t.precision(2);
        sscale_t << templ[0].sscale;
        log_t << "Box " << to_string(match.template_id) << " : "
                << "[" << templ[0].fiducial_src << "], "
                << "(" << int(templ[0].orientation) << ", " << sscale_t.str() << "), sim="
                << to_string(int(round(match.similarity)))
                << "..."
                << templ_roi
                << " -- "
                << img_fiducial.size();
        std::cout << log_t.str() << std::endl;

        // Box
        cv::rectangle(img_show, {match.x, match.y}, {x, y}, randColor, 2);
        cv::putText(
            img_show,
            to_string(match.template_id), cv::Point(match.x+r-10, match.y-3),
            cv::FONT_HERSHEY_PLAIN, 1.0f, randColor, 2
        );

        iWindow++;
        if (((iWindow % 5) == 0))
            jWindow++;
    }

    cv::namedWindow(windowLabel);
    cv::moveWindow(windowLabel, 610, 250);
    cv::imshow(windowLabel, img_show);

    return cv::waitKey(0);
}
