#include "line2Dup.h"
#include "utils.hpp"
#include "dao_wrapper.hpp"

#include <QVector>
#include <QJsonDocument>
#include <QJsonObject>
#include <QJsonArray>

#include <opencv2/core/core.hpp>
#include <iostream>
#include <string>


using namespace cv;
using namespace std;

cv::Mat getImage(std::string parImagePath)
{
    cv::Mat img = cv::imread(parImagePath);
    if (img.empty())
    {
        std::string error_msg = "[ERROR] Error when loading image " + parImagePath + "!";
        throw std::runtime_error(error_msg);
    }

    return img;
}

cv::Size getImageSize(std::string parImagePath)
{
    cv::Mat img = cv::imread(parImagePath);
    if (img.empty())
    {
        std::string error_msg = "[ERROR] Error when loading image " + parImagePath + "!";
        throw std::runtime_error(error_msg);
    }

    return img.size();
}

BBox parsePositions(QString parStringPositions, cv::Size imgSz)
{
    QJsonParseError error;
    QJsonDocument data = QJsonDocument::fromJson(parStringPositions.toUtf8(), &error);
    QJsonObject obj = data.object();
    if (obj.empty())
        throw std::invalid_argument("BBox Json Empty!");

    // maybe should be ceil() instead of int()
    BBox b;
    b.x = int(obj["X"].toString().toFloat() * imgSz.width);
    b.y = int(obj["Y"].toString().toFloat() * imgSz.height);
    b.width = int(obj["width"].toString().toFloat() * imgSz.width);
    b.height = int(obj["height"].toString().toFloat() * imgSz.height);
    b.x_pixels = obj["X_pixels"].toString().toInt();
    b.y_pixels = obj["Y_pixels"].toString().toInt();
    b.width_pixels = obj["width_pixels"].toString().toInt();
    b.height_pixels = obj["height_pixels"].toString().toInt();
    b.w_image = obj["w_image"].toString().toInt();
    b.h_image = obj["h_image"].toString().toInt();

    return b;
}

std::vector<ModelTag> extractTagModelFiducialsFromDB()
{
    std::vector<ModelTag> modelTags;

    DAOWrapper* daoWrapper;
    daoWrapper = DAOWrapper::getInstance();

    QVector<TagModel> tagModels = daoWrapper->getAllTagModels();

    foreach (TagModel tagModel, tagModels)
    {
        ModelTag modelTag;
        modelTag.modelID = tagModel.tagModelID;
        modelTag.modelFileName = tagModel.refImageURL.toStdString();
        modelTag.imageSize = getImageSize(modelTag.modelFileName);
        modelTag.modelName = tagModel.name.toStdString();
        foreach (TagModelField tagModelField, tagModel.tagFields)
        {
            TagField tagField = daoWrapper->getTagField(tagModelField.tagFieldID);

            if (tagField.tagFieldTypeID == 3)
            {
                BBox box = parsePositions(tagModelField.geometricalInfo, modelTag.imageSize);

                if (box.x >= 0 && box.y >= 0 && box.x + box.width <= modelTag.imageSize.width && box.y + box.height <= modelTag.imageSize.height)
                {
                    // The crop is within the image bounds
                    cv::Rect crop(box.x, box.y, box.width, box.height);
                    // modelTag.crops.push_back(crop);
                    modelTag.crops.push_back({tagField.tagFieldID, crop});
                }
                else
                {
                    QString fiducialPosError = QString("A posição do fiducial '%1' do modelo '%2' está incorreta. Favor corrigir o banco de templates.")\
                        .arg(tagField.name)\
                        .arg(QString::fromStdString(modelTag.modelName));
                    throw std::invalid_argument(fiducialPosError.toStdString());
                }
            }
        }

        if (modelTag.crops.size() > 0)
            modelTags.push_back(modelTag);
    }
    return modelTags;
}

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

void rotateScaleImage(cv::Mat &img, float scale, float angle)
{
    if (std::abs(scale-1.0) > FLT_EPSILON)
    {
        cv::resize(img, img, cv::Size(), scale, scale);
    }

    int rotation_angle = static_cast<int>(angle);
    cv::RotateFlags flag;
    if(rotation_angle == 90 || rotation_angle == -270)
    {
        // Rotate clockwise 90 degrees
        flag = cv::ROTATE_90_CLOCKWISE;
    }
    else if(rotation_angle == 270 || rotation_angle == -90)
    {
        // Rotate clockwise 270 degrees
        flag = cv::ROTATE_90_COUNTERCLOCKWISE;
    }
    else if(rotation_angle == 180 || rotation_angle == -180)
    {
        // Rotate clockwise 180 degrees
        flag = cv::ROTATE_180;
    }
    else
    {
        return;
    }

    cv::rotate(img, img, flag);
}

cv::Rect rotateScaleRect(const cv::Rect inRect,
        const double scale,
        const double angle,
        const cv::Size imgSize)
{
    cv::Mat hom_in = cv::Mat::ones(3, 1, CV_64F); // homogeneous coordinates
    cv::Mat hom_rot;

    cv::Mat rotMatrix = cv::getRotationMatrix2D(cv::Point2f(0.0f, 0.0f), -angle, scale);

    // get image center
    cv::Point2f centerRot(imgSize.width / 2.0f, imgSize.height / 2.0f);

    // rotate top-left
    auto tl = inRect.tl();
    hom_in.at<double>(0, 0) = tl.x - centerRot.x;
    hom_in.at<double>(1, 0) = tl.y - centerRot.y;
    hom_rot = rotMatrix * hom_in;
    cv::Point2f tl_rot(hom_rot.at<double>(0, 0), hom_rot.at<double>(1, 0));

    // rotate bottom-right
    auto br = inRect.br();
    hom_in.at<double>(0, 0) = br.x - centerRot.x;
    hom_in.at<double>(1, 0) = br.y - centerRot.y;
    hom_rot = rotMatrix * hom_in;
    cv::Point2f br_rot(hom_rot.at<double>(0, 0), hom_rot.at<double>(1, 0));

    // new image center after rotation
    // obs: note the special case of only multiples of 90)
    cv::Point2f shiftScaled;
    float r1 = fmod(angle, 360.0f);
    float diff90 = std::abs(r1-90.0f);
    float diff270 = std::abs(r1-270.0f);

    if (diff90 <= FLT_EPSILON || diff270 <= FLT_EPSILON)
    {
        shiftScaled = cv::Point2f(imgSize.height / 2.0f, imgSize.width / 2.0f)*scale;
    }
    else
    {
        shiftScaled = centerRot*scale;
    }

    cv::Rect roi_rot(tl_rot + shiftScaled, br_rot + shiftScaled);
    return roi_rot;
}

cv::Mat extractFiducialImg(
    const std::map<std::string,cv::Mat>& matched_fiducials,
    const line2Dup::Template& templ
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

void showIndividualMatchings(const cv::Mat& img_roi, const cv::Mat& imgfid_roi,
                            float similarity, std::string modelName,
                            std::vector<std::string>& extraInfo, int index_plot)
{
    // SHOW ROI AND FIDUCIAL SIDE BY SIDE
    cv::Mat frame;
    std::vector<cv::Mat> images = {img_roi, cv::Mat(img_roi.rows, 50, CV_8U, cv::Scalar(0, 0, 0)), imgfid_roi};
    cv::hconcat(images, frame);

    // Add bottom or right area info depending on aspect ratio
    if (extraInfo.size())
    {
        double aspectRatio = static_cast<double>(frame.cols) / frame.rows;

        if (aspectRatio > 1.0)
        {
            // all string elements have same height
            cv::Size stringSize = cv::getTextSize(extraInfo[0], FONT_HERSHEY_PLAIN, 1.0f, 2, 0) + cv::Size(0, 20);
            cv::Mat bottomArea(extraInfo.size()*stringSize.height, frame.cols, CV_8U, cv::Scalar(0, 0, 0));
            int irow = 0;
            for (auto& t: extraInfo)
            {
                cv::Point tloc = cv::Point(10, 20+irow*stringSize.height);
                cv::putText(bottomArea, t, tloc, cv::FONT_HERSHEY_PLAIN, 1.0f, {255, 255, 255}, 1);
                irow++;
            }
            cv::vconcat(frame, bottomArea, frame);
        }
        else
        {
            std::vector<std::string>::iterator it_widest = std::max_element(
                extraInfo.begin(), extraInfo.end(),
                [](std::string &a, std::string &b) -> bool
                {
                    return (a.size() < b.size());
                }
            );
            // choose the widest string element
            cv::Size widestStringSize = cv::getTextSize(*it_widest, FONT_HERSHEY_PLAIN, 1.0f, 2, 0) + cv::Size(50, 0);
            cv::Mat rightArea(frame.rows, widestStringSize.width, CV_8U, cv::Scalar(0, 0, 0));
            int irow = 0;
            for (auto& t: extraInfo)
            {
                cv::Point tloc = cv::Point(10, 50+irow*(widestStringSize.height+10));
                cv::putText(rightArea, t, tloc, cv::FONT_HERSHEY_PLAIN, 1.0f, {255, 255, 255}, 1);
                irow++;
            }
            cv::hconcat(frame, rightArea, frame);
        }
    }

    std::string windowId = to_string(index_plot);
    cv::namedWindow(windowId, WINDOW_AUTOSIZE);
    cv::setWindowTitle(windowId, modelName);

    int screen_columns = 4;
    int irow = index_plot / screen_columns;
    int icol = index_plot % screen_columns;

    cv::moveWindow(windowId, 450*icol, 200*irow);
    cv::imshow(windowId, frame);
}

int showAllMatchings(const cv::Mat& img,
    const std::vector<line2Dup::Match>& matches,
    const std::vector<int>& indices,
    const std::map<std::string, cv::Mat>& matched_fiducials,
    line2Dup::Detector detector, std::string windowLabel)
{
    cv::Mat img_show = img.clone();
    for (auto idx: indices)
    {
        auto match = matches[idx];
        auto templ = detector.getTemplates(match.class_id, match.template_id);

        // templ[0] == base of pyramid
        int x = templ[0].width + match.x;
        int y = templ[0].height + match.y;
        int r = templ[0].width/2;

        cv::Vec3b randColor = {255, 0, 0};

        for(int i = 0; i < templ[0].features.size(); i++){
            auto feat = templ[0].features[i];
            cv::circle(img_show, {feat.x + match.x, feat.y + match.y}, 2, randColor, -1);
        }

        // Box
        cv::rectangle(img_show, {match.x, match.y}, {x, y}, randColor, 2);
        cv::putText(
            img_show,
            to_string(match.template_id), cv::Point(match.x+r-10, match.y-3),
            cv::FONT_HERSHEY_PLAIN, 1.0f, randColor, 2
        );
    }

    cv::namedWindow(windowLabel);
    cv::moveWindow(windowLabel, 680, 400);
    cv::imshow(windowLabel, img_show);

    return cv::waitKey(0);
}

std::vector<double> calcHistogram(const cv::Mat1b& img, int histSize)
{
    std::vector<double> histogramArray(histSize, 0);

    for (int i = 0; i < img.rows; i++)
    {
        for (int j = 0; j < img.cols; j++)
        {
            int grayLevel = (int)(img.at<uchar>(i, j));
            histogramArray[grayLevel]++;
        }
    }

    for (int i = 0; i < histSize; i++)
    {
        histogramArray[i] = histogramArray[i]/(double)(img.cols * img.rows);
    }
    return histogramArray;
}

double compHistogram(const std::vector<double>& h1, const std::vector<double>& h2)
{

    const double* hist1;
    const double* hist2;
    hist1 = (const double*) h1.data();
    hist2 = (const double*) h2.data();

    double mean1 = 0, mean2 = 0;

    for(unsigned int i = 0; i<h1.size(); i++)
    {
        mean1 += hist1[i];
        mean2 += hist2[i];
    }
    mean1/=h1.size();
    mean2/=h2.size();

    double r1=0,r2=0,r3=0;
    double t1,t2;
    for(unsigned int i = 0; i<h1.size(); i++)
    {
        t1 = hist1[i]-mean1;
        t2 = hist2[i]-mean2;

        r1+=t1*t1;
        r2+=t2*t2;
        r3+=t1*t2;
    }
    return r3/sqrt(r1*r2);
}

std::pair<cv::Scalar, cv::Mat> evalSSIM(const cv::Mat& img1_in, const cv::Mat& img2_in)
{

	// default settings (data range of images = 255)
	const float C1 = 6.5025, C2 = 58.5225;

	cv::Mat img1, img2;

	img1_in.convertTo(img1, CV_32FC3);
	img2_in.convertTo(img2, CV_32FC3);

	cv::Mat mu1, mu1_sq;
	cv::GaussianBlur(img1, mu1, cv::Size(11, 11), 1.5f, 1.5f);
	cv::multiply(mu1, mu1, mu1_sq);
	cv::Mat img1_sq, sigma1_sq;
	cv::multiply(img1, img1, img1_sq);
	cv::GaussianBlur(img1_sq, sigma1_sq, cv::Size(11, 11), 1.5f, 1.5f);
	cv::subtract(sigma1_sq, mu1_sq, sigma1_sq);

	cv::Mat mu2, mu2_sq;
	cv::GaussianBlur(img2, mu2, cv::Size(11, 11), 1.5f, 1.5f);
	cv::multiply(mu2, mu2, mu2_sq);
	cv::Mat img2_sq, sigma2_sq;
	cv::multiply(img2, img2, img2_sq);
	cv::GaussianBlur(img2_sq, sigma2_sq, cv::Size(11, 11), 1.5f, 1.5f);
	cv::subtract(sigma2_sq, mu2_sq, sigma2_sq);


	cv::Mat mu1_mu2;
	cv::multiply(mu1, mu2, mu1_mu2);

	cv::Mat img1_img2, sigma12;
	cv::multiply(img1, img2, img1_img2);
	cv::GaussianBlur(img1_img2, sigma12, cv::Size(11, 11), 1.5f, 1.5f);
	cv::subtract(sigma12, mu1_mu2, sigma12);

	// numerator
	cv::Mat t1;
	cv::multiply(mu1_mu2, 2.0, t1);
	cv::add(t1, C1, t1);

	cv::Mat t2;
	cv::multiply(sigma12, 2.0, t2);
	cv::add(t2, C2, t2);

	cv::Mat t3;
	cv::multiply(t1, t2, t3);

	// denominator
	cv::add(mu1_sq, mu2_sq, t1);
	cv::add(t1, C1, t1);
	cv::add(sigma1_sq, sigma2_sq, t2);
	cv::add(t2, C2, t2);

	cv::multiply(t1, t2, t1);
	cv::Mat ssim_map;
	cv::divide(t3, t1, ssim_map);

#if 1
	//skimage: to avoid edge effects will ignore filter radius strip around edges
	cv::Rect crop(5, 5, ssim_map.cols-5, ssim_map.rows-5);
	cv::Mat ssim_map_cropped = ssim_map(crop);
	const cv::Scalar mssim = cv::mean(ssim_map_cropped);
	return {mssim, std::move(ssim_map_cropped)};
#else
	const cv::Scalar mssim = cv::mean(ssim_map);
	return {mssim, std::move(ssim_map)};
#endif
}
