#include "common_structs.hpp"
#include "dao_wrapper.hpp"

#include <boost/program_options.hpp>
#include <QVector>
#include <QJsonDocument>
#include <QJsonObject>
#include <QJsonArray>
#include <QDebug>

#include <opencv2/core/core.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/highgui.hpp>

#include <string>
#include <cmath>

#define SHOW 0

// Scale range of templates
float SCALE_RANGE_MIN = 0.9f;
float SCALE_RANGE_MAX = 1.1f;
float SCALE_RANGE_STEP = 0.1f;

BBox parsePositions(QString parStringPositions, cv::Size imgSz)
{
    QJsonParseError error;
    QJsonDocument data = QJsonDocument::fromJson(parStringPositions.toUtf8(), &error);
    QJsonObject obj = data.object();
    if (obj.empty())
        throw std::invalid_argument("BBox Json Empty!");

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

void drawRectangle(cv::Mat parImage, BBox parBbox, cv::Scalar rect_color = cv::Scalar(0, 0, 255))
{
    cv::Rect roi(parBbox.x, parBbox.y, parBbox.width, parBbox.height);
    cv::rectangle(parImage, roi, rect_color, 2);  // RED
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
        throw std::runtime_error("Angle must be exact multiples of 90 degrees");
        return;
    }

    cv::rotate(img, img, flag);
}

#if 1
static cv::Rect rotateScaleRect(const cv::Rect inRect,
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
#else
static cv::Rect rotateScaleRect(const cv::Rect inRect,
        const double scale,
        const double angle,
        const cv::Point2f centerRot,
        const cv::Point2f shift = cv::Point2f())
{
    cv::Mat hom_rot;
    cv::Mat hom_in = cv::Mat::ones(3, 1, CV_64F); // homogeneous coordinates
    cv::Mat rotMatrix = cv::getRotationMatrix2D(cv::Point2f(0.0f, 0.0f), -angle, scale);

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

    cv::Rect roi_rot(tl_rot + shift, br_rot + shift);
    return roi_rot;
}
#endif

void test(const int& tagModelID, const double scale, const double angle)
{
    DAOWrapper* daoWrapper = DAOWrapper::getInstance();
    TagModel tagModel = daoWrapper->getTagModel(tagModelID);

    cv::Mat img_in = cv::imread(tagModel.refImageURL.toStdString());
    if (img_in.empty())
    {
        std::string error_msg = "[ERROR] Error when loading image " + tagModel.refImageURL.toStdString() + "!";
        throw std::runtime_error(error_msg);
    }

    cv::Mat img_painted = img_in.clone();
    cv::Mat img_painted_rot = img_in.clone();
    rotateScaleImage(img_painted_rot, scale, angle);

    std::vector<cv::Rect> fiducials, fiducialsRotated;

    for (const auto& tmf: tagModel.tagFields)
    {
        TagField tagField = daoWrapper->getTagField(tmf.tagFieldID);
        if (tagField.tagFieldTypeID == 3)
        {
            BBox box = parsePositions(tmf.geometricalInfo, img_painted.size());

            // convert BBox to cv::Rect
            cv::Rect roi(box.x, box.y, box.width, box.height);
            drawRectangle(img_painted, box);
            fiducials.emplace_back(roi);

            cv::Rect roiRotated = rotateScaleRect(roi, scale, angle, img_in.size());
            cv::rectangle(img_painted_rot, roiRotated, cv::Scalar(255,0,0), 2);
            fiducialsRotated.emplace_back(roiRotated);
        }
    }

    cv::namedWindow("1");
    cv::moveWindow("1", 0, 0);
    cv::imshow("1", img_painted);

    cv::namedWindow("2");
    cv::moveWindow("2", 1000, 0);
    cv::imshow("2", img_painted_rot);
    cv::waitKey(0);
}

int main(int argc, const char** argv)
{
    boost::program_options::options_description desc("Allowed options");
    float scale, angle;

    const float scale_def=1.0f;
    const float angle_def=90.0f;

    desc.add_options()
        (
            "scale,s",
            boost::program_options::value<float>(&scale)->default_value(scale_def),
            "Scale 0 < s < 1"
        )
        (
            "angle,a",
            boost::program_options::value<float>(&angle)->default_value(angle_def),
            "Angle = multiple of 90"
        )
        (
            "tagmodel_id,t",
            boost::program_options::value<int>(),
            "Tag Model ID (check DB)"
        )
        ("help,h", "Print usage information");
    boost::program_options::positional_options_description pdesc;
    pdesc.add("tagmodel_id", 1);

    // parse options
    boost::program_options::variables_map vm;
    boost::program_options::store(boost::program_options::command_line_parser(argc, argv).options(desc).positional(pdesc).run(), vm);
    boost::program_options::notify(vm);


    if(vm.count("help"))
    {
        std::cout << desc << std::endl;
        return 0;
    }

    int tagModelID;
    if(vm.count("tagmodel_id"))
    {
        tagModelID = vm["tagmodel_id"].as<int>();
    }
    else
    {
        tagModelID = 13;  // VIKING
    }

    std::cout << "tagModelID: " << tagModelID << std::endl;
    test(tagModelID, scale, angle);

    return 0;
}
