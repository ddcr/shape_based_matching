#include "line2Dup.h"
#include <memory>
#include <iostream>
#include <boost/program_options.hpp>
#include <assert.h>
#include <chrono>
#include <regex>
#include <experimental/filesystem>
using namespace std;
using namespace cv;

cv::Mat displayQuantized(const cv::Mat& quantized);

static std::string PREFIX_PATH = "/home/ivision/jabil_tag_reader/dev_area/jabil_dev_phase4";

/**
 * \brief Detector Constructor.
 *
 * \param weak_threshold   When quantizing, discard gradients with magnitude less than this.
 * \param num_features     How many features a template must contain.
 * \param strong_threshold Consider as candidate features only gradients whose norms are
 *                         larger than this.
*/
int DEF_NUM_FEATURE = 150;
float DEF_WEAK_THRESHOLD = 235.0f;
float DEF_STRONG_THRESHOLD = 240.0f;

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
// NMS, got from cv::dnn so we don't need opencv contrib
// just collapse it
namespace  cv_dnn
{
    namespace
    {
        template <typename T>
        static inline bool SortScorePairDescend(const std::pair<float, T>& pair1,
                                const std::pair<float, T>& pair2)
        {
            return pair1.first > pair2.first;
        }
    } // namespace

    inline void GetMaxScoreIndex(const std::vector<float>& scores, const float threshold, const int top_k,
                        std::vector<std::pair<float, int> >& score_index_vec)
    {
        for (size_t i = 0; i < scores.size(); ++i)
        {
            if (scores[i] > threshold)
            {
                score_index_vec.push_back(std::make_pair(scores[i], i));
            }
        }
        std::stable_sort(score_index_vec.begin(), score_index_vec.end(),
                        SortScorePairDescend<int>);
        if (top_k > 0 && top_k < (int)score_index_vec.size())
        {
            score_index_vec.resize(top_k);
        }
    }

    template <typename BoxType>
    inline void NMSFast_(const std::vector<BoxType>& bboxes,
        const std::vector<float>& scores, const float score_threshold,
        const float nms_threshold, const float eta, const int top_k,
        std::vector<int>& indices, float (*computeOverlap)(const BoxType&, const BoxType&))
    {
        CV_Assert(bboxes.size() == scores.size());
        std::vector<std::pair<float, int> > score_index_vec;
        GetMaxScoreIndex(scores, score_threshold, top_k, score_index_vec);

        // Do nms.
        float adaptive_threshold = nms_threshold;
        indices.clear();
        for (size_t i = 0; i < score_index_vec.size(); ++i) {
            const int idx = score_index_vec[i].second;
            bool keep = true;
            for (int k = 0; k < (int)indices.size() && keep; ++k) {
                const int kept_idx = indices[k];
                float overlap = computeOverlap(bboxes[idx], bboxes[kept_idx]);
                keep = overlap <= adaptive_threshold;
            }
            if (keep)
                indices.push_back(idx);
            if (keep && eta < 1 && adaptive_threshold > 0.5) {
            adaptive_threshold *= eta;
            }
        }
    }


    // copied from opencv 3.4, not exist in 3.0
    template<typename _Tp> static inline
    double jaccardDistance__(const Rect_<_Tp>& a, const Rect_<_Tp>& b) {
        _Tp Aa = a.area();
        _Tp Ab = b.area();

        if ((Aa + Ab) <= std::numeric_limits<_Tp>::epsilon()) {
            // jaccard_index = 1 -> distance = 0
            return 0.0;
        }

        double Aab = (a & b).area();
        // distance = 1 - jaccard_index
        return 1.0 - Aab / (Aa + Ab - Aab);
    }

    template <typename T>
    static inline float rectOverlap(const T& a, const T& b)
    {
        return 1.f - static_cast<float>(jaccardDistance__(a, b));
    }

    void NMSBoxes(const std::vector<Rect>& bboxes, const std::vector<float>& scores,
                            const float score_threshold, const float nms_threshold,
                            std::vector<int>& indices, const float eta=1, const int top_k=0)
    {
        NMSFast_(bboxes, scores, score_threshold, nms_threshold, eta, top_k, indices, rectOverlap);
    }
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

/* ================================================================================================== */

cv::Mat extractFiducialImg(
    const std::map<std::string,cv::Mat>& matched_fiducials,
    const line2Dup::Template& templ,
    bool addTitle = true
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

int showQuantization(const cv::Mat& img, line2Dup::Detector detector, std::string windowLabel="window")
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
    line2Dup::Detector detector, std::string windowLabel="window")
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

void jabil_test1()
{
    int num_feature = 150;
    line2Dup::Detector detector(num_feature, {4, 8}, 235.0f, 240.0f);

    // read JABIL models
    const std::experimental::filesystem::path path{ PREFIX_PATH + "/model_images" };
    std::vector<std::experimental::filesystem::path> filelist;
    if (is_directory(path))
    {
        for (auto const& dir_entry : std::experimental::filesystem::directory_iterator{ path })
            filelist.push_back(dir_entry.path());
    }
    else if (is_regular_file(path))
    {
        filelist.push_back(path);
    }
    else
    {
        std::cerr << "The folder/file specified was invalid!" << std::endl;
    }

    for (auto &f: filelist)
    {
        // We are insterested only in the original files
        if (f.stem().has_extension())
        {
            continue;
        }

        std::experimental::filesystem::path output_file_mag, output_file_ang;
        output_file_mag = f;
        output_file_mag.replace_extension(".0.jpg");

        output_file_ang = f;
        output_file_ang.replace_extension(".1.jpg");

        Mat test_img = imread(f.string());
        assert(!test_img.empty() && "check your img path");

        // make the img having 32*n width & height
        // at least 16*n here for two pyrimads with strides 4 8
        int stride = 32;
        int n = test_img.rows/stride;
        int m = test_img.cols/stride;
        Rect roi(0, 0, stride*m , stride*n);
        Mat img = test_img(roi).clone();
        assert(img.isContinuous());

        Timer timer;
        // match, img, min score, ids
        // auto matches = detector.match(img, 90, ids);
        // qp = detector->process(img, cv::Mat());
        Ptr<line2Dup::ColorGradientPyramid> qp;
        qp = detector.getModalities()->process(img, cv::Mat());

        cv::imwrite(output_file_mag.string(), qp->magnitude);
        cv::imwrite(output_file_ang.string(), qp->angle);

        timer.out();

        // For each pyramid level, precompute linear memories for each ColorGradient
        for (int l = 0; l < detector.pyramidLevels(); ++l)
        {
            int T = detector.getT(l);
            if (l > 0)
            {
                qp->pyrDown();
            }
            Mat quantized, spread_quantized;
            qp->quantize(quantized);
            line2Dup::spread(quantized, spread_quantized, T);

            std::experimental::filesystem::path output_file_quantized;
            output_file_quantized = f;
            char quantized_ext[300];
            sprintf(quantized_ext, "pyr_%d.jpg", l);
            output_file_quantized.replace_extension(quantized_ext);

            Mat quantized_color = displayQuantized(quantized);
            cv::imwrite(output_file_quantized.string(), quantized_color);

            std::vector<Mat> response_maps;
            line2Dup::computeResponseMaps(spread_quantized, response_maps);
            // std::cout << "N = " << response_maps.size() << std::endl;
            for (int k = 0; k < response_maps.size(); ++k)
            {
                std::experimental::filesystem::path output_file_response;
                output_file_response = f;
                char response_ext[300];
                sprintf(response_ext, "pyr_%d.%d.jpg", l, k);
                output_file_response.replace_extension(response_ext);
                cv::imwrite(output_file_response.string(), response_maps[k]);
            }

        }

    }
}

void jabil_match()
{
    int num_feature = 150;
    line2Dup::Detector detector(num_feature, {4, 8}, 235.0f, 240.0f);

#if 0
    // read JABIL image
    const std::experimental::filesystem::path path{ PREFIX_PATH + "/model_images" };
    std::vector<std::experimental::filesystem::path> filelist;

    const std::regex base_regex("11_1688656382");
    std::smatch base_match;
    for (auto const& dir_entry : std::experimental::filesystem::directory_iterator{ path })
    {
        std::string file_str = dir_entry.path().stem().string();
        if (std::regex_match(file_str, base_match, base_regex))
        {
            filelist.push_back(dir_entry.path());
        }
    }
#else
    const std::experimental::filesystem::path path{ PREFIX_PATH + "/inspection_images/2023-07-27/JabilCam" };
    std::vector<std::experimental::filesystem::path> filelist;

    for (auto const& dir_entry : std::experimental::filesystem::directory_iterator{ path })
    {
        filelist.push_back(dir_entry.path());
    }
#endif

    std::vector<std::string> class_ids;
    class_ids.push_back("11_1688656382");
    detector.readClasses(class_ids, "%s_templ.yaml");

    for (auto &f: filelist)
    {
        std::cout << f << std::endl;

        Mat img_orig = imread(f.string());
        assert(!img_orig.empty() && "check your img path");

        int stride = 16;
        int n = img_orig.rows/stride;
        int m = img_orig.cols/stride;
        Rect roi(0, 0, stride*m , stride*n);

        Mat img = img_orig(roi).clone();

        Timer timer;
        auto matches = detector.match(img, 95, class_ids);
        timer.out();
        std::cout << "matches.size(): " << matches.size() << std::endl;

        for (auto match: matches)
        {
            for (auto class_id: class_ids)
            {
                auto templ = detector.getTemplates(class_id, match.template_id);

                int x = templ[0].width + match.x;
                int y = templ[0].height + match.y;
                int r = templ[0].width/2;

                for(int i = 0; i < templ[0].features.size(); i++){
                    auto feat = templ[0].features[i];
                    cv::circle(
                        img,
                        {feat.x + match.x, feat.y + match.y},
                        2,
                        {150, 0, 150},
                        -1
                    );
                }

                cv::putText(
                    img,
                    to_string(int(round(match.similarity))),
                    cv::Point(match.x + r -10, match.y - 3),
                    cv::FONT_HERSHEY_PLAIN,
                    2,
                    {150, 0, 150}
                );
                cv::rectangle(
                    img,
                    {match.x, match.y},
                    {x, y},
                    {150, 0, 150},
                    2
                );
            }
        }

        // if (matches.size() > 0)
        // {
        //     cv::imshow("", img);
        //     cv::waitKey(0);
        // }
    }
}

void jabil_create_one_template()
{
    int num_feature = 150;
    line2Dup::Detector detector(num_feature, {4, 8}, 235.0f, 240.0f);

    // read JABIL fiducial crops
    const std::experimental::filesystem::path template_path{ PREFIX_PATH + "/model_images" };
    std::vector<std::experimental::filesystem::path> filelist;

    const std::regex base_regex("11_1688656382..*");
    std::smatch base_match;
    for (auto const& dir_entry : std::experimental::filesystem::directory_iterator{ template_path })
    {
        std::string file_str = dir_entry.path().stem().string();
        if (std::regex_match(file_str, base_match, base_regex))
        {
            filelist.push_back(dir_entry.path());
        }
    }

    // run all fiducials
    std::string class_id = "11_1688656382";
    for (auto &f: filelist)
    {
        Mat fiducial_img = imread(f.string());
        assert(!fiducial_img.empty() && "check your img path");

        // ONLY ALLOW MULTIPLES OF 90 DEGREES
        shape_based_matching::shapeInfo_producer fid_shapes(fiducial_img, cv::Mat());
        fid_shapes.angle_range = {0, 270};
        fid_shapes.angle_step = 90;

        fid_shapes.scale_range = {0.8, 1.2};
        fid_shapes.scale_step = 0.1;
        // fid_shapes.scale_range = { 1.0 };

        fid_shapes.produce_infos();
        for (auto& info: fid_shapes.infos)
        {
            cv::Mat to_show = fid_shapes.src_of(info);
            int templ_id = detector.addTemplate(fid_shapes.src_of(info), class_id, fid_shapes.mask_of(info));

            // visualize the features
            auto templ = detector.getTemplates(class_id, templ_id);
            for(int i = 0; i < templ[0].features.size(); i++)
            {
                auto feat = templ[0].features[i];
                cv::circle(to_show, {feat.x+templ[0].tl_x, feat.y+templ[0].tl_y}, 3, {0, 0, 255}, -1);
            }

            // will be faster if not showing this
            std::cout << "Angle: " << info.angle << ", Scale: " << info.scale << std::endl;
            imshow("train", to_show);
            waitKey(0);
        }
    }
    detector.writeClasses("%s_templ.yaml");

    std::cout << detector.numClasses() << std::endl;
    std::cout << detector.numTemplates() << std::endl;
}

void jabil_create_templates(line2Dup::Detector detector)
{
    // line2Dup::Detector detector(DEF_NUM_FEATURE, {4, 8}, DEF_WEAK_THRESHOLD, DEF_STRONG_THRESHOLD);

    // read JABIL fiducial crops
    const std::experimental::filesystem::path template_path{ PREFIX_PATH + "/model_images" };
    std::map<std::string, std::vector<std::string>> file_map;

    const std::regex base_regex("\\d+_\\d+\\..*");
    std::smatch base_match;
    for (auto const& dir_entry : std::experimental::filesystem::directory_iterator{ template_path })
    {
        std::string file_str = dir_entry.path().stem().string();
        if (std::regex_match(file_str, base_match, base_regex))
        {
            file_map[dir_entry.path().stem().stem()].push_back(dir_entry.path());
        }
    }

    for (auto const& fm: file_map)
    {
        std::string class_id = fm.first;
        for (auto const& f: fm.second)
        {
            cv::Mat fiducial_img = cv::imread(f);
            assert(!fiducial_img.empty() && "check your img path");

            std::experimental::filesystem::path f_path(f);
            std::string fiducial_src = f_path.filename();

            // ONLY ALLOW MULTIPLES OF 90 DEGREES
            shape_based_matching::shapeInfo_producer fid_shapes(fiducial_img, cv::Mat());

            if (0)
            {
                // create mask
                cv::Mat fiducial_mask = cv::Mat(fiducial_img.size(), CV_8UC1, {255});
                shape_based_matching::shapeInfo_producer fid_shapes(fiducial_img, fiducial_mask);
            }

            fid_shapes.angle_range = {0, 270};
            fid_shapes.angle_step = 90;

            fid_shapes.scale_range = {0.9, 1.1};
            // fid_shapes.scale_range = {0.8, 1.2};
            fid_shapes.scale_step = 0.1;
            fid_shapes.produce_infos();

            // ddcr: write info directly in template yaml file
            // std::vector<shape_based_matching::shapeInfo_producer::Info> infos_have_templ;
            for (auto& info: fid_shapes.infos)
            {
                int templ_id = detector.addTemplate(
                    fid_shapes.src_of(info),
                    class_id,
                    fid_shapes.mask_of(info),
                    info.scale,
                    info.angle,
                    fiducial_src
                );
                // if (templ_id != -1)
                // {
                //     infos_have_templ.push_back(info);
                // }
            }
            // fid_shapes.save_infos(infos_have_templ, f + ".info.yaml");
        }

        std::cout << "Writing templates ...";
        detector.writeClasses( template_path.string() + "/%s_templ.yaml.gz");
        std::cout << " done!" << std::endl;
    }
}

void jabil_read_all_templates_and_match(
    std::string testdir, float weak_thresh, float strong_thresh, int num_feat, bool create_template
)
{
    line2Dup::Detector detector(num_feat, {4, 8}, weak_thresh, strong_thresh);

    if (create_template)
    {
        jabil_create_templates(detector);
    }

    // read JABIL template models
    std::cout << "Reading templates ... ";
    const std::experimental::filesystem::path template_path{ PREFIX_PATH + "/model_images"};
    std::vector<std::string> class_ids;

    const std::regex base_regex("(.*)_templ\\.yaml.gz");
    for (auto const& dir_entry : std::experimental::filesystem::directory_iterator{ template_path })
    {
        std::string file_str = dir_entry.path().filename().string();
        auto p_beg = std::sregex_iterator(file_str.begin(), file_str.end(), base_regex);
        auto p_end = std::sregex_iterator();

        for (std::sregex_iterator p = p_beg; p != p_end; ++p)
        {
            class_ids.push_back((*p)[1]);
        }
    }
    std::cout << " done!" << std::endl;

    detector.readClasses(class_ids, template_path.string() + "/%s_templ.yaml.gz");
    std::cout << detector.numClasses() << std::endl;
    for (auto const& class_id: detector.classIds())
    {
        std::cout << class_id << std::endl;
    }
    std::cout << detector.numTemplates() << std::endl;

    // Take time of matching for one image
    // read test images
    const std::experimental::filesystem::path path_test_images{
        PREFIX_PATH + "/inspection_images/2023-07-27/JabilCam-modelos/tag_candidate/" + testdir
    };

    std::vector<std::experimental::filesystem::path> filelist;
    for (auto const& dir_entry : std::experimental::filesystem::directory_iterator{ path_test_images })
    {
        filelist.push_back(dir_entry.path());
    }

    for (auto &f: filelist)
    {
        Timer timer_wall;

        std::cout << "===========================================================================================" << std::endl;
        std::cout << f.filename() << std::endl;

        cv::Mat img_orig = imread(f.string());
        assert(!img_orig.empty() && "check your img path");

        // compatibility wth line2Dup::computeResponseMaps()
        // make the img having 16*n width & height
        int stride = 16;
        int n = img_orig.rows/stride;
        int m = img_orig.cols/stride;
        Rect roi(0, 0, stride*m , stride*n);
        Mat img = img_orig(roi).clone();

        Timer timer_match;
        auto matches = detector.match(img, 90, class_ids);
        timer_match.out("[detector match]");
        std::cout << "matches.size(): " << matches.size() << std::endl;

        Timer timer_filter;
        // ======================================= NMS =======================================
        // Use NMS to filter overlapping boxes of different scales but similar score
        std::vector<cv::Rect> boxes;
        std::vector<float> scores;
        std::vector<int> indices;
        for(auto match: matches){
            cv::Rect box;
            box.x = match.x;
            box.y = match.y;

            auto templ = detector.getTemplates(match.class_id, match.template_id);

            box.width = templ[0].width;
            box.height = templ[0].height;
            boxes.push_back(box);
            scores.push_back(match.similarity);
        }
        cv_dnn::NMSBoxes(boxes, scores, 0, 0.5f, indices);
        // cv_dnn::NMSBoxes(boxes, scores, 0, 0.8f, indices, 1.0, 2);
        // ======================================= NMS =======================================

        // First pass: extract fiducial images
        std::map<std::string, cv::Mat> matched_fiducial_crops;
        for (auto idx: indices)
        {
            auto match = matches[idx];
            auto templ = detector.getTemplates(match.class_id, match.template_id);
            std::experimental::filesystem::path fiducial_path;
            std::experimental::filesystem::path fiducial_src_p(templ[0].fiducial_src);
            fiducial_path = template_path / fiducial_src_p;
            cv::Mat img_fid_gray = cv::imread(fiducial_path.string(), cv::IMREAD_GRAYSCALE);
            matched_fiducial_crops[templ[0].fiducial_src] = img_fid_gray;
        }

        if (showMatchings(img, matches, indices, matched_fiducial_crops, detector) == 113)
        {
            break;
        }

        if(showQuantization(img, detector, f.filename()) == 113)
        {
            break;
        }
        cv::destroyAllWindows();

        // Second pass: to filtering
        // for (auto idx: indices)
        // {
        //     auto match = matches[idx];
        //     auto templ = detector.getTemplates(match.class_id, match.template_id);
        // }
        // timer_wall.out("File processing");
    }
}

int main(int argc, const char** argv){
    // jabil_match();
    // jabil_create_one_template();
    // jabil_test();

    float weak_threshold, strong_threshold;
    int num_features;
    bool create_templates;
    boost::program_options::options_description desc("Allowed options");
    desc.add_options()
        (
            "weak_threshold,w",
            boost::program_options::value<float>(&weak_threshold)->default_value(DEF_WEAK_THRESHOLD),
            "Weak threshold"
        )
        (
            "strong_threshold,s",
            boost::program_options::value<float>(&strong_threshold)->default_value(DEF_STRONG_THRESHOLD),
            "Strong threshold"
        )
        (
            "num_features,n",
            boost::program_options::value<int>(&num_features)->default_value(DEF_NUM_FEATURE),
            "Number of features"
        )
        (
            "create_template,c",
            boost::program_options::value<bool>(&create_templates)->default_value(false),
            "Create templates?"
        )
        (
            "testdir,t",
            boost::program_options::value<std::string>(),
            "Test directory"
        )
        ("help,h", "Print usage information");

    boost::program_options::positional_options_description pdesc;
    pdesc.add("testdir", 1);

    // parse options
    boost::program_options::variables_map vm;
    boost::program_options::store(boost::program_options::command_line_parser(argc, argv).options(desc).positional(pdesc).run(), vm);
    boost::program_options::notify(vm);

    if(vm.count("help"))
    {
        std::cout << desc << std::endl;
        // std::cout << cv::getBuildInformation() << std::endl;
        return 0;
    }

    std::string testdir = "VIKING";
    if(vm.count("testdir"))
    {
        testdir = vm["testdir"].as<std::string>();
    }

    std::cout << "Weak threshold: " << weak_threshold << std::endl;
    std::cout << "Strong threshold: " << strong_threshold << std::endl;
    std::cout << "Number of features: " << num_features << std::endl;
    std::cout << "Create templates? " << create_templates << std::endl;
    std::cout << "Test directory: " << testdir << std::endl;

    jabil_read_all_templates_and_match(testdir, weak_threshold, strong_threshold, num_features, create_templates);

    return 0;
}
