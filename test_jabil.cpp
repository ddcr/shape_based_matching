#include "line2Dup.h"
#include "utils.hpp"
#include <memory>
#include <iostream>
#include <boost/program_options.hpp>
#include <assert.h>
#include <regex>
#include <experimental/filesystem>
using namespace std;
using namespace cv;

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

float SCALE_RANGE_MIN = 0.9f;
float SCALE_RANGE_MAX = 1.1f;

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
