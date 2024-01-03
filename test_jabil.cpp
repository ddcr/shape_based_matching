// #include "csv.hpp"
#include "utils.hpp"
#include "nms.hpp"

#include "common_structs.hpp"
#include "dao_wrapper.hpp"
#include <QDebug>

#include <memory>
#include <iostream>
#include <fstream>
#include <boost/program_options.hpp>
#include <assert.h>
#include <regex>
#include <experimental/filesystem>

using namespace std;
using namespace cv;

#define IMAGES_DBG 0

namespace fs = std::experimental::filesystem;
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
// float DEF_WEAK_THRESHOLD = 235.0f;
// float DEF_STRONG_THRESHOLD = 240.0f;
float DEF_WEAK_THRESHOLD = 100.0f;
float DEF_STRONG_THRESHOLD = 200.0f;

// Scale range of templates
float SCALE_RANGE_MIN = 0.9f;
float SCALE_RANGE_MAX = 1.1f;
float SCALE_RANGE_STEP = 0.1f;

// Detector threshold for detection
float DET_THRESHOLD = 90.0f;

void createLinemod2DTemplates(float weak_thresh, float strong_thresh, int num_feat, std::string template_dir="")
{
    std::string current_path = fs::current_path().string();
    std::string template_path = current_path + "/model_images";

    line2Dup::Detector detector(num_feat, {4, 8}, weak_thresh, strong_thresh);

    std::vector<ModelTag> modelTags = extractTagModelFiducialsFromDB();

    std::vector<string> class_ids;
    for (const auto& modelTag: modelTags)
    {
        cv::Mat3b modelImage = getImage(modelTag.modelFileName);
        fs::path modelFileName_p{modelTag.modelFileName};
        fs::path modelFileName_orig = modelFileName_p.stem();

        std::string class_id = to_string(modelTag.modelID);
        for (const auto& crop: modelTag.crops)
        {
            int tagFieldID = std::get<0>(crop);
            cv::Rect cropFid = std::get<1>(crop);
            cv::Mat cpModelImage = modelImage.clone();
            cv::Mat cropFidImage = cpModelImage(cropFid);

            // save the cropped image of the tag fiducial marker for further usage
            fs::path  extraExtension_p{"." + to_string(tagFieldID) + "."};
            fs::path modelFileName_new = modelFileName_orig;
            modelFileName_new += extraExtension_p;
            modelFileName_new.replace_extension(modelFileName_p.extension());
            fs::path modelFileNameFid_p = modelFileName_p.replace_filename(modelFileName_new);
            cv::imwrite(modelFileNameFid_p.string(), cropFidImage);

            // create scale/orientation copies
            shape_based_matching::shapeInfo_producer fid_shapes(cropFidImage, cv::Mat());
            fid_shapes.angle_range = {0, 270};
            fid_shapes.angle_step = 90;

            fid_shapes.scale_range = {SCALE_RANGE_MIN, SCALE_RANGE_MAX};
            fid_shapes.scale_step = SCALE_RANGE_STEP;
            fid_shapes.produce_infos();

            for (auto& info: fid_shapes.infos)
            {
                int templ_id = detector.addTemplate(
                    fid_shapes.src_of(info),
                    class_id,
                    fid_shapes.mask_of(info),
                    info.scale,
                    info.angle,
                    // modelTag.modelID,
                    // modelTag.modelFileName,
                    // modelImage.size(),
                    // tagFieldID,
                    modelFileNameFid_p.string()
                    // cropFid
                );
                if (templ_id == -1)
                    std::cout << "Could not create template with ID:" << templ_id << std::endl;
            }
        }
        class_ids.push_back(class_id);

        std::cout << "Writing template for model: " << modelTag.modelName << std::endl;
        // TODO: the folder '/path/to/model_images/' should be freely configured as a parameter.
        detector.writeClasses( template_path + "/%s.yaml.gz");
    }

    // save line2Dup detector settings
    cv::FileStorage fs(current_path + "/detector_linemod.yaml", cv::FileStorage::WRITE);
    detector.write(fs);
    fs << "templates_dir" << "model_images";
    fs << "classes" << class_ids;
}

// detectTagTypeLinemod
std::pair<int, int> detectTemplateLinemod(const std::string& filename, const cv::Mat& img, std::ofstream& outfile, const bool& img_dbg)
{
    Timer timer;
    line2Dup::Detector *detector = line2Dup::Detector::getInstance();

    // Extract all models
    std::vector<ModelTag> modelTags = extractTagModelFiducialsFromDB();

    std::vector<string> class_ids = detector->classIds();
    auto matches = detector->match(img, DET_THRESHOLD, class_ids);
    timer.record("MATCH");

    // (1) Use NMS to filter overlapping boxes of different scales but similar score
    std::vector<cv::Rect> boxes;
    std::vector<float> scores;
    std::vector<int> indices;
    for(auto match: matches)
    {
        cv::Rect box;
        box.x = match.x;
        box.y = match.y;
        auto templ = detector->getTemplates(match.class_id, match.template_id);

        box.width = templ[0].width;
        box.height = templ[0].height;
        boxes.push_back(box);
        scores.push_back(match.similarity);
    }
    cv_dnn::NMSBoxes(boxes, scores, 0, 0.5f, indices);
    timer.record("NMS");
    // cv_dnn::NMSBoxes(boxes, scores, 0, 0.8f, indices, 1.0, 2); // uses adaptive threshold

    // (2) Filter false positives, comparing detected ROIs with fiducial markers from DB
    // (2.1) retrieve images of fiducial markers
    int imatch = 0;
    cv::Mat img_show = img.clone();

    for (auto idx: indices)
    {
        auto match = matches[idx];
        auto templ = detector->getTemplates(match.class_id, match.template_id);
        int modelID = stoi(match.class_id);

        auto modelTag = std::find_if(std::begin(modelTags), std::end(modelTags),
                                    [modelID](const ModelTag &mt)
                                    {
                                        return mt.modelID == modelID;
                                    });
        if (modelTag == std::end(modelTags))
        {
            std::cout << "Model '" << match.class_id << "' non-existent" << std::endl;
            break;
        }

        if(img_dbg)
        {
            double hcorr = -1.0;

            // extract candidate ROI from img (convertion to grayscale if template matching is used)
            cv::Rect templ_roi = cv::Rect(match.x, match.y, templ[0].width, templ[0].height);
            cv::Mat img_roi = img(templ_roi).clone();
            cv::Mat1b img_roi_gray;
            cv::cvtColor(img_roi, img_roi_gray, cv::COLOR_BGR2GRAY);

            // TODO: In future avoid reading from disk
            // extract fiducial marker source image; rotate and crop to ROI size according
            // to template
            cv::Mat tagFieldImage = cv::imread(templ[0].fiducial_src, cv::IMREAD_GRAYSCALE);
            rotateScaleImage(tagFieldImage, templ[0].sscale, templ[0].orientation);

            cv::Mat tagFieldImageCropped = tagFieldImage(
                cv::Rect(templ[0].tl_x, templ[0].tl_y, templ[0].width, templ[0].height)
            ).clone();

            // draw ROI and features onto img_show
            int x = templ[0].width + match.x;
            int y = templ[0].height + match.y;
            int r = templ[0].width/2;

            cv::Vec3b randColor = {255, 0, 0};

            for(int i = 0; i < templ[0].features.size(); i++){
                auto feat = templ[0].features[i];
                cv::circle(img_show, {feat.x + match.x, feat.y + match.y}, 2, randColor, -1);
            }

            // Detected box in image
            cv::rectangle(img_show, {match.x, match.y}, {x, y}, randColor, 2);
            cv::putText(
                img_show,
                to_string(match.template_id), cv::Point(match.x+r-10, match.y-3),
                cv::FONT_HERSHEY_PLAIN, 1.0f, randColor, 2
            );

            // Text information
            std::stringstream sscale_t, similarity_t, hcorr_t;
            sscale_t.precision(2);
            similarity_t.precision(2);
            hcorr_t.precision(2);
            sscale_t << templ[0].sscale;
            similarity_t << match.similarity;
            hcorr_t << hcorr;
            std::vector<std::string> extraInfo = {
                "Box: " + to_string(match.template_id),
                "Scal/Orient: " + sscale_t.str() + ", " + to_string(int(templ[0].orientation)),
                "Sim: " + similarity_t.str() + ", Filter Corr: " + hcorr_t.str()
            };

            showIndividualMatchings(img_roi_gray, tagFieldImageCropped, match.similarity,
                                    modelTag->modelName,
                                    extraInfo, imatch);
        }
        imatch++;
    }
    timer.record("HCORR");

    if (img_dbg)
    {
        cv::namedWindow(filename);
        cv::moveWindow(filename, 680, 400);
        cv::imshow(filename, img_show);
        int key = cv::waitKey(0);
        if (key == 113)
        {
            exit(1);
        }
        cv::destroyAllWindows();
    }


    std::stringstream ss = timer.displayCSV({"MATCH", "NMS", "HCORR"}, filename);
    std::cout << ss.str();

    if(outfile.is_open())
    {
        outfile << ss.rdbuf();
    }
    return {0, 0};
}

void jabil_read_all_templates_and_match(std::string testdir, const bool& img_dbg)
{
    fs::path path_test_images = fs::canonical(
        "../../../inspection_images/2023-07-27/JabilCam-modelos/tag_candidate/" + testdir
    );

    std::vector<fs::path> filelist;
    for (auto const& dir_entry : fs::directory_iterator{ path_test_images })
    {
        if(fs::is_regular_file(dir_entry.path()))
        {
            filelist.push_back(dir_entry.path());
        }
    }

#if GRAD_DBG
    ofstream fgrad("gradients.csv", std::ios::app);
#endif

    // std::ofstream mycsv(testdir+"_timings.csv");
    std::ofstream mycsv;

    for(auto &f: filelist)
    {
        Timer timer_wall;

        cv::Mat img_orig = imread(f.string());
        assert(!img_orig.empty() && "check your img path");

        // compatibility wth line2Dup::computeResponseMaps()
        // make the img having 16*n width & height
        int stride = 16;
        int n = img_orig.rows/stride;
        int m = img_orig.cols/stride;
        cv::Rect roi(0, 0, stride*m , stride*n);
        cv::Mat img = img_orig(roi).clone();

        std::pair<int, int> matchingResult = detectTemplateLinemod(f.filename(), img, mycsv, img_dbg);
    }
}

int main(int argc, const char** argv)
{
    float weak_threshold, strong_threshold;
    int num_features;
    bool create_templates, create_templates_only, debug;
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
            "Create templates before processing images?"
        )
        (
            "create_template_only,k",
            boost::program_options::value<bool>(&create_templates_only)->default_value(false),
            "Create templates only?"
        )
        (
            "debug,v",
            boost::program_options::value<bool>(&debug)->default_value(false),
            "Debug with images"
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
    std::cout << "Debug (images): " << debug << std::endl;

    if(create_templates_only)
    {
        createLinemod2DTemplates(weak_threshold, strong_threshold, num_features);
    }
    else
    {
        if (create_templates)
        {
            createLinemod2DTemplates(weak_threshold, strong_threshold, num_features);
        }
        jabil_read_all_templates_and_match(testdir, debug);
    }

    return 0;
}
