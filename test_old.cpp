#include "line2Dup.h"
#include "utils.hpp"
#include <string>
#include <boost/program_options.hpp>
#include <assert.h>
#include <regex>
#include <experimental/filesystem>

static std::string PREFIX_PATH = "/home/ivision/jabil_tag_reader/dev_area/jabil_dev_phase4";

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
                    std::to_string(int(round(match.similarity))),
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

void test_preprocess(std::string testdir, bool clahe=true)
{
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

        cv::Mat img_orig = imread(f.string());
        assert(!img_orig.empty() && "check your img path");

        // compatibility wth line2Dup::computeResponseMaps()
        // make the img having 16*n width & height
        int stride = 16;
        int n = img_orig.rows/stride;
        int m = img_orig.cols/stride;
        Rect roi(0, 0, stride*m , stride*n);
        cv::Mat img = img_orig(roi).clone();
        cv::resize(img, img, cv::Size(), 0.5f, 0.5f);

        cv::Mat1b img_gray;
        cv::cvtColor(img, img_gray, cv::COLOR_BGR2GRAY);

        cv::Mat1b img_cdf, concatenated;
        if (clahe)
        {
            cv::Ptr<cv::CLAHE> clahe = cv::createCLAHE(40.0f, cv::Size(8,8));
            clahe->apply(img_gray, img_cdf);
        }
        else
        {
            cv::equalizeHist(img_gray, img_cdf);
        }

        cv::hconcat(img_gray, img_cdf, concatenated);

        std::string windowLabel = f.filename();
        cv::namedWindow(windowLabel, WINDOW_AUTOSIZE);
        cv::moveWindow(windowLabel, 40, 50);
        cv::imshow(windowLabel, concatenated);
        int key = cv::waitKey(0);
        if (key == 113)
        {
            break;
        }
        cv::destroyAllWindows();

        timer_wall.out("File processing");
    }
}

int main(int argc, const char** argv){
    // jabil_test1();
    // jabil_match();
    // jabil_create_one_template();
    boost::program_options::options_description desc("Allowed options");

    desc.add_options()
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

    std::string testdir = "EXAR";
    if(vm.count("testdir"))
    {
        testdir = vm["testdir"].as<std::string>();
    }

    test_preprocess(testdir);

    return 0;
}
