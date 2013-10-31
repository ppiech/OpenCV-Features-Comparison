
#include "ImageTransformation.hpp"
#include "AlgorithmEstimation.hpp"

#include <opencv2/opencv.hpp>
#include <opencv2/nonfree/features2d.hpp>
#include <algorithm>
#include <numeric>
#include <fstream>


const bool USE_VERBOSE_TRANSFORMATIONS = false;

int main(int argc, const char* argv[])
{
    std::vector<FeatureAlgorithm>              algorithms;
    std::vector<cv::Ptr<ImageTransformation> > transformations;

    bool useCrossCheck = true;

    // Initialize list of algorithm tuples:
       
//    algorithms.push_back(FeatureAlgorithm("KAZE",
//        new cv::KAZE(),
//        new cv::FlannBasedMatcher()));
//
//    algorithms.push_back(FeatureAlgorithm("BRISK",
//        new cv::BRISK(60,4),
//        new cv::BFMatcher(cv::NORM_HAMMING, useCrossCheck)));
//
//    algorithms.push_back(FeatureAlgorithm("ORB",
//        new cv::ORB(),
//        new cv::BFMatcher(cv::NORM_HAMMING, useCrossCheck)));
    
//    algorithms.push_back(FeatureAlgorithm("FREAK",
//        new cv::SurfFeatureDetector(),
//        new cv::FREAK(),
//        new cv::BFMatcher(cv::NORM_HAMMING, useCrossCheck)));

    
    // algorithms.push_back(FeatureAlgorithm("SURF+BRISK",
    //     new cv::SurfFeatureDetector(),
    //     new cv::BriskDescriptorExtractor(),
    //     new cv::BFMatcher(cv::NORM_HAMMING, useCrossCheck)));

    algorithms.push_back(FeatureAlgorithm("SURF_BF",
        new cv::SurfFeatureDetector(),
        new cv::SurfDescriptorExtractor(),
        new cv::BFMatcher(cv::NORM_L2, useCrossCheck)));

    FeatureAlgorithm alg =  FeatureAlgorithm("SURF_FLANN_KNN",
    		        new cv::SurfFeatureDetector(),
    		        new cv::SurfDescriptorExtractor(),
    		        new cv::FlannBasedMatcher());
    alg.knMatchSupported = true;
    algorithms.push_back(alg);

//    algorithms.push_back(FeatureAlgorithm("ORB_FREAK-normalized",
//        new cv::OrbFeatureDetector(),
//        new cv::FREAK(),
//        new cv::BFMatcher(cv::NORM_HAMMING, useCrossCheck)));
//
//    algorithms.push_back(FeatureAlgorithm("FREAK-normalized",
//        new cv::SurfFeatureDetector(),
//        new cv::FREAK(),
//        new cv::BFMatcher(cv::NORM_HAMMING, useCrossCheck)));
//
//
//    algorithms.push_back(FeatureAlgorithm("FAST_BRIEF",
//        new cv::FastFeatureDetector(50),
//        new cv::BriefDescriptorExtractor(),
//        new cv::BFMatcher(cv::NORM_HAMMING, useCrossCheck)));

//    algorithms.push_back(FeatureAlgorithm("SIFT_BF",
//        new cv::SiftFeatureDetector(),
//        new cv::SiftDescriptorExtractor(),
//        new cv::BFMatcher(cv::NORM_L2, useCrossCheck)));
//
//    algorithms.push_back(FeatureAlgorithm("SIFT_FLANN",
//        new cv::SiftFeatureDetector(),
//        new cv::SiftDescriptorExtractor(),
//        new cv::FlannBasedMatcher()));


    

    /**/

    // Initialize list of used transformations:
    if (USE_VERBOSE_TRANSFORMATIONS)
    {
        transformations.push_back(new GaussianBlurTransform(9));
        transformations.push_back(new BrightnessImageTransform(-127, +127,1));
        transformations.push_back(new ImageRotationTransformation(0, 360, 1, cv::Point2f(0.5f,0.5f)));
        transformations.push_back(new ImageScalingTransformation(0.25f, 2.0f, 0.01f));
    }
    else
    {
        transformations.push_back(new GaussianBlurTransform(9));
        transformations.push_back(new ImageRotationTransformation(0, 360, 10, cv::Point2f(0.5f,0.5f)));
        transformations.push_back(new ImageScalingTransformation(0.25f, 2.0f, 0.1f));
        transformations.push_back(new BrightnessImageTransform(-127, +127,10));
    }

    if (argc < 3)
    {
        std::cout << "At least two input image should be passed" << std::endl;
    }

    std::string refImagePath(argv[1]);
    cv::Mat refImage = cv::imread(refImagePath);    
    CollectedStatistics fullStat;
    for (int imageIndex = 2; imageIndex < argc; imageIndex++)
    {
        std::string testImagePath(argv[imageIndex]);
        cv::Mat testImage = cv::imread(testImagePath);


        if (testImage.empty())
        {
            std::cout << "Cannot read image from " << testImagePath << std::endl;
        }

        for (size_t algIndex = 0; algIndex < algorithms.size(); algIndex++)
        {
            const FeatureAlgorithm& alg   = algorithms[algIndex];

            std::cout << "Testing " << alg.name << "...";

            performComparison(alg, testImage.clone(), refImage.clone(), fullStat.getStatistics(alg.name, testImagePath), testImagePath);

            std::cout << "done." << std::endl;
        }
        /**/
    }
    fullStat.printAverage(std::cout, StatisticsElementHomographyError);

    std::ofstream performanceStr("Performance.txt");
    fullStat.printPerformanceStatistics(performanceStr);

    std::ofstream matchingRatioStr("MatchingRatio.txt");
    fullStat.printStatistics(matchingRatioStr,  StatisticsElementMatchingRatio);

    std::ofstream percentOfMatchesStr("PercentOfMatches.txt") ;
    fullStat.printStatistics(percentOfMatchesStr, StatisticsElementPercentOfMatches);

    std::ofstream percentOfCorrectMatchesStr("PercentOfCorrectMatches.txt");
    fullStat.printStatistics(percentOfCorrectMatchesStr, StatisticsElementPercentOfCorrectMatches);

    std::ofstream meanDistanceStr("MeanDistance.txt");
    fullStat.printStatistics(meanDistanceStr, StatisticsElementMeanDistance);

    std::ofstream homographyErrorStr("HomographyError.txt");
    fullStat.printStatistics(homographyErrorStr, StatisticsElementHomographyError);

    return 0;
}

