#include "AlgorithmEstimation.hpp"

bool computeMatchesDistanceStatistics(const Matches& matches, float& meanDistance, float& stdDev)
{
    if (matches.empty())
        return false;
    
    std::vector<float> distances(matches.size());
    for (size_t i=0; i<matches.size(); i++)
        distances[i] = matches[i].distance;
    
    cv::Scalar mean, dev;
    cv::meanStdDev(distances, mean, dev);
    
    meanDistance = static_cast<float>(mean.val[0]);
    stdDev       = static_cast<float>(dev.val[0]);
    
    return false;
}

void ratioTest(const std::vector<Matches>& knMatches, float maxRatio, Matches& goodMatches)
{
    goodMatches.clear();
    
    for (size_t i=0; i< knMatches.size(); i++)
    {
        const cv::DMatch& best = knMatches[i][0];
        const cv::DMatch& good = knMatches[i][1];
        
        assert(best.distance <= good.distance);
        float ratio = (best.distance / good.distance);
        
        if (ratio <= maxRatio)
        {
            goodMatches.push_back(best);
        }
    }
}

cv::Scalar computeReprojectionError(const Keypoints& source, const Keypoints& query, const Matches& matches, const cv::Mat& homography);


bool performEstimation
(
 const FeatureAlgorithm& alg,
 const ImageTransformation& transformation,
 const cv::Mat& sourceImage,
 std::vector<FrameMatchingStatistics>& stat
)
{
    Keypoints   sourceKp;
    Descriptors sourceDesc;
    
    cv::Mat gray;
    if (sourceImage.channels() == 3)
        cv::cvtColor(sourceImage, gray, CV_BGR2GRAY);
    else if (sourceImage.channels() == 4)
        cv::cvtColor(sourceImage, gray, CV_BGRA2GRAY);
    else if(sourceImage.channels() == 1)
        gray = sourceImage;
    
    if (!alg.extractFeatures(gray, sourceKp, sourceDesc))
        return false;
    
    std::vector<float> x = transformation.getX();
    stat.resize(x.size());
    
    const int count = x.size();
    
    Keypoints   resKpReal;
    Descriptors resDesc;
    Matches     matches;
    
    // To convert ticks to milliseconds
    const double toMsMul = 1000. / cv::getTickFrequency();
    
    #pragma omp parallel for private(resKpReal, resDesc, matches) schedule(dynamic, 5)
    for (int i = 0; i < count; i++)
    {
        float       arg = x[i];
        FrameMatchingStatistics& s = stat[i];
        
        cv::Mat     transformedImage;
        transformation.transform(arg, gray, transformedImage);

        cv::Mat expectedHomography = transformation.getHomography(arg, gray);
                
        int64 start = cv::getTickCount();
        
        alg.extractFeatures(transformedImage, resKpReal, resDesc);
        
        // Initialize required fields
        s.isValid        = resKpReal.size() > 0;
        s.argumentValue  = arg;
        
        if (!s.isValid)
            continue;
        
        if (alg.knMatchSupported)
        {
            std::vector<Matches> knMatches;
            alg.matchFeatures(sourceDesc, resDesc, 2, knMatches);
            ratioTest(knMatches, 0.75, matches);
            
            // Compute percent of false matches that were rejected by ratio test
            s.ratioTestFalseLevel = (float)(knMatches.size() - matches.size()) / (float) knMatches.size();
        }
        else
        {
            alg.matchFeatures(sourceDesc, resDesc, matches);
        }
        
        int64 end = cv::getTickCount();
        
        Matches correctMatches;
        cv::Mat homography;
        bool homographyFound = ImageTransformation::findHomography(sourceKp, resKpReal, matches, correctMatches, homography);
        //bool homographyFound = ImageTransformation::findHomographySubPix(sourceKp, gray, resKpReal, transformedImage, matches, correctMatches, homography);

        // Some simple stat:
        s.isValid        = homographyFound;
        s.totalKeypoints = resKpReal.size();
        s.consumedTimeMs = (end - start) * toMsMul;
        
        // Compute overall percent of matched keypoints
        s.percentOfMatches      = (float) matches.size() / (float)(std::min(sourceKp.size(), resKpReal.size()));
        s.correctMatchesPercent = (float) correctMatches.size() / (float)matches.size();
        
        // Compute matching statistics
        if (homographyFound)
        {
            cv::Mat r = expectedHomography * homography.inv();
            float error = cv::norm(cv::Mat::eye(3,3, CV_64FC1) - r, cv::NORM_INF);

            computeMatchesDistanceStatistics(correctMatches, s.meanDistance, s.stdDevDistance);
            s.reprojectionError = computeReprojectionError(sourceKp, resKpReal, correctMatches, homography);
            s.homographyError = std::min(error, 1.0f);

            if (0 && error >= 1)
            {
                std::cout << "H expected:" << expectedHomography << std::endl;
                std::cout << "H actual:"   << homography << std::endl;
                std::cout << "H error:"    << error << std::endl;
                std::cout << "R error:"    << s.reprojectionError(0) << ";" 
                                           << s.reprojectionError(1) << ";" 
                                           << s.reprojectionError(2) << ";" 
                                           << s.reprojectionError(3) << std::endl;
                
                cv::Mat matchesImg;
                cv::drawMatches(transformedImage,
                                resKpReal,
                                gray,
                                sourceKp,
                                correctMatches,
                                matchesImg,
                                cv::Scalar::all(-1),
                                cv::Scalar::all(-1),
                                std::vector<char>(),
                                cv::DrawMatchesFlags::NOT_DRAW_SINGLE_POINTS);
                
                cv::imshow("Matches", matchesImg);
                cv::waitKey(-1);
            }
        }
    }
    
    return true;
}

bool performComparison(const FeatureAlgorithm& alg,
  const cv::Mat& sourceImage,
  const cv::Mat& testImage,
  SingleRunStatistics& stat,
  std::string& imageName)
{
    Keypoints   sourceKp;
    Descriptors sourceDesc;
    
    cv::Mat gray;
    if (sourceImage.channels() == 3)
        cv::cvtColor(sourceImage, gray, CV_BGR2GRAY);
    else if (sourceImage.channels() == 4)
        cv::cvtColor(sourceImage, gray, CV_BGRA2GRAY);
    else if(sourceImage.channels() == 1)
        gray = sourceImage;
    
    if (!alg.extractFeatures(gray, sourceKp, sourceDesc))
        return false;
    
    int statIdx = stat.size();
    stat.resize(statIdx + 1);
    
    Keypoints   resKpReal;
    Descriptors resDesc;
    Matches     matches;
    
    // To convert ticks to milliseconds
    const double toMsMul = 1000. / cv::getTickFrequency();
    
    FrameMatchingStatistics& s = stat[statIdx];
    
    cv::Mat transformedImage = testImage;
            
    int64 start = cv::getTickCount();
    
    alg.extractFeatures(transformedImage, resKpReal, resDesc);
    
    // Initialize required fields
    s.isValid        = resKpReal.size() > 0;
    s.argumentValue = 0;
    
    if (!s.isValid) {
        return false;
    }
    if (alg.knMatchSupported)
    {
        std::vector<Matches> knMatches;
        alg.matchFeatures(sourceDesc, resDesc, 2, knMatches);
        ratioTest(knMatches, 0.75, matches);
        
        // Compute percent of false matches that were rejected by ratio test
        s.ratioTestFalseLevel = (float)(knMatches.size() - matches.size()) / (float) knMatches.size();
    }
    else
    {
        alg.matchFeatures(sourceDesc, resDesc, matches);
    }
    
    int64 end = cv::getTickCount();
    
    Matches correctMatches;
    cv::Mat homography;
    bool homographyFound = ImageTransformation::findHomography(sourceKp, resKpReal, matches, correctMatches, homography);
    //bool homographyFound = ImageTransformation::findHomographySubPix(sourceKp, gray, resKpReal, transformedImage, matches, correctMatches, homography);

    // Some simple stat:
    s.isValid        = homographyFound;
    s.totalKeypoints = resKpReal.size();
    s.consumedTimeMs = (end - start) * toMsMul;
    
    // Compute overall percent of matched keypoints
    s.percentOfMatches      = (float) matches.size() / (float)(std::min(sourceKp.size(), resKpReal.size()));
    s.correctMatchesPercent = (float) correctMatches.size() / (float)matches.size();
    
    // Compute matching statistics
	computeMatchesDistanceStatistics(correctMatches, s.meanDistance, s.stdDevDistance);
	if (homographyFound)
	{
		s.reprojectionError = computeReprojectionError(sourceKp, resKpReal, correctMatches, homography);
	}

	if (1)
	{
		cv::Mat matchesImg;
		cv::drawMatches(transformedImage,
						resKpReal,
						gray,
						sourceKp,
						correctMatches,
						matchesImg,
						cv::Scalar::all(-1),
						cv::Scalar::all(-1),
						std::vector<char>(),
						cv::DrawMatchesFlags::NOT_DRAW_SINGLE_POINTS);



		std::string matchName = std::string("Matches/");
		cv::imwrite(matchName + imageName + "_" + alg.name + ".jpg", matchesImg);
	}
    
    return true;
}


cv::Scalar computeReprojectionError(const Keypoints& source, const Keypoints& query, const Matches& matches, const cv::Mat& homography)
{
    assert(matches.size() > 0);

    const int pointsCount = matches.size();
    std::vector<cv::Point2f> srcPoints, dstPoints;
    std::vector<float> distances;

    for (int i = 0; i < pointsCount; i++)
    {
        srcPoints.push_back(source[matches[i].trainIdx].pt);
        dstPoints.push_back(query[matches[i].queryIdx].pt);
    }

    cv::perspectiveTransform(dstPoints, dstPoints, homography.inv());
    for (int i = 0; i < pointsCount; i++)
    {
        const cv::Point2f& src = srcPoints[i];
        const cv::Point2f& dst = dstPoints[i];

        cv::Point2f v = src - dst;
        distances.push_back(sqrtf(v.dot(v)));
    }

    
    cv::Scalar mean, dev;
    cv::meanStdDev(distances, mean, dev);

    cv::Scalar result;
    result(0) = mean(0);
    result(1) = dev(0);
    result(2) = *std::max_element(distances.begin(), distances.end());
    result(3) = *std::min_element(distances.begin(), distances.end());
    return result;
}
