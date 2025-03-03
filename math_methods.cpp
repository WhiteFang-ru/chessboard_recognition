//#include <array>
#include <stdexcept>
#include <opencv2/imgproc.hpp>
#include <opencv2/opencv.hpp>
#include <opencv2/core.hpp>
#include <opencv2/highgui.hpp>
using namespace cv;


namespace Math {
    double median_grayscale(cv::Mat image) {
        if (image.empty())
        {
            throw std::logic_error("empty image");
            return 0;
        }
        // COMPUTE HISTOGRAM OF SINGLE CHANNEL MATRIX
        const int numBins = 4096; // 2^12 bins
        float range[] = {0, numBins};
        const float *histRange = {range};
        bool uniform = true;
        bool accumulate = false;
        cv::Mat hist;
        const int channels = 0;
        cv::calcHist(&image, 1, &channels, cv::Mat(), hist, 1, &numBins, &histRange, uniform, accumulate);

        // COMPUTE CUMULATIVE DISTRIBUTION FUNCTION (CDF)
        cv::Mat cdf;
        hist.copyTo(cdf);
        for (int i = 1; i < numBins; i++)
        {
            cdf.at<float>(i) += cdf.at<float>(i - 1);
        }
        cdf /= image.total();

        // COMPUTE MEDIAN
        double medianVal;
        for (int i = 0; i < numBins; i++)
        {
            if (cdf.at<float>(i) >= 0.5)
            {
                medianVal = i;
                break;
            }
        }
        return medianVal / numBins;
    }

    cv::Vec2d calculate_mean(std::vector<cv::Vec2d>& lines) {
        cv::Vec2d sum(0.0, 0.0);

        for (const auto& vec : lines) {
            sum += vec;
        }

        return sum / static_cast<double>(lines.size());
    }
}
