#pragma once

#include <vector>

#include <opencv2/core.hpp>

namespace ecal::core {

struct CalibrationResult {
  bool success = false;
  cv::Mat camera_matrix;
  cv::Mat dist_coeffs;
  std::vector<cv::Mat> rvecs;
  std::vector<cv::Mat> tvecs;
  double reprojection_error = 0.0;
};

double
computeReprojectionError(const std::vector<std::vector<cv::Point3f>> &objpoints,
                         const std::vector<std::vector<cv::Point2f>> &imgpoints,
                         const cv::Mat &K, const cv::Mat &dist,
                         const std::vector<cv::Mat> &rvecs,
                         const std::vector<cv::Mat> &tvecs);

CalibrationResult
calibrateCheckerboard(const std::vector<std::vector<cv::Point3f>> &objpoints,
                      const std::vector<std::vector<cv::Point2f>> &imgpoints,
                      const cv::Size &image_size);

} // namespace ecal::core
