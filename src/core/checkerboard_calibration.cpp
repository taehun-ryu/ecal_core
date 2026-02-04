#include "ecal/core/checkerboard_calibration.hpp"

#include <opencv2/calib3d.hpp>

namespace ecal::core {

double
computeReprojectionError(const std::vector<std::vector<cv::Point3f>> &objpoints,
                         const std::vector<std::vector<cv::Point2f>> &imgpoints,
                         const cv::Mat &K, const cv::Mat &dist,
                         const std::vector<cv::Mat> &rvecs,
                         const std::vector<cv::Mat> &tvecs) {
  double total_err = 0.0;
  size_t total_pts = 0;
  std::vector<cv::Point2f> proj;
  for (size_t i = 0; i < objpoints.size(); ++i) {
    cv::projectPoints(objpoints[i], rvecs[i], tvecs[i], K, dist, proj);
    double err = 0.0;
    for (size_t j = 0; j < proj.size(); ++j) {
      const cv::Point2f d = proj[j] - imgpoints[i][j];
      err += std::sqrt(d.dot(d));
    }
    total_err += err;
    total_pts += proj.size();
  }
  return (total_pts > 0) ? (total_err / total_pts) : 0.0;
}

CalibrationResult
calibrateCheckerboard(const std::vector<std::vector<cv::Point3f>> &objpoints,
                      const std::vector<std::vector<cv::Point2f>> &imgpoints,
                      const cv::Size &image_size) {
  CalibrationResult out;
  if (objpoints.empty() || imgpoints.empty()) {
    return out;
  }
  cv::Mat K, dist;
  std::vector<cv::Mat> rvecs, tvecs;
  const int flags = 0;
  const double rms = cv::calibrateCamera(objpoints, imgpoints, image_size, K,
                                         dist, rvecs, tvecs, flags);
  out.success = std::isfinite(rms) && rms > 0.0;
  out.camera_matrix = K;
  out.dist_coeffs = dist;
  out.rvecs = rvecs;
  out.tvecs = tvecs;
  out.reprojection_error =
      computeReprojectionError(objpoints, imgpoints, K, dist, rvecs, tvecs);
  return out;
}

} // namespace ecal::core
