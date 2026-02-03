#pragma once

#include <cstdint>
#include <vector>

#include <opencv2/core.hpp>

namespace ecal::core {

struct PatchPointsOptions {
  // circle radius for perimeter sampling (pixel units)
  int radius = 3;

  // adaptive threshold: sign_eps = max(sign_eps_min, sign_eps_ratio * p99_abs)
  float sign_eps_ratio = 0.2f;
  float sign_eps_min = 1e-6f;

  // percentile for abs(piwe) scale (0..100)
  float abs_percentile = 99.0f;

  // optional: skip pixels with very tiny |piwe| to reduce compute
  // If 0, no skip. If >0, require |piwe(y,x)| >= center_abs_min.
  // NOTE: you were right this can kill true corner centers, so default 0.
  float center_abs_min = 0.0f;
};

struct PatchPointsDebug {
  double piwe_min = 0.0;
  double piwe_max = 0.0;
  float p_abs = 0.0f;    // percentile(abs(piwe))
  float sign_eps = 0.0f; // final used threshold
  int num_points = 0;
};

class PatchExtractor {
public:
  explicit PatchExtractor(PatchPointsOptions opt = PatchPointsOptions());

  // Extract patch_points from piwe (CV_32F).
  // Returns points in (x,y) coordinates (OpenCV convention).
  std::vector<cv::Point>
  extractPatchPoints(const cv::Mat &piwe_f32,
                     PatchPointsDebug *dbg = nullptr) const;

  // Draw points on a BGR image (in-place). If img is grayscale, caller should
  // convert first.
  static void drawPatchPoints(cv::Mat &bgr_u8,
                              const std::vector<cv::Point> &pts,
                              int zoom_factor = 1,
                              const cv::Scalar &color = cv::Scalar(0, 255, 0),
                              int radius_px = 1, int thickness = -1);

private:
  PatchPointsOptions opt_;
  std::vector<cv::Point> circle_offsets_;

  static std::vector<cv::Point> buildCirclePerimeterOffsets(int radius);

  static float percentileAbs(const cv::Mat &img_f32, float p);
};

} // namespace ecal::core
