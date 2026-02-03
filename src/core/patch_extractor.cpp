#include "ecal_core/core/patch_extractor.hpp"

#include <algorithm>
#include <cmath>
#include <stdexcept>
#include <vector>

#include <opencv2/imgproc.hpp>

namespace ecal::core {

PatchExtractor::PatchExtractor(PatchPointsOptions opt) : opt_(opt) {
  if (opt_.radius <= 0) {
    throw std::runtime_error("PatchExtractor: radius must be > 0");
  }
  if (opt_.abs_percentile <= 0.0f || opt_.abs_percentile > 100.0f) {
    throw std::runtime_error(
        "PatchExtractor: abs_percentile must be in (0,100]");
  }
  if (opt_.sign_eps_ratio < 0.0f) {
    throw std::runtime_error("PatchExtractor: sign_eps_ratio must be >= 0");
  }
  if (opt_.sign_eps_min < 0.0f) {
    throw std::runtime_error("PatchExtractor: sign_eps_min must be >= 0");
  }

  circle_offsets_ = buildCirclePerimeterOffsets(opt_.radius);
  if (circle_offsets_.empty()) {
    throw std::runtime_error("PatchExtractor: circle perimeter offsets empty");
  }
}

std::vector<cv::Point> PatchExtractor::buildCirclePerimeterOffsets(int radius) {
  // Midpoint circle algorithm, 8-way symmetry, unique perimeter integer
  // offsets. We don't need angle ordering in Step 1.
  std::vector<cv::Point> pts;
  pts.reserve(static_cast<size_t>(8 * radius + 16));

  // Use a tiny bitmap-like uniqueness check by hashing offsets into int64
  auto key = [](int dx, int dy) -> int64_t {
    return (static_cast<int64_t>(dx) << 32) ^ static_cast<uint32_t>(dy);
  };
  std::vector<int64_t> seen;
  seen.reserve(pts.capacity());

  auto add_unique = [&](int dx, int dy) {
    const int64_t k = key(dx, dy);
    if (std::find(seen.begin(), seen.end(), k) == seen.end()) {
      seen.push_back(k);
      pts.emplace_back(dx, dy);
    }
  };

  int x = radius;
  int y = 0;
  int err = 1 - x;

  auto add_sym = [&](int xx, int yy) {
    // (dx,dy)
    add_unique(xx, yy);
    add_unique(yy, xx);
    add_unique(-yy, xx);
    add_unique(-xx, yy);
    add_unique(-xx, -yy);
    add_unique(-yy, -xx);
    add_unique(yy, -xx);
    add_unique(xx, -yy);
  };

  while (x >= y) {
    add_sym(x, y);
    ++y;
    if (err < 0) {
      err += 2 * y + 1;
    } else {
      --x;
      err += 2 * (y - x) + 1;
    }
  }

  return pts;
}

float PatchExtractor::percentileAbs(const cv::Mat &img_f32, float p) {
  if (img_f32.empty() || img_f32.type() != CV_32F) {
    throw std::runtime_error("percentileAbs: img must be non-empty CV_32F");
  }

  const int rows = img_f32.rows;
  const int cols = img_f32.cols;
  const int n = rows * cols;

  std::vector<float> v;
  v.reserve(static_cast<size_t>(n));

  for (int y = 0; y < rows; ++y) {
    const float *row = img_f32.ptr<float>(y);
    for (int x = 0; x < cols; ++x) {
      v.push_back(std::abs(row[x]));
    }
  }

  if (v.empty()) {
    return 0.0f;
  }

  // p in (0,100], convert to index in [0, n-1]
  const float pp = std::min(100.0f, std::max(0.0f, p));
  int k =
      static_cast<int>(std::floor((pp / 100.0f) * static_cast<float>(n - 1)));
  k = std::max(0, std::min(n - 1, k));

  std::nth_element(v.begin(), v.begin() + k, v.end());
  return v[static_cast<size_t>(k)];
}

std::vector<cv::Point>
PatchExtractor::extractPatchPoints(const cv::Mat &piwe_f32,
                                   PatchPointsDebug *dbg) const {
  if (piwe_f32.empty()) {
    if (dbg) {
      *dbg = PatchPointsDebug();
    }
    return {};
  }
  if (piwe_f32.type() != CV_32F) {
    throw std::runtime_error("extractPatchPoints: piwe must be CV_32F");
  }

  const int H = piwe_f32.rows;
  const int W = piwe_f32.cols;

  double mn = 0.0;
  double mx = 0.0;
  cv::minMaxLoc(piwe_f32, &mn, &mx);

  const float p_abs = percentileAbs(piwe_f32, opt_.abs_percentile);
  const float sign_eps =
      std::max(opt_.sign_eps_min, opt_.sign_eps_ratio * p_abs);

  std::vector<cv::Point> out;
  out.reserve(static_cast<size_t>(H * W / 50)); // rough

  const float center_min = opt_.center_abs_min;

  for (int y = 0; y < H; ++y) {
    const float *row = piwe_f32.ptr<float>(y);
    for (int x = 0; x < W; ++x) {
      const float v0 = row[x];
      if (center_min > 0.0f && std::abs(v0) < center_min) {
        continue;
      }

      bool has_pos = false;
      bool has_neg = false;

      for (const auto &off : circle_offsets_) {
        const int xx = x + off.x;
        const int yy = y + off.y;
        if (static_cast<unsigned>(xx) >= static_cast<unsigned>(W) ||
            static_cast<unsigned>(yy) >= static_cast<unsigned>(H)) {
          continue;
        }

        const float v = piwe_f32.at<float>(yy, xx);

        if (v > sign_eps) {
          has_pos = true;
        } else if (v < -sign_eps) {
          has_neg = true;
        }

        if (has_pos && has_neg) {
          break;
        }
      }

      if (has_pos && has_neg) {
        out.emplace_back(x, y); // (x,y)
      }
    }
  }

  if (dbg) {
    dbg->piwe_min = mn;
    dbg->piwe_max = mx;
    dbg->p_abs = p_abs;
    dbg->sign_eps = sign_eps;
    dbg->num_points = static_cast<int>(out.size());
  }

  return out;
}

void PatchExtractor::drawPatchPoints(cv::Mat &bgr_u8,
                                     const std::vector<cv::Point> &pts,
                                     int zoom_factor, const cv::Scalar &color,
                                     int radius_px, int thickness) {
  if (bgr_u8.empty()) {
    return;
  }
  if (zoom_factor <= 0) {
    zoom_factor = 1;
  }
  for (const auto &p : pts) {
    const cv::Point pp(p.x * zoom_factor, p.y * zoom_factor);
    cv::circle(bgr_u8, pp, radius_px, color, thickness, cv::LINE_AA);
  }
}

} // namespace ecal::core
