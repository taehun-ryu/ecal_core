#include "ecal/io/calibration_output.hpp"

#include <filesystem>
#include <fstream>

#include <opencv2/core.hpp>
#include <opencv2/imgcodecs.hpp>

namespace ecal::io {

static std::string zeroPad(size_t v, int width) {
  std::string s = std::to_string(v);
  if (static_cast<int>(s.size()) >= width) {
    return s;
  }
  return std::string(static_cast<size_t>(width - s.size()), '0') + s;
}

void ensureCalibrationOutputDirs(const std::string &out_dir) {
  if (out_dir.empty()) {
    return;
  }
  std::filesystem::create_directories(out_dir);
  std::filesystem::create_directories(std::filesystem::path(out_dir) / "raw");
  std::filesystem::create_directories(std::filesystem::path(out_dir) / "iwe");
  std::filesystem::create_directories(std::filesystem::path(out_dir) / "piwe");
  std::filesystem::create_directories(std::filesystem::path(out_dir) /
                                      "corner_detection");
}

void saveCalibrationOutputs(const std::string &out_dir, size_t window_idx,
                            const ecal::viz::WindowVis &vis) {
  if (out_dir.empty()) {
    return;
  }
  const std::string idx_str = zeroPad(window_idx, 6);
  const std::string raw_path = (std::filesystem::path(out_dir) / "raw" /
                                ("raw_events_" + idx_str + ".png"))
                                   .string();
  const std::string iwe_path =
      (std::filesystem::path(out_dir) / "iwe" / ("iwe_" + idx_str + ".png"))
          .string();
  const std::string piwe_path =
      (std::filesystem::path(out_dir) / "piwe" / ("piwe_" + idx_str + ".png"))
          .string();
  const std::string corner_path =
      (std::filesystem::path(out_dir) / "corner_detection" /
       ("corners_" + idx_str + ".png"))
          .string();

  if (!vis.raw_vis.empty()) {
    cv::imwrite(raw_path, vis.raw_vis);
  }
  if (!vis.piwe_vis.empty()) {
    cv::imwrite(piwe_path, vis.piwe_vis);
  }
  if (!vis.iwe_vis.empty()) {
    cv::imwrite(iwe_path, vis.iwe_vis);
  }
  if (!vis.corners_vis.empty()) {
    cv::imwrite(corner_path, vis.corners_vis);
  }
}

static void writeMatYaml(std::ofstream &ofs, const std::string &key,
                         const cv::Mat &m) {
  ofs << key << ":\n";
  if (m.empty()) {
    ofs << "  []\n";
    return;
  }
  for (int r = 0; r < m.rows; ++r) {
    ofs << "  - [";
    for (int c = 0; c < m.cols; ++c) {
      const double v = m.at<double>(r, c);
      ofs << v;
      if (c + 1 < m.cols)
        ofs << ", ";
    }
    ofs << "]\n";
  }
}

void saveCalibrationYaml(const std::string &out_dir, size_t used_windows,
                         size_t total_windows, int board_rows, int board_cols,
                         float square_size, const cv::Mat &K,
                         const cv::Mat &dist, double reproj_error) {
  if (out_dir.empty()) {
    return;
  }
  const std::string path =
      (std::filesystem::path(out_dir) / "calibration.yaml").string();
  std::ofstream ofs(path);
  if (!ofs) {
    return;
  }

  ofs << "used_windows: " << used_windows << "\n";
  ofs << "total_windows: " << total_windows << "\n";
  ofs << "checkerboard:\n";
  ofs << "  rows: " << board_rows << "\n";
  ofs << "  cols: " << board_cols << "\n";
  ofs << "  square_size: " << square_size << "\n";
  ofs << "reprojection_error: " << reproj_error << "\n";

  cv::Mat Kd, distd;
  K.convertTo(Kd, CV_64F);
  dist.convertTo(distd, CV_64F);
  writeMatYaml(ofs, "camera_matrix", Kd);

  ofs << "dist_coeffs: [";
  if (!distd.empty()) {
    for (int i = 0; i < distd.total(); ++i) {
      ofs << distd.at<double>(static_cast<int>(i));
      if (i + 1 < distd.total())
        ofs << ", ";
    }
  }
  ofs << "]\n";
}

} // namespace ecal::io
