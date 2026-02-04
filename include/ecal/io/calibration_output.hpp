#pragma once

#include <string>

#include "ecal/viz/calibration_viz.hpp"

namespace ecal::io {

void ensureCalibrationOutputDirs(const std::string &out_dir);

void saveCalibrationOutputs(const std::string &out_dir, size_t window_idx,
                            const ecal::viz::WindowVis &vis);

void saveCalibrationYaml(const std::string &out_dir, size_t used_windows,
                         size_t total_windows, int board_rows, int board_cols,
                         float square_size, const cv::Mat &K,
                         const cv::Mat &dist, double reproj_error);

} // namespace ecal::io
