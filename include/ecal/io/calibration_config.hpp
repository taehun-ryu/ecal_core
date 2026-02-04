#pragma once

#include <cstddef>
#include <string>

namespace ecal::io {

struct CalibrationConfig {
  // input/output
  std::string h5_path;
  std::string out_dir;

  // sensor size
  int width = -1;
  int height = -1;

  // windowing
  double window_sec = 0.05;
  size_t min_events = 6000;
  size_t max_events = 20000;
  double min_window_sec = 0.02;
  double max_window_sec = 0.12;

  int expected_corners = -1; // if < 0, auto = board_w * board_h
  int board_w = -1;
  int board_h = -1;
  float square_size = 0.0f;

  // visualization
  int viz_zoom = 2; // visualization only; calibration uses original size

  // patch extraction
  int pp_radius = 5;

  // cm iwe
  bool use_variance = true; // true=L2(variance), false=IWE

  // calibration
  int calib_max_iter = 50;
  bool calib_fix_k3plus = true;
  bool calib_use_intrinsic_guess = true;
  int calib_B = 50;   // number of views per calibration run
  int calib_R = 5;    // number of bootstrap runs
};

bool loadCalibrationConfig(const std::string &path, CalibrationConfig &cfg,
                         std::string *err = nullptr);

} // namespace ecal::io
