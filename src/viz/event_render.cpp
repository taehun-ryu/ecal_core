#include "ecal/viz/event_render.hpp"

#include <opencv2/core.hpp>

#include "ecal/core/simple_event.hpp"

namespace ecal::viz {

cv::Mat eventsToImage(const std::vector<ecal::core::TimedEventNs> &events,
                      int width, int height, int zoom_factor,
                      bool non_max_suppression, int add_value) {
  if (zoom_factor < 1) {
    zoom_factor = 1;
  }

  cv::Mat image =
      cv::Mat::zeros(height * zoom_factor, width * zoom_factor, CV_8UC3);

  for (const auto &e : events) {
    // Match ROS1-style visualization: ON -> Blue, OFF -> Red
    const int channel = (e.polarity == 1) ? 0 : 2;

    int u = static_cast<int>(e.x);
    int v = static_cast<int>(e.y);

    if (u < 0 || v < 0 || u >= width || v >= height) {
      continue;
    }

    u *= zoom_factor;
    v *= zoom_factor;

    for (int du = 0; du < zoom_factor; ++du) {
      for (int dv = 0; dv < zoom_factor; ++dv) {
        auto &pix = image.at<cv::Vec3b>(v + dv, u + du)[channel];
        if (pix > 255 - add_value) {
          pix = 255;
        } else {
          pix = static_cast<unsigned char>(pix + add_value);
        }
      }
    }
  }

  if (non_max_suppression) {
    for (int u = 0; u < width * zoom_factor; ++u) {
      for (int v = 0; v < height * zoom_factor; ++v) {
        for (int c = 0; c < 3; ++c) {
          if (image.at<cv::Vec3b>(v, u)[c] < 80) {
            image.at<cv::Vec3b>(v, u)[c] = 0;
          }
        }
      }
    }
  }

  return image;
}

} // namespace ecal::viz
