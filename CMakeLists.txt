cmake_minimum_required(VERSION 3.8)
project(ecal_core)

if(CMAKE_COMPILER_IS_GNUCXX OR CMAKE_CXX_COMPILER_ID MATCHES "Clang")
  add_compile_options(-Wall -Wextra -Wpedantic)
endif()

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

find_package(ament_cmake REQUIRED)

find_package(OpenCV REQUIRED)
find_package(Ceres REQUIRED)

# ---------------------------
# Library
# ---------------------------
add_library(ecal_core
  src/core/iwe_gaussian.cpp
  src/core/cm_velocity_2d.cpp
  src/core/cm_tracker_2d.cpp
  src/viz/event_render.cpp
)

target_include_directories(ecal_core PUBLIC
  $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
  $<INSTALL_INTERFACE:include>
)

target_link_libraries(ecal_core
  ${OpenCV_LIBS}
  Ceres::ceres
)

# If you use OpenMP in core later, add it explicitly (optional)
# find_package(OpenMP)
# if(OpenMP_CXX_FOUND)
#   target_link_libraries(ecal_core OpenMP::OpenMP_CXX)
# endif()

# Export include dirs + deps so downstream packages can link cleanly
ament_export_include_directories(include)
ament_export_libraries(ecal_core)
ament_target_dependencies(ecal_core
  OpenCV
  Ceres
)
target_link_libraries(ecal_core
  Ceres::ceres
  ${OpenCV_LIBS}
)

# ---------------------------
# Install
# ---------------------------
install(DIRECTORY include/
  DESTINATION include
)

install(TARGETS ecal_core
  EXPORT export_${PROJECT_NAME}
  ARCHIVE DESTINATION lib
  LIBRARY DESTINATION lib
  RUNTIME DESTINATION bin
)

ament_export_targets(export_${PROJECT_NAME} HAS_LIBRARY_TARGET)

ament_package()