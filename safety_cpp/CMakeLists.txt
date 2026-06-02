cmake_minimum_required(VERSION 3.16)
project(safety_layer C CXX)

set(CMAKE_C_STANDARD   99)
set(CMAKE_CXX_STANDARD 17)

# ── 빌드 타입 기본값 ───────────────────────────────────────────
if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release)
endif()

# ── 컴파일 플래그 ──────────────────────────────────────────────
# -O2 + 안전 최적화 (UB sanitizer 포함 가능)
add_compile_options(
  -Wall -Wextra -Wpedantic
  $<$<CONFIG:Release>:-O2>
  $<$<CONFIG:Debug>:-O0 -g -fsanitize=address,undefined>
)
add_link_options(
  $<$<CONFIG:Debug>:-fsanitize=address,undefined>
)

# ── include 경로 ───────────────────────────────────────────────
include_directories(include)

# ════════════════════════════════════════════════════════════════
# 1. 공유 라이브러리 (Python cffi / ROS2 모두 이걸 링크)
# ════════════════════════════════════════════════════════════════
add_library(safety_core SHARED
  src/safety_layer.c
)
target_include_directories(safety_core PUBLIC include)
target_link_libraries(safety_core m)   # libm (math)

set_target_properties(safety_core PROPERTIES
  OUTPUT_NAME   "safety_core"
  VERSION       1.0.0
  SOVERSION     1
  # Python이 찾을 수 있도록 같은 디렉토리에 출력
  LIBRARY_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib
)

# ════════════════════════════════════════════════════════════════
# 2. 정적 라이브러리 (임베디드 / RTOS 배포용)
# ════════════════════════════════════════════════════════════════
add_library(safety_core_static STATIC
  src/safety_layer.c
)
target_include_directories(safety_core_static PUBLIC include)
target_link_libraries(safety_core_static m)

# ════════════════════════════════════════════════════════════════
# 3. 단위 테스트 (ctest)
# ════════════════════════════════════════════════════════════════
enable_testing()

add_executable(test_safety
  test/test_safety_layer.c
)
target_link_libraries(test_safety safety_core)
add_test(NAME SafetyLayerTest COMMAND test_safety)

add_executable(test_sync
  test/test_sync.c
)
target_link_libraries(test_sync safety_core)
add_test(NAME SyncTest COMMAND test_sync)

# ════════════════════════════════════════════════════════════════
# 4. ROS2 노드 (ament_cmake 환경에서만)
# ════════════════════════════════════════════════════════════════
if(DEFINED ENV{AMENT_PREFIX_PATH})
  find_package(ament_cmake QUIET)
  find_package(rclcpp      QUIET)
  find_package(sensor_msgs QUIET)
  find_package(std_msgs    QUIET)

  if(ament_cmake_FOUND AND rclcpp_FOUND)
    message(STATUS "ROS2 감지 — safety_node 빌드")

    add_executable(safety_node
      src/safety_ros2_node.cpp
    )
    target_link_libraries(safety_node
      safety_core
      rclcpp::rclcpp
      ${sensor_msgs_TARGETS}
      ${std_msgs_TARGETS}
    )
    install(TARGETS safety_node
      DESTINATION lib/${PROJECT_NAME})
    ament_package()
  else()
    message(STATUS "ROS2 미감지 — safety_node 스킵")
  endif()
endif()

# ════════════════════════════════════════════════════════════════
# 5. 설치 경로
# ════════════════════════════════════════════════════════════════
install(TARGETS safety_core safety_core_static
  LIBRARY DESTINATION lib
  ARCHIVE DESTINATION lib
)
install(DIRECTORY include/safety DESTINATION include)

# ── 빌드 후 Python 바인딩 자동 복사 ──────────────────────────
add_custom_command(TARGET safety_core POST_BUILD
  COMMAND ${CMAKE_COMMAND} -E copy
    $<TARGET_FILE:safety_core>
    ${CMAKE_SOURCE_DIR}/bindings/
  COMMENT "Copying libsafety_core → bindings/"
)
