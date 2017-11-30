#include "image_processor.h"

#include <algorithm>

#include <GLES2/gl2.h>
#include <EGL/egl.h>

#include <opencv2/core.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/core/ocl.hpp>

#include "common.hpp"

enum DisplayMode {
  DISP_MODE_RAW = 0,
  DISP_MODE_THRESH = 1,
  DISP_MODE_TARGETS = 2,
  DISP_MODE_TARGETS_PLUS = 3
};

enum TargetType {
  TARGET_NONE = 0,
  TARGET_PEG = 1,
  TARGET_BOILER = 2
};

//typedef contourPair_t std::pair<std::vector<cv::Point>,std::vector<cv::Point>>;

struct TargetInfo {
  double centroid_x;
  double centroid_y;
  double width;
  double height;
  std::vector<cv::Point> points;
};

inline void GetTargetInfo(TargetInfo& target, const std::vector<cv::Point>& poly) {
  int min_x = std::numeric_limits<int>::max();
  int max_x = std::numeric_limits<int>::min();
  int min_y = std::numeric_limits<int>::max();
  int max_y = std::numeric_limits<int>::min();
  std::vector<cv::Point> points(4);
  target.centroid_x = 0;
  target.centroid_y = 0;
  for (auto point : poly) {
    if (point.x < min_x)
      min_x = point.x;
    if (point.x > max_x)
      max_x = point.x;
    if (point.y < min_y)
      min_y = point.y;
    if (point.y > max_y)
      max_y = point.y;
    target.centroid_x += point.x;
    target.centroid_y += point.y;
  }
  target.centroid_x /= poly.size();
  target.centroid_y /= poly.size();
  target.width = max_x - min_x;
  target.height = max_y - min_y;
  points[0] = {max_x, max_y};
  points[1] = {max_x, min_y};
  points[2] = {min_x, min_y};
  points[3] = {min_x, max_y};
  target.points = points;
}

inline void GetTargetInfo(TargetInfo& target, const cv::Rect& rect) {
  std::vector<cv::Point> points(4);
  cv::Point center((rect.tl() + rect.br())/2);
  target.centroid_x = center.x;
  target.centroid_y = center.y;
  target.width = rect.width;
  target.height = rect.height;
  points[0] = {rect.x+rect.width, rect.y+rect.height};
  points[1] = {rect.x+rect.width, rect.y};
  points[2] = {rect.x, rect.y};
  points[3] = {rect.x, rect.y+rect.height};
  target.points = points;
}

const double kNearlyHorizontalSlope = 1 / 1.25;
const double kNearlyVerticalSlope = 1.25;
const double kPegHeightSimilarity = .8;
const double kBoilerWidthSimilarity = .8;
const double minArea = 100.0;
const double minPerimeter = 50.0;
const double minWidth = 0.0;
const double maxWidth = 1000.0;
const double minHeight = 0.0;
const double maxHeight = 1000;
const double solidity[] = {85, 100};
const double minRatio = 0.0;
const double maxRatio = 1000.0;

bool filterFcn (std::vector<cv::Point> contour) {
  std::vector<cv::Point> hull;
  cv::Rect bb = boundingRect(contour);
  if (bb.width < minWidth || bb.width > maxWidth) return true;
  if (bb.height < minHeight || bb.height > maxHeight) return true;
  double area = cv::contourArea(contour);
  if (area < minArea) return true;
  if (arcLength(contour, true) < minPerimeter) return true;
  cv::convexHull(cv::Mat(contour, true), hull);
  double solid = 100 * area / cv::contourArea(hull);
  if (solid < solidity[0] || solid > solidity[1]) return true;
  double ratio = (double) bb.width / (double) bb.height;
  if (ratio < minRatio || ratio > maxRatio) return true;
  return false;
}

//Modified from GRIP output
inline void filterContours(std::vector<std::vector<cv::Point> > &contours) {
  contours.erase(std::remove_if(contours.begin(),contours.end(),filterFcn),contours.end());
}

std::vector<TargetInfo> processImpl(int w, int h, int texOut, DisplayMode mode,
                                    int h_min, int h_max, int s_min, int s_max,
                                    int v_min, int v_max) {
  LOGD("Image is %d x %d", w, h);
  LOGD("H %d-%d S %d-%d V %d-%d", h_min, h_max, s_min, s_max, v_min, v_max);
  int64_t t;

  static cv::Mat input;
  input.create(h, w, CV_8UC4);

  // read
  t = getTimeMs();
  glReadPixels(0, 0, w, h, GL_RGBA, GL_UNSIGNED_BYTE, input.data);
  LOGD("glReadPixels() costs %d ms", getTimeInterval(t));

  // modify
  t = getTimeMs();
  static cv::Mat hsv;
  cv::cvtColor(input, hsv, CV_RGBA2RGB);
  cv::cvtColor(hsv, hsv, CV_RGB2HSV);
  LOGD("cvtColor() costs %d ms", getTimeInterval(t));

  t = getTimeMs();
  static cv::Mat thresh;
  cv::inRange(hsv, cv::Scalar(h_min, s_min, v_min),
              cv::Scalar(h_max, s_max, v_max), thresh);
  LOGD("inRange() costs %d ms", getTimeInterval(t));

  t = getTimeMs();
  static cv::Mat contour_input;
  contour_input = thresh.clone();
  std::vector<std::vector<cv::Point>> contours;
  std::vector<cv::Point> convex_contour;
  std::vector<cv::Point> poly;
  //std::vector<contourPair_t> contour_pairs;
  std::vector<TargetInfo> targets;
  std::vector<TargetInfo> rejected_targets;
  cv::findContours(contour_input, contours, cv::RETR_EXTERNAL,
                   cv::CHAIN_APPROX_TC89_KCOS);

  filterContours(contours);

  //Check target positions compared to each other
  if(contours.size() >= 2) {
    for(int i=0;i<contours.size() - 1;i++) {
      cv::Rect bb_i = boundingRect(contours[i]);
      for(int j=i+1;j<contours.size();j++) {
        cv::Rect bb_j = boundingRect(contours[j]);
        cv::Rect candidate_bb = bb_i | bb_j;
        TargetInfo target;
        GetTargetInfo(target,candidate_bb);

        //make sure they do not intersect
        if((bb_i & bb_j).area() > 0) {
          LOGD("Rejected for intersection");
          continue;
        }

        //check for parrallel
        cv::Point center_i((bb_i.tl() + bb_i.br())/2);
        cv::Point center_j((bb_j.tl() + bb_j.br())/2);
        double dx = center_i.x - center_j.x;
        double dy = center_i.y - center_j.y;
        double slope = std::numeric_limits<double>::max();
        if (dx != 0) {
          slope = dy / dx;
        }

        //Horizontal --> check for peg target
        //Vertical --> check for goal target
        TargetType target_type = TARGET_NONE;
        double height_proportion = std::abs(double(std::min(bb_j.height,bb_i.height)) /
                                            double(std::max(bb_j.height,bb_i.height)));
        double width_proportion = std::abs(double(std::min(bb_j.width,bb_i.width)) /
                                            double(std::max(bb_j.width,bb_i.width)));

        if (std::abs(slope) <= kNearlyHorizontalSlope) {
          if(height_proportion >= kPegHeightSimilarity) {
            target_type = TARGET_PEG;
          }
          else {
            target_type = TARGET_NONE;
            LOGD("Rejected for PEG pair height mismatch");
          }
        } else if (std::abs(slope) >= kNearlyVerticalSlope) {
          if(width_proportion >= kBoilerWidthSimilarity) {
            target_type = TARGET_BOILER;
          }
          else {
            target_type = TARGET_NONE;
            LOGD("Rejected for BOILER width mismatch: %f",width_proportion);
            rejected_targets.push_back(std::move(target));
          }
        } else {
          LOGD("Rejected for not parallel pair");
          target_type = TARGET_NONE;
        }
        if(target_type != TARGET_NONE) {

          LOGD("Contour pair matched and added");

          //Filter by final bounding box proportions/size
          //if(candidate_bb.

          //contour_pairs.push_back(contourPair_t(contours[i],contours[j]));
          LOGD("Found target at %.2lf, %.2lf...size %.2lf, %.2lf",
                target.centroid_x, target.centroid_y, target.width, target.height);
          //Some bug in the system prevents more than 3 targets, not sure if
          //its here or elsewhere in the app, ignore for now
          if(targets.size() < 3) {
            targets.push_back(target);
          }

        } else{
          continue;
        }
      }
    }

  }else{
    contours.clear();
  } //if(contours.size() >= 2)

  /*
  for (auto &contour : contours) {

    convex_contour.clear();
    cv::convexHull(contour, convex_contour, false);

    LOGD("Partial target analysis costs %d ms", getTimeInterval(t));

    poly.clear();
    cv::approxPolyDP(convex_contour, poly, 20, true);

    if (poly.size() == 4 && cv::isContourConvex(poly)) {
      TargetInfo target;
      GetTargetInfo(target,poly);

      // Filter based on size
      // Keep in mind width/height are in imager terms...
      const double kMinTargetWidth = 20;
      const double kMaxTargetWidth = 300;
      const double kMinTargetHeight = 10;
      const double kMaxTargetHeight = 100;
      if (target.width < kMinTargetWidth || target.width > kMaxTargetWidth ||
          target.height < kMinTargetHeight ||
          target.height > kMaxTargetHeight) {
        //TODO: run at the end instead
        LOGD("Would Reject target due to size");
        //rejected_targets.push_back(std::move(target));
        //continue;
      }
      // Filter based on shape
      int num_nearly_horizontal_slope = 0;
      int num_nearly_vertical_slope = 0;
      bool last_edge_vertical = false;
      for (size_t i = 0; i < 4; ++i) {
        double dy = target.points[i].y - target.points[(i + 1) % 4].y;
        double dx = target.points[i].x - target.points[(i + 1) % 4].x;
        double slope = std::numeric_limits<double>::max();
        if (dx != 0) {
          slope = dy / dx;
        }
        if (std::abs(slope) <= kNearlyHorizontalSlope &&
            (i == 0 || last_edge_vertical)) {
          last_edge_vertical = false;
          num_nearly_horizontal_slope++;
        } else if (std::abs(slope) >= kNearlyVerticalSlope &&
                   (i == 0 || !last_edge_vertical)) {
          last_edge_vertical = true;
          num_nearly_vertical_slope++;
        } else {
          break;
        }
      }
      if (num_nearly_horizontal_slope != 2 && num_nearly_vertical_slope != 2) {
        LOGD("Rejecting target due to shape");
        rejected_targets.push_back(std::move(target));
        continue;
      }
      // Filter based on fullness
      // TODO: change this to a size check (fullness not so useful in this case)
      const double kMinFullness = .6;
      const double kMaxFullness = 1.4;
      double original_contour_area = cv::contourArea(contour);
      double poly_area = cv::contourArea(poly);
      double fullness = original_contour_area / poly_area;
      if (fullness < kMinFullness || fullness > kMaxFullness) {
        LOGD("Rejected target due to fullness: %f", fullness);
        rejected_targets.push_back(std::move(target));
        continue;
      }

      // We found a --partial-- target
      LOGD("Found partial target");
      target_parts.push_back(std::move(target));
    }
    else {
      LOGD("Poly size: %d",poly.size());
    }
  }
  LOGD("Contour analysis costs %d ms", getTimeInterval(t));
  */

  // write back
  t = getTimeMs();
  static cv::Mat vis;
  if (mode == DISP_MODE_RAW) {
    vis = input;
  } else if (mode == DISP_MODE_THRESH) {
    cv::cvtColor(thresh, vis, CV_GRAY2RGBA);
  } else {
    vis = input;
    // Render the targets
    for (auto &target : targets) {
      cv::polylines(vis, target.points, true, cv::Scalar(0, 112, 255), 3);
      cv::circle(vis, cv::Point(target.centroid_x, target.centroid_y), 5,
                 cv::Scalar(0, 112, 255), 3);
    }
  }
  if (mode == DISP_MODE_TARGETS_PLUS) {
    for (auto &target : rejected_targets) {
      cv::polylines(vis, target.points, true, cv::Scalar(255, 0, 0), 3);
    }
  }
  LOGD("Creating vis costs %d ms", getTimeInterval(t));

  glActiveTexture(GL_TEXTURE0);
  glBindTexture(GL_TEXTURE_2D, texOut);
  t = getTimeMs();
  glTexSubImage2D(GL_TEXTURE_2D, 0, 0, 0, w, h, GL_RGBA, GL_UNSIGNED_BYTE,
                  vis.data);
  LOGD("glTexSubImage2D() costs %d ms", getTimeInterval(t));

  return targets;
}

static bool sFieldsRegistered = false;

static jfieldID sNumTargetsField;
static jfieldID sTargetsField;

static jfieldID sCentroidXField;
static jfieldID sCentroidYField;
static jfieldID sWidthField;
static jfieldID sHeightField;

static void ensureJniRegistered(JNIEnv *env) {
  if (sFieldsRegistered) {
    return;
  }
  sFieldsRegistered = true;
  jclass targetsInfoClass =
      env->FindClass("com/team254/cheezdroid/NativePart$TargetsInfo");
  sNumTargetsField = env->GetFieldID(targetsInfoClass, "numTargets", "I");
  sTargetsField = env->GetFieldID(
      targetsInfoClass, "targets",
      "[Lcom/team254/cheezdroid/NativePart$TargetsInfo$Target;");
  jclass targetClass =
      env->FindClass("com/team254/cheezdroid/NativePart$TargetsInfo$Target");

  sCentroidXField = env->GetFieldID(targetClass, "centroidX", "D");
  sCentroidYField = env->GetFieldID(targetClass, "centroidY", "D");
  sWidthField = env->GetFieldID(targetClass, "width", "D");
  sHeightField = env->GetFieldID(targetClass, "height", "D");
}

extern "C" void processFrame(JNIEnv *env, int tex1, int tex2, int w, int h,
                             int mode, int h_min, int h_max, int s_min,
                             int s_max, int v_min, int v_max,
                             jobject destTargetInfo) {
  auto targets = processImpl(w, h, tex2, static_cast<DisplayMode>(mode), h_min,
                             h_max, s_min, s_max, v_min, v_max);
  int numTargets = targets.size();
  ensureJniRegistered(env);
  env->SetIntField(destTargetInfo, sNumTargetsField, numTargets);
  if (numTargets == 0) {
    return;
  }
  jobjectArray targetsArray = static_cast<jobjectArray>(
      env->GetObjectField(destTargetInfo, sTargetsField));
  for (int i = 0; i < std::min(numTargets, 3); ++i) {
    jobject targetObject = env->GetObjectArrayElement(targetsArray, i);
    const auto &target = targets[i];
    env->SetDoubleField(targetObject, sCentroidXField, target.centroid_x);
    env->SetDoubleField(targetObject, sCentroidYField, target.centroid_y);
    env->SetDoubleField(targetObject, sWidthField, target.width);
    env->SetDoubleField(targetObject, sHeightField, target.height);
  }
}
