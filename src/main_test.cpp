#include <iostream>
#include <sstream>
#include <cstdlib>
#include <opencv2/opencv.hpp>
#include <chrono>
#include <thread>
#include <mutex>
#include <condition_variable>
#include <queue>
#include <filesystem>
#include <fstream>
#include <atomic>
#include <memory>
#include <iomanip>
#include <algorithm>
#include <array>
#include <cstdint>
#include <cmath>
#include <deque>
#include <limits>
#include <map>
#include <optional>
#include <stdexcept>
#include <vector>
#include "yolo_engine.hpp"

// ROS2 includes
#include <rclcpp/rclcpp.hpp>
#include <std_msgs/msg/string.hpp>
#include <geometry_msgs/msg/pose_stamped.hpp>
#include <builtin_interfaces/msg/time.hpp>
#include <px4_msgs/msg/vehicle_odometry.hpp>

// --- CAMERA CONFIGURATION (can be overridden via command line) ---
int CAP_WIDTH = 3840;
int CAP_HEIGHT = 1200;
int FPS = 20;
int CSI_SENSOR_0 = 0;
int CSI_SENSOR_1 = 1;
int CSI_SENSOR_WIDTH = 1920;
int CSI_SENSOR_HEIGHT = 1200;
int CSI_SENSOR_FPS = 60;
std::string ENGINE_PATH = "best.engine";

// Flags to save frames to disk
bool SAVE_ALL = false;
bool SAVE_DETECTED = false;
float CONF_THRESHOLD = 0.45f;
std::string SESSION_FOLDER = "";
constexpr double kPi = 3.141592653589793238462643383279502884;

std::string get_session_folder() {
    auto now = std::chrono::system_clock::now();
    auto in_time_t = std::chrono::system_clock::to_time_t(now);
    std::stringstream ss;
    ss << "/home/sapience/ros_pkgs/saved_images/session_" 
       << std::put_time(std::localtime(&in_time_t), "%Y%m%d_%H%M%S");
    return ss.str();
}

std::string get_pipeline() {
    const int sink_width = CAP_WIDTH / 2;
    return "nvcompositor name=comp "
           "sink_0::width=" + std::to_string(sink_width) + " "
           "sink_0::height=" + std::to_string(CAP_HEIGHT) + " "
           "sink_0::xpos=0 sink_0::ypos=0 "
           "sink_1::width=" + std::to_string(sink_width) + " "
           "sink_1::height=" + std::to_string(CAP_HEIGHT) + " "
           "sink_1::xpos=" + std::to_string(sink_width) + " sink_1::ypos=0 ! "
           "video/x-raw(memory:NVMM), width=" + std::to_string(CAP_WIDTH) + ", "
           "height=" + std::to_string(CAP_HEIGHT) + ", "
           "framerate=" + std::to_string(FPS) + "/1 ! "
           "queue ! "
           "nvvidconv interpolation-method=1 ! "
           "video/x-raw, width=" + std::to_string(CAP_WIDTH) + ", "
           "height=" + std::to_string(CAP_HEIGHT) + ", format=BGRx ! "
           "videoconvert ! "
           "video/x-raw, format=BGR ! "
           "appsink drop=1 "
           "nvarguscamerasrc sensor-id=" + std::to_string(CSI_SENSOR_0) + " "
           "gainrange=\"3 10\" exposuretimerange=\"40000 8333000\" ! "
           "video/x-raw(memory:NVMM), width=" + std::to_string(CSI_SENSOR_WIDTH) + ", "
           "height=" + std::to_string(CSI_SENSOR_HEIGHT) + ", "
           "framerate=" + std::to_string(CSI_SENSOR_FPS) + "/1, format=NV12 ! "
           "videorate ! "
           "video/x-raw(memory:NVMM), framerate=" + std::to_string(FPS) + "/1, format=NV12 ! "
           "comp.sink_0 "
           "nvarguscamerasrc sensor-id=" + std::to_string(CSI_SENSOR_1) + " "
           "gainrange=\"3 10\" exposuretimerange=\"40000 8333000\" ! "
           "video/x-raw(memory:NVMM), width=" + std::to_string(CSI_SENSOR_WIDTH) + ", "
           "height=" + std::to_string(CSI_SENSOR_HEIGHT) + ", "
           "framerate=" + std::to_string(CSI_SENSOR_FPS) + "/1, format=NV12 ! "
           "videorate ! "
           "video/x-raw(memory:NVMM), framerate=" + std::to_string(FPS) + "/1, format=NV12 ! "
           "comp.sink_1";
}

// ========== THREAD-SAFE SHARED FRAME BUFFER ==========
class SharedFrameBuffer {
public:
    void update(const cv::Mat& new_frame, std::chrono::system_clock::time_point timestamp) {
        std::unique_lock<std::mutex> lock(mutex_);
        frame_ = new_frame.clone(); // Deep copy
        timestamp_ = timestamp;
        frame_ready_ = true;
        lock.unlock();
        cv_.notify_one();
    }

    bool get_and_clear(cv::Mat& output_frame, std::chrono::system_clock::time_point& timestamp) {
        // Must be called with lock held
        if (!frame_ready_) return false;
        output_frame = frame_.clone(); // Deep copy
        timestamp = timestamp_;
        frame_ready_ = false; // Clear the flag so we wait for next frame
        return true;
    }

    bool wait_for_new_frame(std::unique_lock<std::mutex>& lock, const std::atomic<bool>& running) {
        cv_.wait(lock, [this, &running]{ return frame_ready_ || !running.load(); });
        return frame_ready_;
    }

    void notify_shutdown() {
        cv_.notify_all();
    }

    std::mutex& get_mutex() { return mutex_; }

private:
    cv::Mat frame_;
    std::chrono::system_clock::time_point timestamp_;
    bool frame_ready_ = false;
    std::mutex mutex_;
    std::condition_variable cv_;
};

// ========== THREAD-SAFE DETECTION QUEUE ==========
struct DetectionResult {
    std::vector<Detection> detections;
    std::chrono::system_clock::time_point capture_timestamp;  // When frame was captured
    std::chrono::system_clock::time_point inference_timestamp; // When inference completed
    int frame_id;
    int frame_width = CAP_WIDTH;
    int frame_height = CAP_HEIGHT;
};

class DetectionQueue {
public:
    void push(const DetectionResult& result) {
        std::lock_guard<std::mutex> lock(mutex_);
        queue_.push(result);
        // Limit queue size to prevent memory buildup
        while (queue_.size() > 30) {
            queue_.pop();
        }
        cv_.notify_one();
    }

    bool pop(DetectionResult& result) {
        std::lock_guard<std::mutex> lock(mutex_);
        if (queue_.empty()) return false;
        result = queue_.front();
        queue_.pop();
        return true;
    }

    bool wait_and_pop(DetectionResult& result, int timeout_ms = 1000) {
        std::unique_lock<std::mutex> lock(mutex_);
        if (cv_.wait_for(lock, std::chrono::milliseconds(timeout_ms), 
                         [this]{ return !queue_.empty(); })) {
            result = queue_.front();
            queue_.pop();
            return true;
        }
        return false;
    }

private:
    std::queue<DetectionResult> queue_;
    std::mutex mutex_;
    std::condition_variable cv_;
};


// ========== GROUND TARGET POSE ESTIMATION (PORTED FROM ground_target_estimator.py) ==========
class GroundTargetPoseEstimator {
public:
    explicit GroundTargetPoseEstimator(const rclcpp::Node::SharedPtr& node)
        : node_(node) {
        ground_z_ = node_->declare_parameter<double>("ground_z", 0.0);
        drone_frame_id_ = node_->declare_parameter<std::string>("drone_frame_id", "drone_frd");
        min_depression_deg_ = node_->declare_parameter<double>("min_depression_deg", 8.0);
        sigma_attitude_deg_ = node_->declare_parameter<double>("sigma_attitude_deg", 1.0);
        sigma_height_m_ = node_->declare_parameter<double>("sigma_height_m", 0.3);
        process_noise_rate_ = node_->declare_parameter<double>("process_noise_rate", 0.01);
        assoc_gate_chi2_ = node_->declare_parameter<double>("assoc_gate_chi2", 9.21);
        min_observations_ = node_->declare_parameter<int>("min_observations", 4);
        max_track_age_sec_ = node_->declare_parameter<double>("max_track_age_sec", 5.0);
        confirm_max_sigma_m_ = node_->declare_parameter<double>("confirm_max_sigma_m", 1.0);
        max_odom_dt_sec_ = node_->declare_parameter<double>("max_odom_dt_sec", 0.15);
        odom_buffer_size_ = node_->declare_parameter<int>("odom_buffer_size", 400);
        // Latest camera intrinsics/distortion and camera extrinsics from aruco_nano_pose_estimator.cpp.
        const double front_pitch_deg = node_->declare_parameter<double>("camera_front_pitch_deg", 20.0);
        const double down_pitch_deg = node_->declare_parameter<double>("camera_down_pitch_deg", 90.0);
        addCamera("forward", front_pitch_deg, Anchor::BOTTOM_CENTER);
        addCamera("down", down_pitch_deg, Anchor::CENTER);

        pose_pub_ = node_->create_publisher<geometry_msgs::msg::PoseStamped>(
            "/drone/detections/pose", rclcpp::QoS(10));

        rclcpp::QoS odom_qos(10);
        odom_qos.best_effort();
        odom_sub_ = node_->create_subscription<px4_msgs::msg::VehicleOdometry>(
            "/fmu/out/vehicle_odometry", odom_qos,
            [this](const px4_msgs::msg::VehicleOdometry::SharedPtr msg) { odomCallback(msg); });

        RCLCPP_INFO(node_->get_logger(),
            "Ground target pose estimator integrated. output=/drone/detections/pose type=geometry_msgs/msg/PoseStamped, frame=%s, always active",
            drone_frame_id_.c_str());
    }

    void processDetections(const DetectionResult& result) {
        std::lock_guard<std::mutex> track_lock(track_mutex_);

        const double t_capture = captureTimeInNodeClock(result.capture_timestamp);
        auto odom = interpolateOdom(t_capture);
        if (!odom.has_value()) {
            pruneTracks(t_capture);
            return;
        }

        const cv::Matx33d R_world_body = quaternionToRotationMatrix(odom->q_world_body);
        bool updated_any = false;

        const int composite_w = result.frame_width > 0 ? result.frame_width : CAP_WIDTH;
        const int composite_h = result.frame_height > 0 ? result.frame_height : CAP_HEIGHT;
        const int half_w = std::max(1, composite_w / 2);

        for (const auto& det : result.detections) {
            Observation obs;
            if (!detectionToObservation(det, composite_h, half_w, obs)) {
                continue;
            }

            const auto cam_it = cameras_.find(obs.camera_name);
            if (cam_it == cameras_.end()) {
                continue;
            }

            Projection proj;
            if (!projectToGround(obs.u_px, obs.v_px, cam_it->second,
                                 R_world_body, odom->p_world_body, proj)) {
                continue;
            }

            updateTracker(proj.world_xy, proj.sigma, det.class_id, t_capture);
            updated_any = true;
        }

        pruneTracks(t_capture);
        if (updated_any) {
            publishConfirmedTracks(*odom, t_capture);
        }
    }

private:
    enum class Anchor { CENTER, BOTTOM_CENTER };

    struct CameraModel {
        std::string name;
        int width{1920};
        int height{1200};
        cv::Mat K;
        cv::Mat D;
        cv::Matx33d R_body_camera{cv::Matx33d::eye()};
        Anchor anchor{Anchor::CENTER};
    };

    struct OdomSample {
        double t_sec{0.0};
        cv::Vec3d p_world_body{0.0, 0.0, 0.0};
        std::array<double, 4> q_world_body{0.0, 0.0, 0.0, 1.0};  // x, y, z, w
    };

    struct Observation {
        std::string camera_name;
        double u_px{0.0};
        double v_px{0.0};
    };

    struct Projection {
        cv::Vec2d world_xy{0.0, 0.0};
        double sigma{1.0};
    };

    struct Track {
        cv::Vec2d s{0.0, 0.0};
        cv::Matx22d P{cv::Matx22d::eye()};
        int class_id{-1};
        int observations{0};
        double last_update_t{0.0};
    };

    rclcpp::Node::SharedPtr node_;
    rclcpp::Publisher<geometry_msgs::msg::PoseStamped>::SharedPtr pose_pub_;
    rclcpp::Subscription<px4_msgs::msg::VehicleOdometry>::SharedPtr odom_sub_;

    std::map<std::string, CameraModel> cameras_;
    std::deque<OdomSample> odom_buffer_;
    std::vector<Track> tracks_;
    mutable std::mutex odom_mutex_;
    std::mutex track_mutex_;

    double ground_z_{0.0};
    std::string drone_frame_id_{"drone_frd"};
    double min_depression_deg_{8.0};
    double sigma_attitude_deg_{1.0};
    double sigma_height_m_{0.3};
    double process_noise_rate_{0.01};
    double assoc_gate_chi2_{9.21};
    int min_observations_{4};
    double max_track_age_sec_{5.0};
    double confirm_max_sigma_m_{1.0};
    double max_odom_dt_sec_{0.15};
    int odom_buffer_size_{400};
    bool odom_time_offset_initialized_{false};
    double odom_time_offset_sec_{0.0};

    void addCamera(const std::string& name, double pitch_deg, Anchor anchor) {
        CameraModel cam;
        cam.name = name;
        cam.width = 1920;
        cam.height = 1200;
        cam.anchor = anchor;
        cam.K = (cv::Mat_<double>(3, 3) <<
            1134.3171058472644, 0.0, 945.03097055585,
            0.0, 1134.9884203778502, 607.0921091018031,
            0.0, 0.0, 1.0);
        cam.D = (cv::Mat_<double>(1, 5) <<
            0.02717223068021268, -0.060021841339080874,
            0.0007009591920443065, -4.569105989855734e-05,
            0.020871849677932104);
        cam.R_body_camera = makeCameraRotation(pitch_deg);
        cameras_[name] = cam;
    }

    // Same rotation as aruco_nano_pose_estimator.cpp::makeCameraTransform().
    // Camera axes: X=right, Y=down, Z=optical. Drone/body axes: FRD, X=fwd, Y=right, Z=down.
    static cv::Matx33d makeCameraRotation(double pitch_deg) {
        const double theta = pitch_deg * kPi / 180.0;
        const double s = std::sin(theta);
        const double c = std::cos(theta);
        return cv::Matx33d(
            0.0, -s,  c,
            1.0,  0.0, 0.0,
            0.0,  c,  s);
    }

    void odomCallback(const px4_msgs::msg::VehicleOdometry::SharedPtr msg) {
        const double recv_t = node_->get_clock()->now().seconds();
        double t_sec = recv_t;
        if (msg->timestamp != 0) {
            const double px4_t = static_cast<double>(msg->timestamp) * 1e-6;
            if (!odom_time_offset_initialized_) {
                odom_time_offset_sec_ = recv_t - px4_t;
                odom_time_offset_initialized_ = true;
            }
            t_sec = px4_t + odom_time_offset_sec_;
        }

        OdomSample s;
        s.t_sec = t_sec;
        s.p_world_body = cv::Vec3d(
            static_cast<double>(msg->position[0]),
            static_cast<double>(msg->position[1]),
            static_cast<double>(msg->position[2]));
        s.q_world_body = normalizeQuaternion({
            static_cast<double>(msg->q[1]),
            static_cast<double>(msg->q[2]),
            static_cast<double>(msg->q[3]),
            static_cast<double>(msg->q[0])});

        std::lock_guard<std::mutex> lock(odom_mutex_);
        if (odom_buffer_.empty() || s.t_sec >= odom_buffer_.back().t_sec) {
            odom_buffer_.push_back(s);
        } else {
            auto it = std::lower_bound(
                odom_buffer_.begin(), odom_buffer_.end(), s.t_sec,
                [](const OdomSample& a, double t) { return a.t_sec < t; });
            odom_buffer_.insert(it, s);
        }
        while (static_cast<int>(odom_buffer_.size()) > odom_buffer_size_) {
            odom_buffer_.pop_front();
        }
    }

    std::optional<OdomSample> interpolateOdom(double t_query) const {
        std::lock_guard<std::mutex> lock(odom_mutex_);
        if (odom_buffer_.empty()) {
            return std::nullopt;
        }

        const auto& front = odom_buffer_.front();
        const auto& back = odom_buffer_.back();
        if (t_query <= front.t_sec) {
            if (front.t_sec - t_query > max_odom_dt_sec_) return std::nullopt;
            return front;
        }
        if (t_query >= back.t_sec) {
            if (t_query - back.t_sec > max_odom_dt_sec_) return std::nullopt;
            return back;
        }

        auto it = std::upper_bound(
            odom_buffer_.begin(), odom_buffer_.end(), t_query,
            [](double t, const OdomSample& s) { return t < s.t_sec; });
        if (it == odom_buffer_.begin() || it == odom_buffer_.end()) {
            return std::nullopt;
        }

        const auto& b = *it;
        const auto& a = *(it - 1);
        const double dt = b.t_sec - a.t_sec;
        const double u = dt <= 1e-9 ? 0.0 : std::clamp((t_query - a.t_sec) / dt, 0.0, 1.0);

        OdomSample out;
        out.t_sec = t_query;
        out.p_world_body = a.p_world_body + u * (b.p_world_body - a.p_world_body);
        out.q_world_body = slerpQuaternion(a.q_world_body, b.q_world_body, u);
        return out;
    }

    bool detectionToObservation(const Detection& det, int composite_h,
                                int half_w, Observation& obs) const {
        if (half_w <= 0 || composite_h <= 0) {
            return false;
        }

        const double box_x = static_cast<double>(det.box.x);
        const double box_y = static_cast<double>(det.box.y);
        const double box_w = static_cast<double>(det.box.width);
        const double box_h = static_cast<double>(det.box.height);
        const double center_x_composite = box_x + 0.5 * box_w;
        const bool is_forward = center_x_composite < static_cast<double>(half_w);

        const CameraModel& cam = is_forward ? cameras_.at("forward") : cameras_.at("down");
        obs.camera_name = is_forward ? "forward" : "down";

        double x_single = is_forward ? box_x : (box_x - static_cast<double>(half_w));
        x_single = std::clamp(x_single, 0.0, static_cast<double>(half_w - 1));

        double u = x_single + 0.5 * box_w;
        double v = box_y + 0.5 * box_h;
        if (cam.anchor == Anchor::BOTTOM_CENTER) {
            v = box_y + box_h;
        }

        u = std::clamp(u, 0.0, static_cast<double>(half_w - 1));
        v = std::clamp(v, 0.0, static_cast<double>(composite_h - 1));

        // Scale from the current half-image dimensions into the calibrated 1920x1200 image plane.
        obs.u_px = u * static_cast<double>(cam.width) / static_cast<double>(half_w);
        obs.v_px = v * static_cast<double>(cam.height) / static_cast<double>(composite_h);
        return true;
    }

    bool projectToGround(double u_px, double v_px, const CameraModel& cam,
                         const cv::Matx33d& R_world_body,
                         const cv::Vec3d& p_world_body,
                         Projection& out) const {
        cv::Mat src(1, 1, CV_64FC2);
        src.at<cv::Vec2d>(0, 0) = cv::Vec2d(u_px, v_px);
        cv::Mat undistorted;
        cv::undistortPoints(src, undistorted, cam.K, cam.D);
        const cv::Vec2d xy = undistorted.at<cv::Vec2d>(0, 0);

        const cv::Vec3d ray_cam(xy[0], xy[1], 1.0);
        cv::Vec3d ray_body = matVec(cam.R_body_camera, ray_cam);
        ray_body = ray_body / std::max(1e-12, cv::norm(ray_body));
        cv::Vec3d ray_world = matVec(R_world_body, ray_body);
        ray_world = ray_world / std::max(1e-12, cv::norm(ray_world));

        const double min_down = std::sin(min_depression_deg_ * kPi / 180.0);
        if (ray_world[2] <= min_down) {
            return false;
        }

        const double t = (ground_z_ - p_world_body[2]) / ray_world[2];
        if (t <= 0.0 || !std::isfinite(t)) {
            return false;
        }

        const cv::Vec3d Pw = p_world_body + t * ray_world;
        out.world_xy = cv::Vec2d(Pw[0], Pw[1]);

        const double range = cv::norm(Pw - p_world_body);
        const double attitude_sigma = range * std::tan(sigma_attitude_deg_ * kPi / 180.0);
        const double height_sigma = std::fabs(sigma_height_m_ / std::max(ray_world[2], 0.05));
        const double pixel_sigma = std::max(0.02, range * 2.0 / std::max(cam.K.at<double>(0, 0), cam.K.at<double>(1, 1)));
        out.sigma = std::max(0.05, std::sqrt(
            attitude_sigma * attitude_sigma +
            height_sigma * height_sigma +
            pixel_sigma * pixel_sigma));
        return true;
    }

    void updateTracker(const cv::Vec2d& z, double sigma, int class_id, double t_sec) {
        const cv::Matx22d R = cv::Matx22d::eye() * (sigma * sigma);
        double best_d2 = std::numeric_limits<double>::infinity();
        int best_idx = -1;

        for (size_t i = 0; i < tracks_.size(); ++i) {
            Track& tr = tracks_[i];
            if (tr.class_id != class_id) {
                continue;
            }
            predictTrack(tr, t_sec);
            const cv::Vec2d e = z - tr.s;
            const cv::Matx22d S = tr.P + R;
            const double d2 = mahalanobis2(e, S);
            if (d2 < best_d2) {
                best_d2 = d2;
                best_idx = static_cast<int>(i);
            }
        }

        if (best_idx >= 0 && best_d2 <= assoc_gate_chi2_) {
            correctTrack(tracks_[best_idx], z, R, t_sec);
            return;
        }

        Track tr;
        tr.s = z;
        tr.P = R;
        tr.class_id = class_id;
        tr.observations = 1;
        tr.last_update_t = t_sec;
        tracks_.push_back(tr);
    }

    void predictTrack(Track& tr, double t_sec) const {
        const double dt = std::max(0.0, t_sec - tr.last_update_t);
        const double q = process_noise_rate_ * std::max(dt, 1e-3);
        tr.P(0, 0) += q;
        tr.P(1, 1) += q;
    }

    void correctTrack(Track& tr, const cv::Vec2d& z, const cv::Matx22d& R, double t_sec) {
        const cv::Vec2d e = z - tr.s;
        const cv::Matx22d S = tr.P + R;
        const cv::Matx22d S_inv = inverse2x2(S);
        const cv::Matx22d K = tr.P * S_inv;
        tr.s = tr.s + K * e;
        tr.P = (cv::Matx22d::eye() - K) * tr.P;
        tr.P(0, 1) = tr.P(1, 0) = 0.5 * (tr.P(0, 1) + tr.P(1, 0));
        tr.observations += 1;
        tr.last_update_t = t_sec;
    }

    void pruneTracks(double t_sec) {
        tracks_.erase(
            std::remove_if(tracks_.begin(), tracks_.end(),
                [this, t_sec](const Track& tr) {
                    return (t_sec - tr.last_update_t) > max_track_age_sec_;
                }),
            tracks_.end());
    }

    bool isConfirmed(const Track& tr) const {
        const double max_sigma = std::sqrt(std::max(tr.P(0, 0), tr.P(1, 1)));
        return tr.observations >= min_observations_ && max_sigma <= confirm_max_sigma_m_;
    }

    void publishConfirmedTracks(const OdomSample& odom, double stamp_sec) {
        const cv::Matx33d R_world_body = quaternionToRotationMatrix(odom.q_world_body);

        for (const auto& tr : tracks_) {
            if (!isConfirmed(tr)) {
                continue;
            }

            const cv::Vec3d Pw(tr.s[0], tr.s[1], ground_z_);
            const cv::Vec3d p_body = matVec(R_world_body.t(), Pw - odom.p_world_body);

            geometry_msgs::msg::PoseStamped msg;
            msg.header.stamp = secToTimeMsg(stamp_sec);
            msg.header.frame_id = drone_frame_id_;
            msg.pose.position.x = p_body[0];
            msg.pose.position.y = p_body[1];
            msg.pose.position.z = p_body[2];
            msg.pose.orientation.x = 0.0;
            msg.pose.orientation.y = 0.0;
            msg.pose.orientation.z = 0.0;
            msg.pose.orientation.w = 1.0;
            pose_pub_->publish(msg);
        }
    }

    double captureTimeInNodeClock(const std::chrono::system_clock::time_point& capture_time) const {
        const double measured_latency_sec = std::max(
            0.0,
            std::chrono::duration<double>(std::chrono::system_clock::now() - capture_time).count());
        return node_->get_clock()->now().seconds() - measured_latency_sec;
    }

    static builtin_interfaces::msg::Time secToTimeMsg(double t_sec) {
        builtin_interfaces::msg::Time msg;
        if (t_sec < 0.0 || !std::isfinite(t_sec)) {
            msg.sec = 0;
            msg.nanosec = 0u;
            return msg;
        }
        const double sec_part = std::floor(t_sec);
        msg.sec = static_cast<int32_t>(sec_part);
        msg.nanosec = static_cast<uint32_t>(std::round((t_sec - sec_part) * 1e9));
        if (msg.nanosec >= 1000000000u) {
            msg.sec += 1;
            msg.nanosec -= 1000000000u;
        }
        return msg;
    }

    static std::array<double, 4> normalizeQuaternion(const std::array<double, 4>& q) {
        const double n = std::sqrt(q[0]*q[0] + q[1]*q[1] + q[2]*q[2] + q[3]*q[3]);
        if (n <= 1e-12) {
            return {0.0, 0.0, 0.0, 1.0};
        }
        return {q[0]/n, q[1]/n, q[2]/n, q[3]/n};
    }

    static std::array<double, 4> slerpQuaternion(
        const std::array<double, 4>& q0_in,
        const std::array<double, 4>& q1_in,
        double t) {
        const auto q0 = normalizeQuaternion(q0_in);
        auto q1 = normalizeQuaternion(q1_in);
        double dot = q0[0]*q1[0] + q0[1]*q1[1] + q0[2]*q1[2] + q0[3]*q1[3];
        if (dot < 0.0) {
            for (auto& v : q1) v = -v;
            dot = -dot;
        }
        if (dot > 0.9995) {
            return normalizeQuaternion({
                q0[0] + t * (q1[0] - q0[0]),
                q0[1] + t * (q1[1] - q0[1]),
                q0[2] + t * (q1[2] - q0[2]),
                q0[3] + t * (q1[3] - q0[3])});
        }
        const double theta0 = std::acos(std::clamp(dot, -1.0, 1.0));
        const double theta = theta0 * t;
        const double sin_theta = std::sin(theta);
        const double sin_theta0 = std::sin(theta0);
        const double s0 = std::cos(theta) - dot * sin_theta / sin_theta0;
        const double s1 = sin_theta / sin_theta0;
        return normalizeQuaternion({
            s0*q0[0] + s1*q1[0],
            s0*q0[1] + s1*q1[1],
            s0*q0[2] + s1*q1[2],
            s0*q0[3] + s1*q1[3]});
    }

    static cv::Matx33d quaternionToRotationMatrix(const std::array<double, 4>& q_in) {
        const auto q = normalizeQuaternion(q_in);
        const double x = q[0], y = q[1], z = q[2], w = q[3];
        return cv::Matx33d(
            1.0 - 2.0 * (y*y + z*z), 2.0 * (x*y - z*w),       2.0 * (x*z + y*w),
            2.0 * (x*y + z*w),       1.0 - 2.0 * (x*x + z*z), 2.0 * (y*z - x*w),
            2.0 * (x*z - y*w),       2.0 * (y*z + x*w),       1.0 - 2.0 * (x*x + y*y));
    }

    static cv::Vec3d matVec(const cv::Matx33d& R, const cv::Vec3d& v) {
        return cv::Vec3d(
            R(0, 0) * v[0] + R(0, 1) * v[1] + R(0, 2) * v[2],
            R(1, 0) * v[0] + R(1, 1) * v[1] + R(1, 2) * v[2],
            R(2, 0) * v[0] + R(2, 1) * v[1] + R(2, 2) * v[2]);
    }

    static cv::Matx22d inverse2x2(const cv::Matx22d& A) {
        const double det = A(0, 0) * A(1, 1) - A(0, 1) * A(1, 0);
        if (std::fabs(det) <= 1e-12) {
            return cv::Matx22d::eye() * 1e12;
        }
        const double inv_det = 1.0 / det;
        return cv::Matx22d(
            A(1, 1) * inv_det, -A(0, 1) * inv_det,
            -A(1, 0) * inv_det, A(0, 0) * inv_det);
    }

    static double mahalanobis2(const cv::Vec2d& e, const cv::Matx22d& S) {
        const cv::Vec2d y = inverse2x2(S) * e;
        return e.dot(y);
    }

};

// ========== THREAD 1: PRODUCER (CAMERA CAPTURE) ==========
void producer_thread(SharedFrameBuffer& frame_buffer, std::atomic<bool>& running) {
    std::string pipeline = get_pipeline();
    std::cout << "[PRODUCER] Opening Camera: " << pipeline << std::endl;
    
    cv::VideoCapture cap(pipeline, cv::CAP_GSTREAMER);
    if (!cap.isOpened()) {
        std::cerr << "[PRODUCER ERROR] Failed to open camera!" << std::endl;
        running = false;
        return;
    }

    cv::Mat temp_frame;
    int frame_count = 0;
    std::cout << "[PRODUCER] Camera running at " << FPS << " FPS. Capturing..." << std::endl;

    while (running) {
        if (!cap.read(temp_frame)) {
            std::cerr << "[PRODUCER WARN] Failed to read frame. Retrying..." << std::endl;
            std::this_thread::sleep_for(std::chrono::milliseconds(10));
            continue;
        }

        if (temp_frame.empty()) continue;

        frame_count++;
        
        // Timestamp the frame capture
        auto capture_time = std::chrono::system_clock::now();
        
        // Update shared buffer (overwrites if inference is slow - this prevents drift!)
        frame_buffer.update(temp_frame, capture_time);

        // Debug heartbeat every 30 frames (~1 second)
        if (frame_count % 30 == 0) {
            std::cout << "[PRODUCER] Captured " << frame_count << " frames" << std::endl;
        }
    }

    cap.release();
    std::cout << "[PRODUCER] Thread stopped." << std::endl;
}

// ========== THREAD 2: CONSUMER (TENSORRT INFERENCE) ==========
void consumer_thread(SharedFrameBuffer& frame_buffer, 
                     DetectionQueue& detection_queue,
                     std::atomic<bool>& running) {
    std::filesystem::create_directories("/home/sapience/ros_pkgs/saved_images");
    std::cout << "[CONSUMER] Loading TensorRT Engine..." << std::endl;
    YoloEngine detector(ENGINE_PATH);
    std::cout << "[CONSUMER] Engine loaded. Starting inference loop..." << std::endl;

    cv::Mat inference_buffer;
    int processed_count = 0;

    while (running) {
        // Wait for new frame notification, but also wake cleanly during shutdown.
        std::unique_lock<std::mutex> lock(frame_buffer.get_mutex());
        if (!frame_buffer.wait_for_new_frame(lock, running)) {
            break;
        }
        
        // Deep copy frame to local buffer and get its capture timestamp
        std::chrono::system_clock::time_point capture_time;
        if (!frame_buffer.get_and_clear(inference_buffer, capture_time)) {
            lock.unlock();
            continue;
        }
        lock.unlock();

        // Run inference (this can take 20-50ms)
        auto start = std::chrono::high_resolution_clock::now();
        auto detections = detector.infer(inference_buffer, CONF_THRESHOLD);
        auto end = std::chrono::high_resolution_clock::now();
        auto infer_time = std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count();

        processed_count++;

        // Package result with timestamps
        DetectionResult result;
        result.detections = std::move(detections);
        result.capture_timestamp = capture_time;
        result.inference_timestamp = std::chrono::system_clock::now();
        result.frame_id = processed_count;
        result.frame_width = inference_buffer.cols;
        result.frame_height = inference_buffer.rows;

        // Save raw frames and YOLO labels according to flags
        if (SAVE_ALL || SAVE_DETECTED) {
            bool has_left_detection = false;
            bool has_right_detection = false;
            int half_w = inference_buffer.cols / 2;
            int h = inference_buffer.rows;

            // Classify detections into left (forward) or right (down) camera space
            for (const auto& det : result.detections) {
                if (det.box.x < half_w) {
                    has_left_detection = true;
                } else {
                    has_right_detection = true;
                }
            }

            // Split composite frame into raw left (forward) and right (down) halves
            cv::Mat left_frame = inference_buffer(cv::Rect(0, 0, half_w, h));
            cv::Mat right_frame = inference_buffer(cv::Rect(half_w, 0, half_w, h));

            // Save left frame and YOLO labels
            if (SAVE_ALL || (SAVE_DETECTED && has_left_detection)) {
                std::string base_name = SESSION_FOLDER + "/forward/frame_" + std::to_string(processed_count);
                cv::imwrite(base_name + ".jpg", left_frame);
                
                if (has_left_detection) {
                    std::ofstream label_file(base_name + ".txt");
                    for (const auto& det : result.detections) {
                        if (det.box.x < half_w) {
                            double x_center = det.box.x + det.box.width / 2.0;
                            double y_center = det.box.y + det.box.height / 2.0;
                            double norm_cx = x_center / half_w;
                            double norm_cy = y_center / h;
                            double norm_w = (double)det.box.width / half_w;
                            double norm_h = (double)det.box.height / h;
                            
                            label_file << det.class_id << " "
                                       << std::fixed << std::setprecision(6)
                                       << norm_cx << " " << norm_cy << " "
                                       << norm_w << " " << norm_h << " "
                                       << det.confidence << "\n";
                        }
                    }
                }
            }

            // Save right frame and YOLO labels
            if (SAVE_ALL || (SAVE_DETECTED && has_right_detection)) {
                std::string base_name = SESSION_FOLDER + "/down/frame_" + std::to_string(processed_count);
                cv::imwrite(base_name + ".jpg", right_frame);
                
                if (has_right_detection) {
                    std::ofstream label_file(base_name + ".txt");
                    for (const auto& det : result.detections) {
                        if (det.box.x >= half_w) {
                            double x_single = det.box.x - half_w;
                            double x_center = x_single + det.box.width / 2.0;
                            double y_center = det.box.y + det.box.height / 2.0;
                            double norm_cx = x_center / half_w;
                            double norm_cy = y_center / h;
                            double norm_w = (double)det.box.width / half_w;
                            double norm_h = (double)det.box.height / h;
                            
                            label_file << det.class_id << " "
                                       << std::fixed << std::setprecision(6)
                                       << norm_cx << " " << norm_cy << " "
                                       << norm_w << " " << norm_h << " "
                                       << det.confidence << "\n";
                        }
                    }
                }
            }
        }

        // Push to ROS thread
        detection_queue.push(result);

        // Debug output
        if (!result.detections.empty()) {
            std::cout << "[CONSUMER] Frame " << processed_count 
                      << " | " << infer_time << "ms | Found: " << result.detections.size() << " objects" << std::endl;
        } else if (processed_count % 30 == 0) {
            std::cout << "[CONSUMER] Frame " << processed_count << " | " << infer_time << "ms | No detections" << std::endl;
        }
    }

    std::cout << "[CONSUMER] Thread stopped." << std::endl;
}

// ========== THREAD 3: COMMUNICATOR (ROS2 PUBLISHER) ==========
void communicator_thread(DetectionQueue& detection_queue,
                         std::atomic<bool>& running,
                         rclcpp::Node::SharedPtr node,
                         std::shared_ptr<GroundTargetPoseEstimator> pose_estimator) {
    auto publisher = node->create_publisher<std_msgs::msg::String>("/drone/detections", 10);
    std::cout << "[COMMUNICATOR] ROS2 publisher ready on /drone/detections" << std::endl;

    int published_count = 0;
    auto last_hz_calc_time = std::chrono::steady_clock::now();
    int frames_in_window = 0;
    double current_hz = 0.0;

    while (running && rclcpp::ok()) {
        DetectionResult result;
        
        // Wait for new detections (with timeout to check running flag)
        if (!detection_queue.wait_and_pop(result, 500)) {
            continue;
        }

        // Calculate latency (time from capture to now)
        auto now = std::chrono::system_clock::now();
        auto total_latency_ms = std::chrono::duration_cast<std::chrono::milliseconds>(
            now - result.capture_timestamp).count();
        auto inference_latency_ms = std::chrono::duration_cast<std::chrono::milliseconds>(
            result.inference_timestamp - result.capture_timestamp).count();

        // Convert to simplified ROS2 message
        // Format: "frame_id,class_id,confidence,x,y,width,height" (one detection per line)
        std_msgs::msg::String msg;
        std::stringstream ss;
        
        for (const auto& det : result.detections) {
            // Format: frame_id,class_id,confidence,x,y,width,height
            ss << result.frame_id << ","
               << det.class_id << "," 
               << std::fixed << std::setprecision(3) << det.confidence << ","
               << det.box.x << "," 
               << det.box.y << "," 
               << det.box.width << "," 
               << det.box.height << "\n";
        }
        
        msg.data = ss.str();

        // Publish raw detection CSV
        publisher->publish(msg);

        // Estimate and publish confirmed ground-target poses on /drone/detections/pose.
        if (pose_estimator) {
            pose_estimator->processDetections(result);
        }
        published_count++;
        frames_in_window++;

        // Calculate Hz every second
        auto now_steady = std::chrono::steady_clock::now();
        auto elapsed_ms = std::chrono::duration_cast<std::chrono::milliseconds>(
            now_steady - last_hz_calc_time).count();
        
        if (elapsed_ms >= 1000) {
            current_hz = (frames_in_window * 1000.0) / elapsed_ms;
            last_hz_calc_time = now_steady;
            frames_in_window = 0;
        }

        // Display stats
        if (!result.detections.empty()) {
            std::cout << "[COMMUNICATOR] Frame " << result.frame_id 
                      << " | Latency: " << total_latency_ms << "ms"
                      << " (infer: " << inference_latency_ms << "ms)"
                      << " | Objects: " << result.detections.size()
                      << " | Output Hz: " << std::fixed << std::setprecision(1) << current_hz
                      << std::endl;
        } else if (published_count % 30 == 0) {
            std::cout << "[COMMUNICATOR] Frame " << result.frame_id 
                      << " | Latency: " << total_latency_ms << "ms"
                      << " | No detections"
                      << " | Output Hz: " << std::fixed << std::setprecision(1) << current_hz
                      << std::endl;
        }
    }

    std::cout << "[COMMUNICATOR] Thread stopped." << std::endl;
}

// ========== MAIN: THREAD ORCHESTRATION ==========
int main(int argc, char** argv) {
    // Parse command-line arguments for camera settings
    for (int i = 1; i < argc; ++i) {
        std::string arg = argv[i];
        if ((arg == "-w" || arg == "--width") && i + 1 < argc) {
            CAP_WIDTH = std::atoi(argv[++i]);
        } else if ((arg == "-h" || arg == "--height") && i + 1 < argc) {
            CAP_HEIGHT = std::atoi(argv[++i]);
        } else if ((arg == "-f" || arg == "--fps") && i + 1 < argc) {
            FPS = std::atoi(argv[++i]);
        } else if (arg == "--sensor0" && i + 1 < argc) {
            CSI_SENSOR_0 = std::atoi(argv[++i]);
        } else if (arg == "--sensor1" && i + 1 < argc) {
            CSI_SENSOR_1 = std::atoi(argv[++i]);
        } else if (arg == "--sensor-width" && i + 1 < argc) {
            CSI_SENSOR_WIDTH = std::atoi(argv[++i]);
        } else if (arg == "--sensor-height" && i + 1 < argc) {
            CSI_SENSOR_HEIGHT = std::atoi(argv[++i]);
        } else if (arg == "--sensor-fps" && i + 1 < argc) {
            CSI_SENSOR_FPS = std::atoi(argv[++i]);
        } else if ((arg == "-m" || arg == "--engine") && i + 1 < argc) {
            ENGINE_PATH = argv[++i];
        } else if (arg == "--save-all") {
            SAVE_ALL = true;
        } else if (arg == "--save-detected") {
            SAVE_DETECTED = true;
        } else if ((arg == "-c" || arg == "--conf") && i + 1 < argc) {
            CONF_THRESHOLD = std::atof(argv[++i]);
        } else if (arg == "--help") {
            std::cout << "Usage: " << argv[0] << " [options]\n"
                      << "Options:\n"
                      << "  -w, --width <pixels>   Composited frame width (default: 3840)\n"
                      << "  --height <pixels>      Composited frame height (default: 1200)\n"
                      << "  -f, --fps <rate>       Output FPS after videorate (default: 20)\n"
                      << "  --sensor0 <id>         First CSI sensor id (default: 0)\n"
                      << "  --sensor1 <id>         Second CSI sensor id (default: 1)\n"
                      << "  --sensor-width <px>    Per-sensor capture width (default: 1920)\n"
                      << "  --sensor-height <px>   Per-sensor capture height (default: 1200)\n"
                      << "  --sensor-fps <rate>    Per-sensor capture FPS before videorate (default: 60)\n"
                      << "  -m, --engine <path>    TensorRT engine path (default: best.engine)\n"
                      << "  --save-all             Save every processed frame to disk\n"
                      << "  --save-detected        Save only frames containing detections to disk\n"
                      << "  -c, --conf <threshold> Confidence threshold (default: 0.45)\n"
                      << "  --help                 Show this help message\n"
                      << "Examples:\n"
                      << "  " << argv[0] << "\n"
                      << "  " << argv[0] << " --sensor0 0 --sensor1 1 --engine best.engine\n";
            return 0;
        }
    }

    // Initialize ROS2
    rclcpp::init(argc, argv);
    auto node = std::make_shared<rclcpp::Node>("yolo_detector_node");

    // Initialize session folder for saving frames
    SESSION_FOLDER = get_session_folder();
    if (SAVE_ALL || SAVE_DETECTED) {
        std::filesystem::create_directories(SESSION_FOLDER + "/forward");
        std::filesystem::create_directories(SESSION_FOLDER + "/down");
        std::cout << "[MAIN] Created session directories under: " << SESSION_FOLDER << std::endl;
    }

    std::cout << "========================================" << std::endl;
    std::cout << "  Multi-Threaded YOLO TensorRT + ROS2 System  " << std::endl;
    std::cout << "========================================" << std::endl;
    std::cout << "Camera: MIPI/CSI sensors " << CSI_SENSOR_0 << "+" << CSI_SENSOR_1 << " | ";
    std::cout << CAP_WIDTH << "x" << CAP_HEIGHT << " @ " << FPS << " FPS" << std::endl;
    std::cout << "Engine: " << ENGINE_PATH << std::endl;
    std::cout << "========================================" << std::endl;

    auto pose_estimator = std::make_shared<GroundTargetPoseEstimator>(node);

    // Shared data structures
    SharedFrameBuffer frame_buffer;
    DetectionQueue detection_queue;
    std::atomic<bool> running{true};

    // Spawn threads
    std::thread producer(producer_thread, std::ref(frame_buffer), std::ref(running));
    std::thread consumer(consumer_thread, std::ref(frame_buffer), std::ref(detection_queue), std::ref(running));
    std::thread communicator(communicator_thread, std::ref(detection_queue), std::ref(running), node, pose_estimator);

    std::cout << "[MAIN] All threads started. System running..." << std::endl;
    std::cout << "[MAIN] Press Ctrl+C to stop." << std::endl;

    // Spin ROS2 (handles callbacks, shutdown signals)
    rclcpp::spin(node);

    // Shutdown
    std::cout << "\n[MAIN] Shutdown signal received. Stopping threads..." << std::endl;
    running = false;
    frame_buffer.notify_shutdown();

    producer.join();
    consumer.join();
    communicator.join();

    rclcpp::shutdown();
    std::cout << "[MAIN] Shutdown complete." << std::endl;
    return 0;
}
