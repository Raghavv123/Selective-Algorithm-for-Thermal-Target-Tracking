#include <opencv2/opencv.hpp>
#include <iostream>
#include <vector>
#include <algorithm>
#include <cmath>
#include <string>

const float FOCAL_LENGTH_MM = 55.0f;
const float PIXEL_PITCH_MM  = 0.012f;

float       g_smoothed_range = 0.0f;
const float g_alpha          = 0.15f;

int g_threshold              = 40;
int g_kernel                 = 12;   
int g_min_area               = 5;    
int g_max_area               = 1000;  
int g_min_extent_px          = 2;    
int g_max_extent_px          = 80;   
int g_min_fill_pct           = 12;   
int g_centroid_threshold_pct = 30;
int g_min_centroid_pixels    = 1;

bool        g_has_lock           = false;
cv::Point2f g_last_centroid(0.f, 0.f);
cv::Point2f g_velocity(0.f, 0.f);
int         g_miss_count         = 0;
int         g_miss_tolerance     = 15;   
float       g_base_jump_distance = 100.0f; 
float       g_jump_velocity_gain = 2.0f;

cv::Mat   g_bg_model;
bool      g_bg_ready  = false;
int       g_bg_frames = 0;
const int BG_INIT_FRAMES  = 25;
const float BG_LEARN_RATE = 0.0008f;

const std::string WIN_MAIN  = "LWIR Tracker - Dual View (Raw vs. Selective Engine)";
const std::string WIN_DEBUG = "Debug - Local Contrast Map";

int odd(int v) { return (v % 2 == 0) ? v + 1 : v; }

cv::Point2f weighted_centroid(const cv::Mat& contrast, const cv::Rect& roi)
{
    cv::Rect safe = roi & cv::Rect(0, 0, contrast.cols, contrast.rows);
    if (safe.empty())
        return cv::Point2f(roi.x + roi.width/2.f, roi.y + roi.height/2.f);

    cv::Mat patch = contrast(safe);
    double minv, maxv;
    cv::minMaxLoc(patch, &minv, &maxv);
    double cutoff = maxv * (g_centroid_threshold_pct / 100.0);

    double sw=0, swx=0, swy=0;
    int qpx = 0;
    for (int r = 0; r < patch.rows; r++) {
        const uchar* row_ptr = patch.ptr<uchar>(r);
        for (int c = 0; c < patch.cols; c++) {
            double v = row_ptr[c];
            if (v > cutoff) {
                sw  += v;
                swx += v * (c + safe.x);
                swy += v * (r + safe.y);
                qpx++;
            }
        }
    }
    if (sw < 1e-6 || qpx < g_min_centroid_pixels)
        return cv::Point2f(roi.x + roi.width/2.f, roi.y + roi.height/2.f);
    return cv::Point2f((float)(swx/sw), (float)(swy/sw));
}

struct Target {
    cv::Rect    bbox;
    cv::Point2f centroid;
    double      score;
    int         area;
    double      fill_ratio;
    std::string type;
    float       estimated_range_m;
    float       pixel_speed;
};

double calculate_confidence(const Target& t)
{
    double size_score = 1.0;
    if (t.area > 150) size_score = 0.2;
    else if (t.area < 4) size_score = 0.3;

    double ar = (double)t.bbox.width / t.bbox.height;
    double ar_score = 1.0 - std::abs(ar - 1.5) / 2.0;
    ar_score = std::max(0.1, ar_score);

    double fill_score = 1.0;
    if (t.fill_ratio < 0.12 || t.fill_ratio > 0.60) {
        fill_score = 0.3;
    }

    return t.score * size_score * ar_score * fill_score;
}

float calculate_estimated_range(float bw_px, const std::string& type)
{
    if (bw_px <= 0) return 0.0f;

    float expected_width_m = 1.0f;
    if (type == "DRONE")            expected_width_m = 0.4f;
    if (type == "BIRD")             expected_width_m = 0.3f;
    if (type == "FIXED-WING")       expected_width_m = 4.0f;
    if (type == "HELICOPTER")       expected_width_m = 12.0f;
    if (type == "AIRCRAFT_DISTANT") expected_width_m = 3.8f;

    float sensor_width_mm = bw_px * PIXEL_PITCH_MM;
    float range_m = (expected_width_m * FOCAL_LENGTH_MM) / sensor_width_mm;
    return range_m;
}

std::string classify_target_coarse(float bw_px, float bh_px, int area, double fill_ratio, float pixel_speed)
{
    float ar = bw_px / bh_px;

    if (ar > 5.0f || ar < 0.18f) return "BIRD";
    if (area < 35) {
        if (pixel_speed > 4.5f) return "DRONE";
        return "AIRCRAFT_DISTANT";
    }
    if (area > 350) {
        if (fill_ratio < 0.38 || ar < 0.5f || ar > 4.0f) return "BIRD";
        return "HELICOPTER";
    }
    if (ar >= 1.5f) return "FIXED-WING";
    return "HELICOPTER";
}

std::vector<Target> detect(const cv::Mat& gray_residual, cv::Mat& contrast_out)
{
    cv::Mat smooth;
    cv::GaussianBlur(gray_residual, smooth, cv::Size(3,3), 0.8);

    int k = odd(g_kernel);
    cv::Mat se = cv::getStructuringElement(cv::MORPH_ELLIPSE, cv::Size(k,k));

    cv::Mat white_hot, black_hot;
    cv::morphologyEx(smooth, white_hot, cv::MORPH_TOPHAT, se);
    cv::morphologyEx(smooth, black_hot, cv::MORPH_BLACKHAT, se);

    cv::Mat local_contrast = cv::max(white_hot, black_hot);
    contrast_out = local_contrast.clone();

    cv::Mat binary;
    cv::threshold(local_contrast, binary, g_threshold, 255, cv::THRESH_BINARY);

    cv::Mat k3 = cv::getStructuringElement(cv::MORPH_ELLIPSE, cv::Size(3,3));
    cv::morphologyEx(binary, binary, cv::MORPH_CLOSE, k3);

    cv::Mat labels, stats, centroids_cc;
    int n = cv::connectedComponentsWithStats(binary, labels, stats, centroids_cc, 8);

    float current_speed = 0.0f;
    if (g_has_lock) {
        current_speed = std::hypot(g_velocity.x, g_velocity.y);
    }

    std::vector<Target> targets;
    for (int i = 1; i < n; i++) {
        int area = stats.at<int>(i, cv::CC_STAT_AREA);
        if (area < g_min_area || area > g_max_area) continue;

        int bx = stats.at<int>(i, cv::CC_STAT_LEFT);
        int by = stats.at<int>(i, cv::CC_STAT_TOP);
        int bw = stats.at<int>(i, cv::CC_STAT_WIDTH);
        int bh = stats.at<int>(i, cv::CC_STAT_HEIGHT);

        if (bw < g_min_extent_px || bw > g_max_extent_px) continue;
        if (bh < g_min_extent_px || bh > g_max_extent_px) continue;
        if (bh == 0) continue;

        double ar = (double)bw / bh;
        if (ar < 0.15 || ar > 5.5) continue;

        double bbox_area  = (double)bw * bh;
        double fill_ratio = (bbox_area > 0) ? (area / bbox_area) : 0.0;
        if (fill_ratio < (g_min_fill_pct / 100.0)) continue;

        std::string target_type = classify_target_coarse((float)bw, (float)bh, area, fill_ratio, current_speed);
        float range_m = calculate_estimated_range((float)bw, target_type);

        cv::Rect bbox(bx, by, bw, bh);
        cv::Point2f cx = weighted_centroid(local_contrast, bbox);

        double minv, maxv;
        cv::minMaxLoc(local_contrast(bbox), &minv, &maxv);

        targets.push_back({bbox, cx, maxv, area, fill_ratio, target_type, range_m, current_speed});
    }

    if (g_has_lock) {
        cv::Point2f predicted = g_last_centroid + g_velocity;
        std::sort(targets.begin(), targets.end(),
            [predicted](const Target& a, const Target& b){
                float da = std::hypot(a.centroid.x - predicted.x, a.centroid.y - predicted.y);
                float db = std::hypot(b.centroid.x - predicted.x, b.centroid.y - predicted.y);
                return da < db;
            });
    } else {
        std::sort(targets.begin(), targets.end(),
            [](const Target& a, const Target& b){
                return calculate_confidence(a) > calculate_confidence(b);
            });
    }

    return targets;
}

void update_lock_state(const std::vector<Target>& targets)
{
    if (targets.empty()) {
        g_miss_count++;
        g_velocity *= 0.90f;
        if (g_miss_count > g_miss_tolerance) {
            g_has_lock = false;
            g_velocity = cv::Point2f(0.f, 0.f);
        }
        return;
    }

    if (!g_has_lock) {
        g_has_lock      = true;
        g_last_centroid = targets[0].centroid;
        g_velocity      = cv::Point2f(0.f, 0.f);
        g_miss_count    = 0;
        return;
    }

    float current_speed = std::hypot(g_velocity.x, g_velocity.y);
    float max_jump = g_base_jump_distance + g_jump_velocity_gain * current_speed;

    cv::Point2f predicted = g_last_centroid + g_velocity;
    float jump = std::hypot(targets[0].centroid.x - predicted.x, targets[0].centroid.y - predicted.y);

    if (jump < max_jump) {
        cv::Point2f new_vel = targets[0].centroid - g_last_centroid;
        g_velocity      = 0.20f * new_vel + 0.80f * g_velocity;
        g_last_centroid = targets[0].centroid;
        g_miss_count    = 0;
    } else {
        g_miss_count++;
        g_velocity *= 0.90f;
        if (g_miss_count > g_miss_tolerance) {
            g_has_lock = false;
            g_velocity = cv::Point2f(0.f, 0.f);
        }
    }
}

void reset_state()
{
    g_bg_model  = cv::Mat();
    g_bg_ready  = false;
    g_bg_frames = 0;
    g_has_lock  = false;
    g_velocity  = cv::Point2f(0.f, 0.f);
    g_miss_count = 0;
    g_smoothed_range = 0.0f;
    std::cout << "[INFO] Tracking Engine Vector Reset.\n";
}

void draw(cv::Mat& display, const std::vector<Target>& targets,
          int frame_idx, int total_frames, int W, int H)
{
    for (size_t i = 0; i < targets.size(); i++) {
        const Target& t = targets[i];
        bool is_best = (i == 0);

        cv::Scalar col_box   = is_best ? cv::Scalar(0,255,80)  : cv::Scalar(0,180,180);
        cv::Scalar col_cross = is_best ? cv::Scalar(0,0,255)    : cv::Scalar(0,140,140);

        std::string lbl;
        if (is_best) {
            lbl = "LOCKED: " + t.type;
            if (t.estimated_range_m > 0.1f) {
                lbl += " [" + std::to_string((int)t.estimated_range_m) + "m]";
            }
        } else {
            lbl = "candidate";
        }

        cv::rectangle(display, t.bbox, col_box, 2);

        int arm = 16;
        cv::Point pt((int)t.centroid.x, (int)t.centroid.y);
        cv::line(display, {pt.x-arm,pt.y}, {pt.x-5,  pt.y}, col_box, 1);
        cv::line(display, {pt.x+5,  pt.y}, {pt.x+arm,pt.y}, col_box, 1);
        cv::line(display, {pt.x,pt.y-arm}, {pt.x,pt.y-5  }, col_box, 1);
        cv::line(display, {pt.x,pt.y+5  }, {pt.x,pt.y+arm}, col_box, 1);
        cv::circle(display, pt, 3, col_cross, -1);

        if (is_best) {
            int roi_sz = 240;
            cv::Rect roi(pt.x-roi_sz/2, pt.y-roi_sz/2, roi_sz, roi_sz);
            roi &= cv::Rect(0,0,W,H);
            cv::rectangle(display, roi, cv::Scalar(255,180,0), 2);
        }

        cv::putText(display, lbl, {t.bbox.x, t.bbox.y-8},
                    cv::FONT_HERSHEY_SIMPLEX, 0.48, col_box, 1);
        std::string info = std::to_string(t.bbox.width) + "x" + std::to_string(t.bbox.height) + "px area=" + std::to_string(t.area);
        cv::putText(display, info, {t.bbox.x, t.bbox.y+t.bbox.height+14},
                    cv::FONT_HERSHEY_SIMPLEX, 0.34, col_box, 1);
    }

    std::string frame_str = "Frame:" + std::to_string(frame_idx) + "/" + std::to_string(total_frames);
    cv::putText(display, frame_str, {8,22}, cv::FONT_HERSHEY_SIMPLEX, 0.5, {180,180,180}, 1);

    std::string status = g_has_lock ? "LOCKED" : "SEARCHING...";
    cv::Scalar  scol   = g_has_lock ? cv::Scalar(0,255,80) : cv::Scalar(0,160,255);
    cv::putText(display, status, {W-170,26}, cv::FONT_HERSHEY_SIMPLEX, 0.7, scol, 2);

    if (!g_bg_ready) {
        std::string bg_msg = "Building background model... " + std::to_string(g_bg_frames) + "/" + std::to_string(BG_INIT_FRAMES);
        cv::putText(display, bg_msg, {W/2-220, H/2}, cv::FONT_HERSHEY_SIMPLEX, 0.7, {0,255,255}, 2);
    }
}

int main(int argc, char** argv)
{
    int choice = 0;
    std::cout << "=======================================\n";
    std::cout << "         LWIR TRACKER         \n";
    std::cout << "=======================================\n";
    std::cout << "[1] Use Live External LWIR Camera Feed\n";
    std::cout << "[2] Use Pre-recorded Video File\n";
    std::cout << "---------------------------------------\n";
    std::cout << "Enter choice (1 or 2): ";
    
    if (!(std::cin >> choice)) {
        std::cin.clear();
        choice = 0;
    }
    
    std::cin.ignore(std::numeric_limits<std::streamsize>::max(), '\n');

    cv::VideoCapture cap;
    std::string out_path;

    if (choice == 1) {
        std::cout << "\n[INFO] Opening external camera (index 1)... Please wait.\n";
        
        cap.open(1, cv::CAP_DSHOW); 
        if (!cap.isOpened()) cap.open(1, cv::CAP_ANY); 
        if (!cap.isOpened()) cap.open(2, cv::CAP_DSHOW); 

        out_path = "live_capture_tracked.mp4";
    } 
    else if (choice == 2) {
        std::cout << "\n[INFO] Ensure your video is in the same folder as this program.\n";
        std::cout << "Enter the video file name (e.g., drone.mp4): ";
        
        std::string video_path;
        std::getline(std::cin, video_path);
        
        if (!video_path.empty() && video_path.front() == '"' && video_path.back() == '"') {
            video_path = video_path.substr(1, video_path.size() - 2);
        }
        
        cap.open(video_path);
        size_t dot = video_path.find_last_of('.');
        std::string stem = (dot == std::string::npos) ? "video" : video_path.substr(0, dot);
        out_path = stem + "_tracked.mp4";
    } 
    else {
        std::cerr << "\n[ERROR] Invalid choice selected.\n";
        std::cout << "Press Enter to exit...";
        std::cin.get();
        return -1;
    }

    if (!cap.isOpened()) {
        std::cerr << "\n[ERROR] Failed to open the video feed or camera!\n";
        std::cerr << "Check your camera connection or verify that the filename is typed correctly (including the .mp4 part).\n";
        std::cout << "\nPress Enter to exit...";
        std::cin.get(); 
        return -1;
    }

    int W           = (int)cap.get(cv::CAP_PROP_FRAME_WIDTH);
    int H           = (int)cap.get(cv::CAP_PROP_FRAME_HEIGHT);
    int total       = (int)cap.get(cv::CAP_PROP_FRAME_COUNT);
    double src_fps  = cap.get(cv::CAP_PROP_FPS);
    
    if (src_fps <= 1.0) src_fps = 25.0; 
    
    int frame_idx   = 0;
    bool paused     = false;

    // Set VideoWriter to record side-by-side view (2 * W)
    int fourcc = cv::VideoWriter::fourcc('m','p','4','v');
    cv::VideoWriter writer(out_path, fourcc, src_fps, cv::Size(2 * W, H), true);
    if (!writer.isOpened()) {
        std::cerr << "[WARN] Could not open VideoWriter for: " << out_path << ". Falling back to .avi\n";
        out_path = out_path.substr(0, out_path.find_last_of('.')) + ".avi";
        fourcc = cv::VideoWriter::fourcc('X','V','I','D');
        writer.open(out_path, fourcc, src_fps, cv::Size(2 * W, H), true);
    }
    if (writer.isOpened())
        std::cout << "[INFO] Saving annotated output to: " << out_path << "\n";

    cv::namedWindow(WIN_MAIN,  cv::WINDOW_NORMAL);
    cv::namedWindow(WIN_DEBUG, cv::WINDOW_NORMAL);
    
    // Fit dual window cleanly on display
    cv::resizeWindow(WIN_MAIN,  std::min(2 * W, 1600), std::min(H, 800));
    cv::resizeWindow(WIN_DEBUG, std::min(W/2, 600), std::min(H/2, 450));

    cv::createTrackbar("Threshold",     WIN_MAIN, &g_threshold,   100, nullptr);
    cv::createTrackbar("Kernel Size",   WIN_MAIN, &g_kernel,      40,  nullptr);
    cv::createTrackbar("Min area",      WIN_MAIN, &g_min_area,    50,  nullptr);
    cv::createTrackbar("Max area",      WIN_MAIN, &g_max_area,    2000,nullptr);

    cv::Mat frame, gray, display, contrast;

    while (true) {
        if (!paused) {
            cap >> frame;
            if (frame.empty()) break;
            frame_idx++;

            if (frame.channels() == 3)
                cv::cvtColor(frame, gray, cv::COLOR_BGR2GRAY);
            else
                gray = frame.clone();

            if (!g_bg_ready) {
                cv::Mat gf; gray.convertTo(gf, CV_32F);
                if (g_bg_model.empty()) g_bg_model = gf.clone();
                else cv::accumulateWeighted(gf, g_bg_model, 0.5f);
                g_bg_frames++;
                if (g_bg_frames >= BG_INIT_FRAMES) g_bg_ready = true;

                cv::Mat disp_init;
                cv::cvtColor(gray, disp_init, cv::COLOR_GRAY2BGR);
                draw(disp_init, {}, frame_idx, total, W, H);

                // Prepare Dual Frame View during background build
                cv::Mat raw_init = disp_init.clone();
                cv::putText(raw_init, "RAW INPUT FEED", {15, 30}, cv::FONT_HERSHEY_SIMPLEX, 0.6, {255,255,255}, 2);
                cv::putText(disp_init, "SELECTIVE ALGORITHM HUD", {15, 30}, cv::FONT_HERSHEY_SIMPLEX, 0.6, {0,255,0}, 2);

                cv::Mat side_by_side_init;
                cv::hconcat(raw_init, disp_init, side_by_side_init);

                cv::imshow(WIN_MAIN, side_by_side_init);
                if ((cv::waitKey(1) & 0xFF) == 'q') break;
                continue;
            }

            cv::Mat gf; gray.convertTo(gf, CV_32F);
            cv::accumulateWeighted(gf, g_bg_model, BG_LEARN_RATE);

            cv::Mat bg_8u, residual;
            g_bg_model.convertTo(bg_8u, CV_8U);
            
            cv::absdiff(gray, bg_8u, residual);

            std::vector<Target> targets = detect(residual, contrast);
            update_lock_state(targets);

            if (g_has_lock && !targets.empty()) {
                if (g_smoothed_range == 0.0f) g_smoothed_range = targets[0].estimated_range_m;
                else g_smoothed_range = (g_alpha * targets[0].estimated_range_m) + ((1.0f - g_alpha) * g_smoothed_range);
                targets[0].estimated_range_m = g_smoothed_range;
            } else {
                g_smoothed_range = 0.0f;
            }

            // --- 1. PREPARE RAW PRE-PROCESSED FEED (LEFT) ---
            cv::Mat eq, raw_display;
            cv::equalizeHist(gray, eq);
            cv::cvtColor(eq, raw_display, cv::COLOR_GRAY2BGR);
            cv::putText(raw_display, "RAW INPUT / PRE-PROCESSED", {15, 30}, 
                        cv::FONT_HERSHEY_SIMPLEX, 0.6, {255,255,255}, 2);

            // --- 2. PREPARE SELECTIVE HUD FEED (RIGHT) ---
            cv::cvtColor(eq, display, cv::COLOR_GRAY2BGR);
            draw(display, targets, frame_idx, total, W, H);
            cv::putText(display, "SELECTIVE ALGORITHM HUD", {15, 30}, 
                        cv::FONT_HERSHEY_SIMPLEX, 0.6, {0,255,0}, 2);

            // --- 3. CONCATENATE HORIZONTALLY SIDE-BY-SIDE ---
            cv::Mat side_by_side;
            cv::hconcat(raw_display, display, side_by_side);

            // Display and Record Side-by-Side Video
            cv::imshow(WIN_MAIN, side_by_side);

            if (writer.isOpened())
                writer.write(side_by_side);

            cv::Mat debug_col;
            cv::applyColorMap(contrast, debug_col, cv::COLORMAP_HOT);
            cv::imshow(WIN_DEBUG, debug_col);

            if (!targets.empty() && g_has_lock) {
                if (total > 0) {
                    std::cout << "Frame " << frame_idx << "/" << total
                              << "  Type: " << targets[0].type
                              << "  Est. Range: " << (int)targets[0].estimated_range_m << "m"
                              << "\r" << std::flush;
                } else {
                    std::cout << "Live Frame " << frame_idx
                              << "  Type: " << targets[0].type
                              << "  Est. Range: " << (int)targets[0].estimated_range_m << "m"
                              << "\r" << std::flush;
                }
            }
        }

        int key = cv::waitKey(paused ? 30 : 1) & 0xFF;
        if (key == 'q' || key == 27) break;
        if (key == ' ') paused = !paused;
        if (key == 'r') reset_state();
    }

    cap.release();
    if (writer.isOpened()) {
        writer.release();
        std::cout << "\n[INFO] Saved side-by-side recording to: " << out_path << "\n";
    }
    cv::destroyAllWindows();
    
    std::cout << "\nProgram finished. Press Enter to exit...";
    std::cin.get();
    
    return 0;
}