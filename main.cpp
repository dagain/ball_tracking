#include <opencv2/opencv.hpp>
#include <iostream>

int main(int argc, char** argv) {
    // Declare capture object
    cv::VideoCapture cap;

    // 1. Open video source: webcam (0) using the appropriate backend
    int camIndex = 0;
    if (argc > 1) {
        cap.open(argv[1]);               // open file or arbitrary pipeline
    } else {
    #ifdef _WIN32
        cap.open(camIndex, cv::CAP_DSHOW);
    #else
        cap.open(camIndex, cv::CAP_V4L2);
    #endif
    }
    if (!cap.isOpened()) {
        std::cerr << "ERROR: Could not open video source\n";
        return -1;
    }

    // 2. Limit resolution & frame-rate to reduce buffer pressure
    cap.set(cv::CAP_PROP_FRAME_WIDTH,  640);
    cap.set(cv::CAP_PROP_FRAME_HEIGHT, 480);
    cap.set(cv::CAP_PROP_FPS,          15);

    const float minRadiusThresh = 20.0f;  // ignore circles smaller than this

    cv::Mat frame, hsv, mask1, mask2, mask, cleaned;
    std::vector<std::vector<cv::Point>> contours;

    // Pre-create the morphological kernel once
    cv::Mat kernel = cv::getStructuringElement(
        cv::MORPH_ELLIPSE, cv::Size(5,5));

    while (true) {
        cap >> frame;
        if (frame.empty()) break;  // end of video / camera disconnected

        // 3. Convert to HSV & threshold red (two ranges)
        cv::cvtColor(frame, hsv, cv::COLOR_BGR2HSV);
        cv::inRange(hsv, cv::Scalar(0, 100, 100),
                         cv::Scalar(10, 255, 255), mask1);
        cv::inRange(hsv, cv::Scalar(160, 100, 100),
                         cv::Scalar(179, 255, 255), mask2);
        mask = mask1 | mask2;

        // 4. Morphological cleanup
        cv::morphologyEx(mask,   cleaned, cv::MORPH_OPEN,  kernel);
        cv::morphologyEx(cleaned, cleaned, cv::MORPH_CLOSE, kernel);

        // 5. Find contours
        contours.clear();
        cv::findContours(cleaned, contours,
                         cv::RETR_EXTERNAL, cv::CHAIN_APPROX_SIMPLE);

        // 6. Pick the largest contour by area
        double maxArea = 0;  int bestIdx = -1;
        for (int i = 0; i < (int)contours.size(); i++) {
            double a = cv::contourArea(contours[i]);
            if (a > maxArea) {
                maxArea = a;
                bestIdx = i;
            }
        }

        // 7. Compute enclosing circle & filter by size
        if (bestIdx >= 0) {
            cv::Point2f center;
            float       radius;
            cv::minEnclosingCircle(contours[bestIdx], center, radius);

            if (radius >= minRadiusThresh) {
                // draw circle and center
                cv::circle(frame, center, (int)radius,
                           cv::Scalar(0,255,0), 2);
                cv::circle(frame, center, 3,
                           cv::Scalar(0,0,255), -1);

                std::cout << "Ball @ ("
                          << int(center.x) << ", "
                          << int(center.y) << "), r="
                          << int(radius) << "    \r";
            } else {
                std::cout << "Noise detected (r="
                          << int(radius) << ")          \r";
            }
        } else {
            std::cout << "No red region found          \r";
        }

        // 8. Show result
        cv::imshow("Red-Ball Tracking", frame);

        // 9. Exit on ESC
        if (cv::waitKey(1) == 27) break;
    }

    cap.release();
    cv::destroyAllWindows();
    return 0;
}
