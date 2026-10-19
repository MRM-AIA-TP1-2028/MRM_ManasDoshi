import cv2
import numpy as np
import os

script_dir = os.path.dirname(os.path.abspath(__file__))
calibration_file = os.path.join(script_dir, "camera_calibration.yaml")
if not os.path.exists(calibration_file):
    print(f"Error: '{calibration_file}' not found! Please run the calibration script first.")
    exit()

cv_file = cv2.FileStorage(calibration_file, cv2.FILE_STORAGE_READ)
mtx = cv_file.getNode("camera_matrix").mat()
dist = cv_file.getNode("distortion_coefficients").mat()
cv_file.release()

focal_length = (mtx[0, 0] + mtx[1, 1]) / 2.0
TENNIS_BALL_RADIUS_CM = 3.35

cap = cv2.VideoCapture(0)

lower_blue = np.array([90, 50, 50])
upper_blue = np.array([130, 255, 255])

lower_tennis = np.array([30, 50, 50])
upper_tennis = np.array([85, 255, 255])

print("=== MULTI-OBJECT REAL-WORLD AREA & DEPTH TRACKER ===")
print("Press 'q' to quit.")

while True:
    ret, frame = cap.read()
    if not ret:
        break

    h, w = frame.shape[:2]
    newcameramtx, roi = cv2.getOptimalNewCameraMatrix(mtx, dist, (w, h), 1, (w, h))
    undistorted_frame = cv2.undistort(frame, mtx, dist, None, newcameramtx)

    hsv = cv2.cvtColor(undistorted_frame, cv2.COLOR_BGR2HSV)

    tennis_mask = cv2.inRange(hsv, lower_tennis, upper_tennis)
    tennis_mask = cv2.erode(tennis_mask, None, iterations=2)
    tennis_mask = cv2.dilate(tennis_mask, None, iterations=2)

    tennis_contours, _ = cv2.findContours(tennis_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    tennis_objects = []

    for c_tennis in tennis_contours:
        ((x_t, y_t), radius) = cv2.minEnclosingCircle(c_tennis)
        
        if radius > 10:  
            center = (int(x_t), int(y_t))
            r = int(radius)
            
            depth_cm = (TENNIS_BALL_RADIUS_CM * focal_length) / r
            scale_cm_per_px = depth_cm / focal_length
            real_area = np.pi * (TENNIS_BALL_RADIUS_CM ** 2)

            tennis_objects.append({
                "center": center,
                "radius": r,
                "depth": depth_cm,
                "scale": scale_cm_per_px
            })

            cv2.circle(undistorted_frame, center, r, (0, 255, 255), 2)
            cv2.circle(undistorted_frame, center, 5, (0, 0, 255), -1)

            ball_text = f"Ball | Dist:{depth_cm:.1f}cm R:{r}px"
            cv2.putText(undistorted_frame, ball_text, (center[0] - 60, center[1] - r - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 255, 255), 2)

    blue_mask = cv2.inRange(hsv, lower_blue, upper_blue)
    blue_mask = cv2.erode(blue_mask, None, iterations=2)
    blue_mask = cv2.dilate(blue_mask, None, iterations=2)

    blue_contours, _ = cv2.findContours(blue_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    blue_count = 0
    for c_blue in blue_contours:
        blue_area_px = cv2.contourArea(c_blue)

        if blue_area_px > 500:
            blue_count += 1
            x, y, bw, bh = cv2.boundingRect(c_blue)
            cv2.rectangle(undistorted_frame, (x, y), (x + bw, y + bh), (255, 0, 0), 2)

            M = cv2.moments(c_blue)
            if M["m00"] > 0:
                blue_center = (int(M["m10"] / M["m00"]), int(M["m01"] / M["m00"]))
                cv2.circle(undistorted_frame, blue_center, 4, (255, 0, 0), -1)

                closest_ball = None
                if tennis_objects:
                    closest_ball = min(tennis_objects, key=lambda b: np.hypot(b["center"][0] - blue_center[0], b["center"][1] - blue_center[1]))

                if closest_ball is not None:
                    scale = closest_ball["scale"]
                    real_blue_area_cm2 = blue_area_px * (scale ** 2)
                    blue_text = f"Blue #{blue_count}: {real_blue_area_cm2:.1f}cm^2"
                    
                    cv2.line(undistorted_frame, blue_center, closest_ball["center"], (0, 255, 0), 2)
                    dx = closest_ball["center"][0] - blue_center[0]
                    dy = closest_ball["center"][1] - blue_center[1]
                    dx_cm = dx * scale
                    dy_cm = dy * scale
                    
                    rel_text = f"dX:{dx_cm:.1f}cm, dY:{dy_cm:.1f}cm"
                    cv2.putText(undistorted_frame, rel_text, ((blue_center[0] + closest_ball["center"][0]) // 2, 
                                                              (blue_center[1] + closest_ball["center"][1]) // 2),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 255, 0), 2)
                else:
                    blue_text = f"Blue #{blue_count}: {int(blue_area_px)}px^2 (No Ref)"

                cv2.putText(undistorted_frame, blue_text, (x, y - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)

    cv2.imshow("Multi-Object Real-World Tracker", undistorted_frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()