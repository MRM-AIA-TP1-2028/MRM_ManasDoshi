from collections import deque
import os
import cv2
import numpy as np

script_dir = os.path.dirname(os.path.abspath(__file__))
calibration_file = os.path.join(script_dir, "camera_calibration.yaml")
if not os.path.exists(calibration_file):
    print(
        f"Error: '{calibration_file}' not found! Please run the calibration"
        " script first."
    )
    exit()

cv_file = cv2.FileStorage(calibration_file, cv2.FILE_STORAGE_READ)
mtx = cv_file.getNode("camera_matrix").mat()
dist = cv_file.getNode("distortion_coefficients").mat()
cv_file.release()

cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

ret, frame = cap.read()
if not ret:
    cap.release()
    exit()

h, w = frame.shape[:2]
newcameramtx, roi = cv2.getOptimalNewCameraMatrix(mtx, dist, (w, h), 1, (w, h))
map1, map2 = cv2.initUndistortRectifyMap(
    mtx, dist, None, newcameramtx, (w, h), cv2.CV_16SC2
)

lower_blue = np.array([100, 100, 50])
upper_blue = np.array([130, 255, 255])

lower_tennis = np.array([20, 100, 100])
upper_tennis = np.array([38, 255, 255])

kernel = np.ones((5, 5), np.uint8)
PIXEL_TO_CM2 = 0.05

active_blue_objects = {}
active_tennis_objects = {}
next_blue_id = 0
next_tennis_id = 0
MAX_DISAPPEARED = 15

def get_unique_color(obj_id, is_tennis=False):
    np.random.seed(obj_id * 37 + (150 if is_tennis else 10))
    color = np.random.randint(50, 255, size=3).tolist()
    return (int(color[0]), int(color[1]), int(color[2]))

print("=== MULTI-OBJECT UNIQUE COLOR TRACKER ===")
print("Press 'q' to quit.")

while True:
    ret, frame = cap.read()
    if not ret:
        break

    undistorted_frame = cv2.remap(frame, map1, map2, cv2.INTER_LINEAR)
    blurred = cv2.GaussianBlur(undistorted_frame, (5, 5), 0)
    hsv = cv2.cvtColor(blurred, cv2.COLOR_BGR2HSV)

    blue_mask = cv2.inRange(hsv, lower_blue, upper_blue)
    blue_mask = cv2.morphologyEx(blue_mask, cv2.MORPH_OPEN, kernel, iterations=2)
    blue_mask = cv2.morphologyEx(blue_mask, cv2.MORPH_CLOSE, kernel, iterations=1)

    # Stronger morphological closing for tennis ball to fuse internal fragments
    tennis_mask = cv2.inRange(hsv, lower_tennis, upper_tennis)
    tennis_mask = cv2.morphologyEx(tennis_mask, cv2.MORPH_OPEN, kernel, iterations=2)
    tennis_mask = cv2.morphologyEx(tennis_mask, cv2.MORPH_CLOSE, kernel, iterations=3)

    blue_contours, _ = cv2.findContours(blue_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    tennis_contours, _ = cv2.findContours(tennis_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    black_canvas = np.zeros((h, w, 3), dtype=np.uint8)

    # --- PROCESS BLUE OBJECTS ---
    current_blue_detections = []
    for c in blue_contours:
        area_px = cv2.contourArea(c)
        if area_px > 400:
            M = cv2.moments(c)
            if M["m00"] > 0:
                cx = int(M["m10"] / M["m00"])
                cy = int(M["m01"] / M["m00"])
                current_blue_detections.append({"center": (cx, cy), "area": area_px})

    updated_blue_objects = {}
    for det in current_blue_detections:
        matched_id = None
        min_dist = float("inf")
        for obj_id, data in active_blue_objects.items():
            dist = np.hypot(data["center"][0] - det["center"][0], data["center"][1] - det["center"][1])
            if dist < min_dist and dist < 80:
                min_dist = dist
                matched_id = obj_id

        if matched_id is not None:
            prev_cx, prev_cy = active_blue_objects[matched_id]["center"]
            smooth_cx = int(0.6 * prev_cx + 0.4 * det["center"][0])
            smooth_cy = int(0.6 * prev_cy + 0.4 * det["center"][1])
            smooth_center = (smooth_cx, smooth_cy)

            q = active_blue_objects[matched_id]["deque"]
            q.appendleft(smooth_center)

            updated_blue_objects[matched_id] = {
                "center": smooth_center,
                "deque": q,
                "area": det["area"],
                "disappeared": 0,
                "color": active_blue_objects[matched_id]["color"]
            }
            active_blue_objects.pop(matched_id)
        else:
            q = deque(maxlen=32)
            q.appendleft(det["center"])
            assigned_color = get_unique_color(next_blue_id, is_tennis=False)
            updated_blue_objects[next_blue_id] = {
                "center": det["center"],
                "deque": q,
                "area": det["area"],
                "disappeared": 0,
                "color": assigned_color
            }
            next_blue_id += 1

    for obj_id, data in active_blue_objects.items():
        data["disappeared"] += 1
        if data["disappeared"] <= MAX_DISAPPEARED:
            updated_blue_objects[obj_id] = data
    active_blue_objects = updated_blue_objects

    # --- PROCESS TENNIS BALLS WITH NON-MAXIMUM SUPPRESSION (NMS) ---
    raw_tennis_detections = []
    for c in tennis_contours:
        ((x_t, y_t), radius) = cv2.minEnclosingCircle(c)
        if radius > 12:  # Raised minimum size filter to ignore speckles
            raw_tennis_detections.append({"center": (int(x_t), int(y_t)), "radius": int(radius)})

    # Sort detections by radius descending so largest/most complete circles take priority
    raw_tennis_detections = sorted(raw_tennis_detections, key=lambda x: x["radius"], reverse=True)
    
    current_tennis_detections = []
    for det in raw_tennis_detections:
        # Check if this detection is too close to an already accepted detection (merging duplicates)
        is_duplicate = False
        for accepted in current_tennis_detections:
            dist = np.hypot(accepted["center"][0] - det["center"][0], accepted["center"][1] - det["center"][1])
            if dist < (accepted["radius"] + det["radius"]) * 0.8:
                is_duplicate = True
                break
        if not is_duplicate:
            current_tennis_detections.append(det)

    updated_tennis_objects = {}
    for det in current_tennis_detections:
        matched_id = None
        min_dist = float("inf")
        for obj_id, data in active_tennis_objects.items():
            dist = np.hypot(data["center"][0] - det["center"][0], data["center"][1] - det["center"][1])
            if dist < min_dist and dist < 80:
                min_dist = dist
                matched_id = obj_id

        if matched_id is not None:
            prev_cx, prev_cy = active_tennis_objects[matched_id]["center"]
            smooth_cx = int(0.6 * prev_cx + 0.4 * det["center"][0])
            smooth_cy = int(0.6 * prev_cy + 0.4 * det["center"][1])
            smooth_center = (smooth_cx, smooth_cy)

            q = active_tennis_objects[matched_id]["deque"]
            q.appendleft(smooth_center)

            updated_tennis_objects[matched_id] = {
                "center": smooth_center,
                "deque": q,
                "radius": det["radius"],
                "disappeared": 0,
                "color": active_tennis_objects[matched_id]["color"]
            }
            active_tennis_objects.pop(matched_id)
        else:
            q = deque(maxlen=32)
            q.appendleft(det["center"])
            assigned_color = get_unique_color(next_tennis_id, is_tennis=True)
            updated_tennis_objects[next_tennis_id] = {
                "center": det["center"],
                "deque": q,
                "radius": det["radius"],
                "disappeared": 0,
                "color": assigned_color
            }
            next_tennis_id += 1

    for obj_id, data in active_tennis_objects.items():
        data["disappeared"] += 1
        if data["disappeared"] <= MAX_DISAPPEARED:
            updated_tennis_objects[obj_id] = data
    active_tennis_objects = updated_tennis_objects

    # --- DRAWING & METRICS ---
    primary_blue_center = None
    primary_tennis_center = None

    for obj_id, data in active_blue_objects.items():
        if data["disappeared"] == 0:
            center = data["center"]
            area_px = data["area"]
            area_cm2 = area_px * PIXEL_TO_CM2
            obj_color = data["color"]

            if primary_blue_center is None:
                primary_blue_center = center

            cv2.circle(undistorted_frame, center, 5, obj_color, -1)
            text = f"Blue #{obj_id}: {int(area_px)}px^2 ({area_cm2:.1f}cm^2)"
            cv2.putText(undistorted_frame, text, (center[0] - 30, center[1] - 15),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, obj_color, 2)

        pts = data["deque"]
        obj_color = data["color"]
        for i in range(1, len(pts)):
            if pts[i - 1] is not None and pts[i] is not None:
                cv2.line(black_canvas, pts[i - 1], pts[i], obj_color, 2)

    for obj_id, data in active_tennis_objects.items():
        if data["disappeared"] == 0:
            center = data["center"]
            radius = data["radius"]
            area_px = np.pi * (radius ** 2)
            area_cm2 = area_px * PIXEL_TO_CM2
            obj_color = data["color"]

            if primary_tennis_center is None:
                primary_tennis_center = center

            cv2.circle(undistorted_frame, center, radius, obj_color, 2)
            cv2.circle(undistorted_frame, center, 5, obj_color, -1)

            metrics_text = f"Ball #{obj_id} | C:{center} | R:{radius}px | Area:{int(area_px)}px^2 ({area_cm2:.1f}cm^2)"
            cv2.putText(undistorted_frame, metrics_text, (20, 30 + (obj_id * 25)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, obj_color, 2)

        pts = data["deque"]
        obj_color = data["color"]
        for i in range(1, len(pts)):
            if pts[i - 1] is not None and pts[i] is not None:
                cv2.line(black_canvas, pts[i - 1], pts[i], obj_color, 2)

    if primary_blue_center and primary_tennis_center:
        cv2.line(undistorted_frame, primary_blue_center, primary_tennis_center, (0, 255, 0), 2)
        dx = primary_tennis_center[0] - primary_blue_center[0]
        dy = primary_tennis_center[1] - primary_blue_center[1]
        rel_text = f"Rel Pos -> dX: {dx}px, dY: {dy}px"
        cv2.putText(undistorted_frame, rel_text, (20, h - 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

    cv2.imshow("Blue Object & Tennis Ball Tracker", undistorted_frame)
    cv2.imshow("Centroid Tracking (Pitch Black)", black_canvas)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()