from collections import deque
import os
import cv2
import numpy as np
import math

script_dir = os.path.dirname(os.path.abspath(__file__))
calibration_file = os.path.join(script_dir, "camera_calibration.yaml")

FRAME_WIDTH = 1280
FRAME_HEIGHT = 720

MAX_TRACK_DISTANCE = 80
MAX_DISAPPEARED = 15
TRAIL_LENGTH = 32

MIN_BLUE_AREA = 400

MIN_BALL_AREA = 120
MIN_BALL_RADIUS = 10
MAX_BALL_RADIUS = 250
MIN_CIRCULARITY = 0.75

lower_blue = np.array([90, 65, 35], dtype=np.uint8)
upper_blue = np.array([135, 255, 255], dtype=np.uint8)

lower_yellow = np.array([18, 60, 50], dtype=np.uint8)
upper_yellow = np.array([40, 255, 255], dtype=np.uint8)

if not os.path.exists(calibration_file):
    raise FileNotFoundError(
        f"Calibration file not found: {calibration_file}"
    )

calibration = cv2.FileStorage(
    calibration_file,
    cv2.FILE_STORAGE_READ
)

if not calibration.isOpened():
    raise RuntimeError(
        "Could not open camera_calibration.yaml"
    )

camera_matrix = calibration.getNode("camera_matrix").mat()
dist_coeffs = calibration.getNode(
    "distortion_coefficients"
).mat()

calibration.release()

if camera_matrix is None:
    raise ValueError("camera_matrix is missing from the YAML file.")

if dist_coeffs is None:
    raise ValueError(
        "distortion_coefficients is missing from the YAML file."
    )

camera_matrix = np.asarray(camera_matrix, dtype=np.float64)
dist_coeffs = np.asarray(dist_coeffs, dtype=np.float64).reshape(-1, 1)

if camera_matrix.shape != (3, 3):
    raise ValueError("camera_matrix must be a 3x3 matrix.")

if not np.all(np.isfinite(camera_matrix)):
    raise ValueError("camera_matrix contains invalid values.")

if not np.all(np.isfinite(dist_coeffs)):
    raise ValueError("distortion_coefficients contains invalid values.")

if camera_matrix[0, 0] <= 0 or camera_matrix[1, 1] <= 0:
    raise ValueError("Camera focal lengths must be positive.")

cap = cv2.VideoCapture(0)

cap.set(cv2.CAP_PROP_FRAME_WIDTH, FRAME_WIDTH)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, FRAME_HEIGHT)

if not cap.isOpened():
    raise RuntimeError("Could not open webcam.")

ret, first_frame = cap.read()

if not ret:
    cap.release()
    raise RuntimeError("Could not read the first webcam frame.")

actual_height, actual_width = first_frame.shape[:2]

scale_x = actual_width / FRAME_WIDTH
scale_y = actual_height / FRAME_HEIGHT

scaled_camera_matrix = camera_matrix.copy()
scaled_camera_matrix[0, 0] *= scale_x
scaled_camera_matrix[0, 1] *= scale_x
scaled_camera_matrix[0, 2] *= scale_x
scaled_camera_matrix[1, 0] *= scale_y
scaled_camera_matrix[1, 1] *= scale_y
scaled_camera_matrix[1, 2] *= scale_y

new_camera_matrix, roi = cv2.getOptimalNewCameraMatrix(
    scaled_camera_matrix,
    dist_coeffs,
    (actual_width, actual_height),
    1,
    (actual_width, actual_height)
)

map_x, map_y = cv2.initUndistortRectifyMap(
    scaled_camera_matrix,
    dist_coeffs,
    None,
    new_camera_matrix,
    (actual_width, actual_height),
    cv2.CV_32FC1
)

kernel = np.ones((5, 5), np.uint8)

clahe = cv2.createCLAHE(
    clipLimit=2.0,
    tileGridSize=(8, 8)
)

active_blue_objects = {}
active_yellow_objects = {}

next_blue_id = 0
next_yellow_id = 0

def get_unique_color(object_id, object_type):
    rng = np.random.default_rng(
        object_id + (1000 if object_type == "yellow" else 0)
    )

    return (
        int(rng.integers(100, 256)),
        int(rng.integers(100, 256)),
        int(rng.integers(100, 256))
    )

def get_contour_centroid(contour):
    moments = cv2.moments(contour)

    if moments["m00"] == 0:
        return None

    cx = int(moments["m10"] / moments["m00"])
    cy = int(moments["m01"] / moments["m00"])

    return cx, cy

def detect_objects(frame):
    blurred = cv2.GaussianBlur(frame, (5, 5), 0)
    hsv = cv2.cvtColor(blurred, cv2.COLOR_BGR2HSV)

    blue_mask = cv2.inRange(
        hsv,
        lower_blue,
        upper_blue
    )

    blue_mask = cv2.morphologyEx(
        blue_mask,
        cv2.MORPH_OPEN,
        kernel,
        iterations=1
    )

    blue_mask = cv2.morphologyEx(
        blue_mask,
        cv2.MORPH_CLOSE,
        kernel,
        iterations=2
    )

    blue_contours, _ = cv2.findContours(
        blue_mask,
        cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_SIMPLE
    )

    blue_detections = []

    for contour in blue_contours:
        area = cv2.contourArea(contour)

        if area < MIN_BLUE_AREA:
            continue

        centre = get_contour_centroid(contour)

        if centre is None:
            continue

        x, y, box_w, box_h = cv2.boundingRect(contour)

        blue_detections.append({
            "center": centre,
            "area": float(area),
            "contour": contour,
            "bbox": (x, y, box_w, box_h),
            "radius": None,
            "circularity": None,
            "solidity": None
        })

    yellow_mask = cv2.inRange(
        hsv,
        lower_yellow,
        upper_yellow
    )

    yellow_mask = cv2.morphologyEx(
        yellow_mask,
        cv2.MORPH_OPEN,
        kernel,
        iterations=1
    )

    yellow_mask = cv2.morphologyEx(
        yellow_mask,
        cv2.MORPH_CLOSE,
        kernel,
        iterations=1
    )

    yellow_contours, _ = cv2.findContours(
        yellow_mask,
        cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_SIMPLE
    )

    yellow_detections = []

    for contour in yellow_contours:
        area = cv2.contourArea(contour)

        if area < MIN_BALL_AREA:
            continue

        perimeter = cv2.arcLength(contour, True)
        if perimeter == 0:
            continue

        circularity = (4 * math.pi * area) / (perimeter ** 2)

        if circularity < MIN_CIRCULARITY:
            continue

        (cx, cy), radius = cv2.minEnclosingCircle(contour)
        cx, cy, radius = int(cx), int(cy), int(radius)

        if radius < MIN_BALL_RADIUS or radius > MAX_BALL_RADIUS:
            continue

        hull = cv2.convexHull(contour)
        hull_area = cv2.contourArea(hull)
        solidity = float(area) / float(hull_area) if hull_area > 0 else 0.0

        x, y, box_w, box_h = cv2.boundingRect(contour)

        yellow_detections.append({
            "center": (cx, cy),
            "area": float(area),
            "contour": contour,
            "bbox": (x, y, box_w, box_h),
            "radius": float(radius),
            "circularity": float(circularity),
            "solidity": float(solidity)
        })

    return blue_detections, yellow_detections

def update_tracks(detections, active_objects, next_id, object_type):
    for track in active_objects.values():
        track["matched"] = False
        track["visible"] = False

    available_tracks = set(active_objects.keys())

    for detection in detections:
        centre = detection["center"]

        best_id = None
        best_distance = float("inf")

        for track_id in available_tracks:
            old_centre = active_objects[track_id]["center"]
            distance = math.dist(centre, old_centre)

            if (
                distance < MAX_TRACK_DISTANCE
                and distance < best_distance
            ):
                best_distance = distance
                best_id = track_id

        if best_id is not None:
            track = active_objects[best_id]

            old_x, old_y = track["center"]
            new_x, new_y = centre

            track["center"] = (
                int(0.6 * old_x + 0.4 * new_x),
                int(0.6 * old_y + 0.4 * new_y)
            )

            track["area"] = detection["area"]
            track["contour"] = detection["contour"]
            track["bbox"] = detection["bbox"]
            track["radius"] = detection["radius"]
            track["circularity"] = detection["circularity"]
            track["solidity"] = detection["solidity"]
            track["disappeared"] = 0
            track["visible"] = True
            track["matched"] = True
            track["trail"].appendleft(track["center"])

            available_tracks.remove(best_id)

        else:
            track_id = next_id
            next_id += 1

            active_objects[track_id] = {
                "center": centre,
                "area": detection["area"],
                "contour": detection["contour"],
                "bbox": detection["bbox"],
                "radius": detection["radius"],
                "circularity": detection["circularity"],
                "solidity": detection["solidity"],
                "color": get_unique_color(track_id, object_type),
                "trail": deque([centre], maxlen=TRAIL_LENGTH),
                "disappeared": 0,
                "visible": True,
                "matched": True
            }

    for track_id in list(active_objects.keys()):
        track = active_objects[track_id]

        if not track["matched"]:
            track["disappeared"] += 1

            if track["disappeared"] > MAX_DISAPPEARED:
                del active_objects[track_id]

    return active_objects, next_id

def draw_tracks(frame, black_canvas, active_objects, object_type):
    visible_centres = []

    for track_id, track in active_objects.items():
        centre = track["center"]
        colour = track["color"]
        points = list(track["trail"])

        for i in range(1, len(points)):
            thickness = max(
                1,
                int(3 * (1.0 - i / len(points)))
            )

            cv2.line(
                black_canvas,
                points[i - 1],
                points[i],
                colour,
                thickness
            )

        if not track["visible"]:
            continue

        visible_centres.append(centre)

        area_px = track["area"]

        if object_type == "blue":
            cv2.drawContours(
                frame,
                [track["contour"]],
                -1,
                (255, 0, 0),
                2
            )

            cv2.circle(
                frame,
                centre,
                5,
                (0, 0, 255),
                -1
            )

            label = f"Blue ID {track_id}"

        else:
            cv2.circle(
                frame,
                centre,
                int(track["radius"]),
                (0, 255, 255),
                2
            )

            cv2.circle(
                frame,
                centre,
                4,
                (0, 0, 255),
                -1
            )

            label = f"Yellow Ball ID {track_id}"

        cv2.putText(
            frame,
            label,
            (centre[0] + 8, centre[1] - 25),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.48,
            colour,
            2
        )

        cv2.putText(
            frame,
            f"Area: {area_px:.0f} px",
            (centre[0] + 8, centre[1] - 8),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.45,
            (255, 255, 255),
            1
        )

        cv2.circle(
            black_canvas,
            centre,
            5,
            colour,
            -1
        )

        cv2.putText(
            black_canvas,
            f"{object_type.capitalize()} {track_id}",
            (centre[0] + 8, centre[1] - 8),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            colour,
            2
        )

    return visible_centres

try:
    while True:
        ret, frame = cap.read()

        if not ret:
            print("Could not read webcam frame.")
            break

        undistorted_frame = cv2.remap(
            frame,
            map_x,
            map_y,
            cv2.INTER_LINEAR
        )

        height, width = undistorted_frame.shape[:2]

        black_canvas = np.zeros(
            (height, width, 3),
            dtype=np.uint8
        )

        (
            blue_detections,
            yellow_detections
        ) = detect_objects(undistorted_frame)

        active_blue_objects, next_blue_id = update_tracks(
            blue_detections,
            active_blue_objects,
            next_blue_id,
            "blue"
        )

        active_yellow_objects, next_yellow_id = update_tracks(
            yellow_detections,
            active_yellow_objects,
            next_yellow_id,
            "yellow"
        )

        draw_tracks(
            undistorted_frame,
            black_canvas,
            active_blue_objects,
            "blue"
        )

        draw_tracks(
            undistorted_frame,
            black_canvas,
            active_yellow_objects,
            "yellow"
        )

        visible_blue_ids = sorted(
            track_id
            for track_id, track in active_blue_objects.items()
            if track["visible"]
        )

        visible_yellow_ids = sorted(
            track_id
            for track_id, track in active_yellow_objects.items()
            if track["visible"]
        )

        if visible_blue_ids and visible_yellow_ids:
            primary_blue = active_blue_objects[visible_blue_ids[0]]
            primary_yellow = active_yellow_objects[visible_yellow_ids[0]]

            blue_centre = primary_blue["center"]
            yellow_centre = primary_yellow["center"]

            dx = yellow_centre[0] - blue_centre[0]
            dy = yellow_centre[1] - blue_centre[1]

            cv2.line(
                undistorted_frame,
                blue_centre,
                yellow_centre,
                (255, 255, 255),
                2
            )

            midpoint = (
                (blue_centre[0] + yellow_centre[0]) // 2,
                (blue_centre[1] + yellow_centre[1]) // 2
            )

            cv2.putText(
                undistorted_frame,
                f"dx: {dx}px, dy: {dy}px",
                midpoint,
                cv2.FONT_HERSHEY_SIMPLEX,
                0.55,
                (255, 255, 255),
                2
            )

        blue_visible_count = len(visible_blue_ids)
        yellow_visible_count = len(visible_yellow_ids)

        cv2.putText(
            undistorted_frame,
            f"Blue objects: {blue_visible_count}",
            (10, 25),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (255, 100, 0),
            2
        )

        cv2.putText(
            undistorted_frame,
            f"Yellow balls: {yellow_visible_count}",
            (10, 50),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (0, 255, 255),
            2
        )

        cv2.imshow(
            "Blue Object & Yellow Ball Tracker",
            undistorted_frame
        )

        cv2.imshow(
            "Centroid Tracking (Pitch Black)",
            black_canvas
        )

        key = cv2.waitKey(1) & 0xFF

        if key == ord("q"):
            break

finally:
    cap.release()
    cv2.destroyAllWindows()