import math
import os
import time
from collections import deque

import cv2
import numpy as np

if not hasattr(cv2, "aruco"):
    raise RuntimeError("ArUco is unavailable. Install opencv-contrib-python.")

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
CALIBRATION_FILE = os.path.join(SCRIPT_DIR, "camera_calibration.yaml")

CAMERA_INDEX = 0
FRAME_WIDTH = 1280
FRAME_HEIGHT = 720
WARMUP_FRAMES = 5

MARKER_SIZE_METERS = 0.034
BALL_RADIUS_METERS = 0.0335
MIN_MARKER_DEPTH = 0.05
MAX_MARKER_DEPTH = 5.0
DUPLICATE_MARKER_DISTANCE = 10.0
TAG_MARGIN = 25.0
MIN_COS_TILT = 0.3

ARUCO_DICTIONARY_NAMES = [
    "DICT_4X4_250",
    "DICT_5X5_250",
    "DICT_6X6_250",
    "DICT_ARUCO_ORIGINAL",
]

MIN_BLUE_AREA = 500
MIN_BLUE_FRAGMENT_AREA = 60
BLUE_GROUP_GAP = 15

MIN_BALL_AREA = 300
MIN_BALL_RADIUS = 8
MAX_BALL_RADIUS = 250
MIN_BALL_FILL = 0.55
MIN_BALL_CIRCULARITY = 0.72
MIN_BALL_ASPECT = 0.6
MAX_BALL_ASPECT = 1.6

MIN_GATE = 60.0
GATE_SCALE = 1.0
MIN_HITS = 3
MAX_DISAPPEARED = 45
TENTATIVE_MAX_MISSED = 1
CENTER_SMOOTHING = 0.55
AREA_SMOOTHING = 0.35
DEPTH_SMOOTHING = 0.5
DEPTH_HOLD_FRAMES = 20
TRAIL_LENGTH = 48

CONTROL_WINDOW = "HSV Controls"
TRACKBARS = [
    ("Y H min", 22, 179),
    ("Y H max", 45, 179),
    ("Y S min", 70, 255),
    ("Y V min", 80, 255),
    ("B H min", 95, 179),
    ("B H max", 130, 179),
    ("B S min", 90, 255),
    ("B V min", 50, 255),
]

FONT = cv2.FONT_HERSHEY_SIMPLEX
ZERO_DIST = np.zeros((5, 1), dtype=np.float64)

HALF_MARKER = MARKER_SIZE_METERS / 2.0
MARKER_OBJECT_POINTS = np.array(
    [
        [-HALF_MARKER, HALF_MARKER, 0.0],
        [HALF_MARKER, HALF_MARKER, 0.0],
        [HALF_MARKER, -HALF_MARKER, 0.0],
        [-HALF_MARKER, -HALF_MARKER, 0.0],
    ],
    dtype=np.float32,
)

OPEN_KERNEL = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
BLUE_CLOSE_KERNEL = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9))
YELLOW_CLOSE_KERNEL = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (11, 11))
GROUP_KERNEL = cv2.getStructuringElement(
    cv2.MORPH_ELLIPSE, (2 * BLUE_GROUP_GAP + 1, 2 * BLUE_GROUP_GAP + 1)
)


def noop(_):
    pass


def get_color(track_id, kind):
    hue = (track_id * 37 + (0 if kind == "blue" else 90)) % 180
    pixel = np.uint8([[[hue, 220, 255]]])
    b, g, r = cv2.cvtColor(pixel, cv2.COLOR_HSV2BGR)[0, 0]
    return int(b), int(g), int(r)


def draw_text(image, text, origin, color=(255, 255, 255), scale=0.5, thickness=1):
    cv2.putText(image, text, origin, FONT, scale, (0, 0, 0), thickness + 2, cv2.LINE_AA)
    cv2.putText(image, text, origin, FONT, scale, color, thickness, cv2.LINE_AA)


def load_calibration(path):
    if not os.path.exists(path):
        raise FileNotFoundError(f"Calibration file not found: {path}")

    storage = cv2.FileStorage(path, cv2.FILE_STORAGE_READ)

    if not storage.isOpened():
        raise RuntimeError("Could not open camera_calibration.yaml")

    matrix = storage.getNode("camera_matrix").mat()
    dist = storage.getNode("distortion_coefficients").mat()
    width = int(storage.getNode("image_width").real())
    height = int(storage.getNode("image_height").real())
    storage.release()

    if matrix is None:
        raise ValueError("camera_matrix is missing from the YAML file.")

    if dist is None:
        raise ValueError("distortion_coefficients is missing from the YAML file.")

    matrix = np.asarray(matrix, dtype=np.float64)
    dist = np.asarray(dist, dtype=np.float64).reshape(-1, 1)

    if matrix.shape != (3, 3):
        raise ValueError("camera_matrix must be a 3x3 matrix.")

    if not np.all(np.isfinite(matrix)) or not np.all(np.isfinite(dist)):
        raise ValueError("Calibration contains invalid values.")

    if matrix[0, 0] <= 0 or matrix[1, 1] <= 0:
        raise ValueError("Camera focal lengths must be positive.")

    if width <= 0:
        width = FRAME_WIDTH

    if height <= 0:
        height = FRAME_HEIGHT

    return matrix, dist, width, height


def open_camera():
    if os.name == "nt":
        cap = cv2.VideoCapture(CAMERA_INDEX, cv2.CAP_DSHOW)

        if cap.isOpened():
            return cap

        cap.release()

    return cv2.VideoCapture(CAMERA_INDEX)


def build_aruco_parameters():
    if hasattr(cv2.aruco, "DetectorParameters"):
        parameters = cv2.aruco.DetectorParameters()
    else:
        parameters = cv2.aruco.DetectorParameters_create()

    parameters.cornerRefinementMethod = cv2.aruco.CORNER_REFINE_SUBPIX
    parameters.adaptiveThreshWinSizeMin = 3
    parameters.adaptiveThreshWinSizeMax = 53
    parameters.adaptiveThreshWinSizeStep = 10
    parameters.minMarkerPerimeterRate = 0.02
    parameters.maxMarkerPerimeterRate = 4.0
    parameters.polygonalApproxAccuracyRate = 0.05
    parameters.minDistanceToBorder = 1
    parameters.errorCorrectionRate = 0.6

    return parameters


def make_detect_function(dictionary, parameters):
    if hasattr(cv2.aruco, "ArucoDetector"):
        detector = cv2.aruco.ArucoDetector(dictionary, parameters)

        def detect(image):
            return detector.detectMarkers(image)

        return detect

    def detect_legacy(image):
        return cv2.aruco.detectMarkers(image, dictionary, parameters=parameters)

    return detect_legacy


def build_aruco_detectors(parameters):
    detectors = []

    for name in ARUCO_DICTIONARY_NAMES:
        constant = getattr(cv2.aruco, name, None)

        if constant is None:
            print(f"Skipping unavailable dictionary: {name}")
            continue

        dictionary = cv2.aruco.getPredefinedDictionary(constant)
        detectors.append((name, make_detect_function(dictionary, parameters)))

    if not detectors:
        raise RuntimeError("No ArUco dictionaries could be loaded.")

    return detectors


def collect_markers(gray, detectors, pose_matrix):
    markers = []

    for name, detect in detectors:
        corners_list, ids, _ = detect(gray)

        if ids is None or len(corners_list) == 0:
            continue

        for corners, marker_id in zip(corners_list, ids.flatten()):
            points = corners.reshape(4, 2).astype(np.float32)
            center = points.mean(axis=0)

            duplicate = False

            for existing in markers:
                if np.linalg.norm(center - existing["center"]) < DUPLICATE_MARKER_DISTANCE:
                    duplicate = True
                    break

            if duplicate:
                continue

            try:
                success, rvec, tvec = cv2.solvePnP(
                    MARKER_OBJECT_POINTS,
                    points,
                    pose_matrix,
                    ZERO_DIST,
                    flags=cv2.SOLVEPNP_IPPE_SQUARE,
                )
            except cv2.error:
                continue

            if not success:
                continue

            depth = float(tvec[2][0])

            if depth < MIN_MARKER_DEPTH or depth > MAX_MARKER_DEPTH:
                continue

            rotation, _ = cv2.Rodrigues(rvec)

            markers.append(
                {
                    "key": f"{name}:{int(marker_id)}",
                    "dict": name,
                    "id": int(marker_id),
                    "center": center,
                    "corners": points,
                    "depth": depth,
                    "cos_tilt": float(abs(rotation[2, 2])),
                    "rvec": rvec,
                    "tvec": tvec,
                }
            )

    return markers


def detect_markers(clean, detectors, clahe, pose_matrix):
    gray = cv2.cvtColor(clean, cv2.COLOR_BGR2GRAY)
    markers = collect_markers(gray, detectors, pose_matrix)

    if not markers:
        markers = collect_markers(clahe.apply(gray), detectors, pose_matrix)

    return markers


def detect_blue(hsv, lower, upper):
    mask = cv2.inRange(hsv, lower, upper)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, OPEN_KERNEL)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, BLUE_CLOSE_KERNEL, iterations=2)

    grouping = cv2.dilate(mask, GROUP_KERNEL)
    group_contours, _ = cv2.findContours(
        grouping, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )

    detections = []

    for group_contour in group_contours:
        x, y, w, h = cv2.boundingRect(group_contour)

        group_mask = np.zeros((h, w), dtype=np.uint8)
        cv2.drawContours(group_mask, [group_contour], -1, 255, -1, offset=(-x, -y))

        member = cv2.bitwise_and(group_mask, mask[y:y + h, x:x + w])

        contours, _ = cv2.findContours(
            member, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE, offset=(x, y)
        )

        contours = [
            contour
            for contour in contours
            if cv2.contourArea(contour) >= MIN_BLUE_FRAGMENT_AREA
        ]

        if not contours:
            continue

        area = sum(cv2.contourArea(contour) for contour in contours)

        if area < MIN_BLUE_AREA:
            continue

        moments = cv2.moments(member, binaryImage=True)

        if moments["m00"] == 0:
            continue

        center = (
            moments["m10"] / moments["m00"] + x,
            moments["m01"] / moments["m00"] + y,
        )

        detections.append(
            {
                "kind": "blue",
                "center": center,
                "area": float(area),
                "contours": contours,
                "radius": None,
                "bbox": cv2.boundingRect(np.vstack(contours)),
                "tag_key": None,
                "depth": None,
                "cos_tilt": None,
            }
        )

    return detections, mask


def detect_yellow(hsv, lower, upper):
    mask = cv2.inRange(hsv, lower, upper)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, OPEN_KERNEL)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, YELLOW_CLOSE_KERNEL, iterations=2)

    contours, _ = cv2.findContours(
        mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )

    candidates = []

    for contour in contours:
        area = cv2.contourArea(contour)

        if area < MIN_BALL_AREA:
            continue

        hull = cv2.convexHull(contour)
        hull_area = cv2.contourArea(hull)

        if hull_area <= 0:
            continue

        if area / hull_area < MIN_BALL_FILL:
            continue

        perimeter = cv2.arcLength(hull, True)

        if perimeter <= 0:
            continue

        circularity = 4.0 * math.pi * hull_area / (perimeter ** 2)

        if circularity < MIN_BALL_CIRCULARITY:
            continue

        x, y, w, h = cv2.boundingRect(hull)

        if h == 0:
            continue

        aspect = w / float(h)

        if aspect < MIN_BALL_ASPECT or aspect > MAX_BALL_ASPECT:
            continue

        (_, _), radius = cv2.minEnclosingCircle(hull)

        if radius < MIN_BALL_RADIUS or radius > MAX_BALL_RADIUS:
            continue

        moments = cv2.moments(hull)

        if moments["m00"] == 0:
            continue

        center = (
            moments["m10"] / moments["m00"],
            moments["m01"] / moments["m00"],
        )

        candidates.append(
            {
                "kind": "yellow",
                "center": center,
                "area": float(hull_area),
                "contours": [hull],
                "radius": float(radius),
                "bbox": (x, y, w, h),
                "tag_key": None,
                "depth": None,
                "cos_tilt": None,
            }
        )

    candidates.sort(key=lambda item: item["area"], reverse=True)

    accepted = []

    for candidate in candidates:
        inside_existing = False

        for other in accepted:
            separation = math.hypot(
                candidate["center"][0] - other["center"][0],
                candidate["center"][1] - other["center"][1],
            )

            if separation < other["radius"]:
                inside_existing = True
                break

        if not inside_existing:
            accepted.append(candidate)

    return accepted, mask


def associate_tags(detections, markers):
    pairs = []

    for detection_index, detection in enumerate(detections):
        for marker_index, marker in enumerate(markers):
            point = (float(marker["center"][0]), float(marker["center"][1]))

            inside = max(
                cv2.pointPolygonTest(contour, point, True)
                for contour in detection["contours"]
            )

            if inside >= -TAG_MARGIN:
                distance = math.hypot(
                    detection["center"][0] - point[0],
                    detection["center"][1] - point[1],
                )
                pairs.append((distance, detection_index, marker_index))

    pairs.sort(key=lambda item: item[0])

    used_detections = set()
    used_markers = set()

    for _, detection_index, marker_index in pairs:
        if detection_index in used_detections or marker_index in used_markers:
            continue

        detection = detections[detection_index]
        marker = markers[marker_index]

        detection["tag_key"] = marker["key"]
        detection["depth"] = marker["depth"]
        detection["cos_tilt"] = marker["cos_tilt"]

        used_detections.add(detection_index)
        used_markers.add(marker_index)


class Track:
    def __init__(self, kind, detection):
        self.kind = kind
        self.id = None
        self.color = (255, 255, 255)
        self.center = np.array(detection["center"], dtype=np.float64)
        self.velocity = np.zeros(2, dtype=np.float64)
        self.area = float(detection["area"])
        self.radius = detection["radius"]
        self.contours = detection["contours"]
        self.bbox = detection["bbox"]
        self.tag_key = detection["tag_key"]
        self.depth = detection["depth"]
        self.cos_tilt = (
            detection["cos_tilt"] if detection["cos_tilt"] is not None else 1.0
        )
        self.depth_age = 0
        self.hits = 1
        self.missed = 0
        self.visible = True
        self.trail = deque(maxlen=TRAIL_LENGTH)

    @property
    def size(self):
        if self.radius is not None:
            return 2.0 * self.radius

        return math.sqrt(max(self.area, 1.0))

    def predicted(self):
        return self.center + self.velocity

    def update(self, detection):
        measured = np.array(detection["center"], dtype=np.float64)
        previous = self.center.copy()

        self.center = (
            (1.0 - CENTER_SMOOTHING) * self.center + CENTER_SMOOTHING * measured
        )
        self.velocity = 0.7 * self.velocity + 0.3 * (self.center - previous)

        self.area = (
            (1.0 - AREA_SMOOTHING) * self.area + AREA_SMOOTHING * detection["area"]
        )

        if detection["radius"] is not None:
            if self.radius is None:
                self.radius = detection["radius"]
            else:
                self.radius = (
                    (1.0 - AREA_SMOOTHING) * self.radius
                    + AREA_SMOOTHING * detection["radius"]
                )

        self.contours = detection["contours"]
        self.bbox = detection["bbox"]

        if detection["tag_key"] is not None:
            self.tag_key = detection["tag_key"]

            if self.depth is None:
                self.depth = detection["depth"]
                self.cos_tilt = detection["cos_tilt"]
            else:
                self.depth = (
                    (1.0 - DEPTH_SMOOTHING) * self.depth
                    + DEPTH_SMOOTHING * detection["depth"]
                )
                self.cos_tilt = (
                    (1.0 - DEPTH_SMOOTHING) * self.cos_tilt
                    + DEPTH_SMOOTHING * detection["cos_tilt"]
                )

            self.depth_age = 0
        else:
            self.depth_age += 1

            if self.depth_age > DEPTH_HOLD_FRAMES:
                self.depth = None

        self.hits += 1
        self.missed = 0
        self.visible = True

    def mark_missed(self):
        self.missed += 1
        self.visible = False
        self.center = self.center + self.velocity
        self.velocity = self.velocity * 0.8
        self.depth_age += 1

        if self.depth_age > DEPTH_HOLD_FRAMES:
            self.depth = None


class Tracker:
    def __init__(self, kind):
        self.kind = kind
        self.tracks = []
        self.next_id = 1

    def reset(self):
        self.tracks = []
        self.next_id = 1

    def confirm(self, track):
        if track.id is None and track.hits >= MIN_HITS:
            track.id = self.next_id
            self.next_id += 1
            track.color = get_color(track.id, self.kind)

    def update(self, detections):
        for track in self.tracks:
            track.visible = False

        free_tracks = set(range(len(self.tracks)))
        free_detections = set(range(len(detections)))
        matches = []

        for detection_index in sorted(free_detections):
            key = detections[detection_index]["tag_key"]

            if key is None:
                continue

            measured = np.array(detections[detection_index]["center"], dtype=np.float64)
            best_track = None
            best_distance = float("inf")

            for track_index in free_tracks:
                track = self.tracks[track_index]

                if track.tag_key != key:
                    continue

                distance = float(np.linalg.norm(track.predicted() - measured))

                if distance < best_distance:
                    best_distance = distance
                    best_track = track_index

            if best_track is not None:
                matches.append((best_track, detection_index))
                free_tracks.discard(best_track)
                free_detections.discard(detection_index)

        candidates = []

        for track_index in free_tracks:
            track = self.tracks[track_index]
            predicted = track.predicted()
            gate = max(MIN_GATE, GATE_SCALE * track.size)
            gate *= 1.0 + 0.1 * min(track.missed, 10)

            for detection_index in free_detections:
                detection = detections[detection_index]

                if (
                    track.tag_key is not None
                    and detection["tag_key"] is not None
                    and track.tag_key != detection["tag_key"]
                ):
                    continue

                measured = np.array(detection["center"], dtype=np.float64)
                distance = float(np.linalg.norm(predicted - measured))

                if distance <= gate:
                    candidates.append((distance, track_index, detection_index))

        candidates.sort(key=lambda item: item[0])

        for _, track_index, detection_index in candidates:
            if track_index in free_tracks and detection_index in free_detections:
                matches.append((track_index, detection_index))
                free_tracks.discard(track_index)
                free_detections.discard(detection_index)

        for track_index, detection_index in matches:
            track = self.tracks[track_index]
            track.update(detections[detection_index])
            self.confirm(track)

            if track.id is not None:
                track.trail.appendleft(
                    (int(round(track.center[0])), int(round(track.center[1])))
                )

        for track_index in free_tracks:
            self.tracks[track_index].mark_missed()

        for detection_index in sorted(free_detections):
            track = Track(self.kind, detections[detection_index])
            self.confirm(track)

            if track.id is not None:
                track.trail.appendleft(
                    (int(round(track.center[0])), int(round(track.center[1])))
                )

            self.tracks.append(track)

        survivors = []

        for track in self.tracks:
            limit = TENTATIVE_MAX_MISSED if track.id is None else MAX_DISAPPEARED

            if track.missed <= limit:
                survivors.append(track)

        self.tracks = survivors

    def visible_confirmed(self):
        return [t for t in self.tracks if t.id is not None and t.visible]


def real_area_cm2(track, pose_matrix):
    if track.depth is None:
        return None

    depth = track.depth

    if track.kind == "yellow":
        depth += BALL_RADIUS_METERS

    fx = pose_matrix[0, 0]
    fy = pose_matrix[1, 1]

    area_m2 = track.area * depth * depth / (fx * fy)

    if track.kind == "blue":
        area_m2 /= max(track.cos_tilt, MIN_COS_TILT)

    return area_m2 * 10000.0


def camera_point(track, pose_matrix):
    depth = track.depth

    if track.kind == "yellow":
        depth += BALL_RADIUS_METERS

    x = (track.center[0] - pose_matrix[0, 2]) * depth / pose_matrix[0, 0]
    y = (track.center[1] - pose_matrix[1, 2]) * depth / pose_matrix[1, 1]

    return np.array([x, y, depth], dtype=np.float64)


def draw_markers(display, markers, pose_matrix):
    for marker in markers:
        points = marker["corners"].astype(np.int32).reshape(-1, 1, 2)
        cv2.polylines(display, [points], True, (0, 255, 0), 2)

        cv2.drawFrameAxes(
            display,
            pose_matrix,
            ZERO_DIST,
            marker["rvec"],
            marker["tvec"],
            MARKER_SIZE_METERS * 0.5,
        )

        cx = int(marker["center"][0])
        cy = int(marker["center"][1])
        label = f"{marker['dict'].replace('DICT_', '')} #{marker['id']}"

        draw_text(
            display,
            f"{label}: {marker['depth']:.2f} m",
            (cx + 10, cy - 10),
            (0, 255, 0),
            0.45,
        )


def draw_tracks(display, canvas, tracker, pose_matrix):
    contour_color = (255, 80, 0) if tracker.kind == "blue" else (0, 255, 255)

    for track in tracker.tracks:
        if track.id is None:
            continue

        points = list(track.trail)
        count = len(points)

        for i in range(1, count):
            thickness = max(1, int(4 * (1.0 - i / count)))
            cv2.line(canvas, points[i - 1], points[i], track.color, thickness, cv2.LINE_AA)

        if not track.visible:
            continue

        center = (int(round(track.center[0])), int(round(track.center[1])))

        if track.kind == "yellow":
            radius = int(round(track.radius)) if track.radius is not None else 20
            cv2.circle(display, center, radius, (0, 255, 255), 2, cv2.LINE_AA)
        else:
            cv2.drawContours(display, track.contours, -1, contour_color, 2)

        cv2.circle(display, center, 5, (0, 0, 255), -1)
        cv2.circle(canvas, center, 6, track.color, -1)

        name = "Blue" if track.kind == "blue" else "Yellow Ball"

        lines = [
            (f"{name} ID {track.id}", track.color),
            (f"Area: {track.area:.0f} px", (255, 255, 255)),
        ]

        if track.depth is not None:
            area_cm2 = real_area_cm2(track, pose_matrix)
            lines.append((f"Depth: {track.depth:.2f} m", (255, 255, 255)))
            lines.append((f"Real area: {area_cm2:.2f} cm2", (255, 255, 255)))
            lines.append((f"Tag: {track.tag_key.replace('DICT_', '')}", (255, 255, 255)))
        else:
            lines.append(("Depth: no tag", (200, 200, 200)))

        for index, (text, color) in enumerate(lines):
            draw_text(
                display,
                text,
                (center[0] + 12, center[1] - 20 + index * 18),
                color,
                0.48,
            )

        draw_text(
            canvas,
            f"{name} {track.id}",
            (center[0] + 10, center[1] - 10),
            track.color,
            0.5,
        )


def draw_links(display, canvas, blue_tracker, yellow_tracker, pose_matrix):
    blues = blue_tracker.visible_confirmed()
    yellows = yellow_tracker.visible_confirmed()

    if not blues or not yellows:
        return

    for yellow in yellows:
        blue = min(blues, key=lambda item: float(np.linalg.norm(item.center - yellow.center)))

        blue_point = (int(round(blue.center[0])), int(round(blue.center[1])))
        yellow_point = (int(round(yellow.center[0])), int(round(yellow.center[1])))

        dx = yellow_point[0] - blue_point[0]
        dy = yellow_point[1] - blue_point[1]
        pixel_distance = math.hypot(dx, dy)

        cv2.line(display, blue_point, yellow_point, (255, 255, 255), 2, cv2.LINE_AA)
        cv2.line(canvas, blue_point, yellow_point, (120, 120, 120), 1, cv2.LINE_AA)

        midpoint = ((blue_point[0] + yellow_point[0]) // 2, (blue_point[1] + yellow_point[1]) // 2)

        text = f"dx {dx} dy {dy} d {pixel_distance:.0f}px"

        if blue.depth is not None and yellow.depth is not None:
            separation = np.linalg.norm(
                camera_point(blue, pose_matrix) - camera_point(yellow, pose_matrix)
            )
            text += f" | {separation * 100.0:.1f} cm"

        draw_text(display, text, midpoint, (255, 255, 255), 0.5)


def read_hsv_bounds():
    def pos(name):
        return cv2.getTrackbarPos(name, CONTROL_WINDOW)

    yellow_lower = np.array([pos("Y H min"), pos("Y S min"), pos("Y V min")], dtype=np.uint8)
    yellow_upper = np.array([pos("Y H max"), 255, 255], dtype=np.uint8)
    blue_lower = np.array([pos("B H min"), pos("B S min"), pos("B V min")], dtype=np.uint8)
    blue_upper = np.array([pos("B H max"), 255, 255], dtype=np.uint8)

    return yellow_lower, yellow_upper, blue_lower, blue_upper


def main():
    camera_matrix, dist_coeffs, calib_width, calib_height = load_calibration(CALIBRATION_FILE)

    cap = open_camera()

    if not cap.isOpened():
        raise RuntimeError("Could not open webcam.")

    cap.set(cv2.CAP_PROP_FRAME_WIDTH, FRAME_WIDTH)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, FRAME_HEIGHT)

    first_frame = None

    for _ in range(WARMUP_FRAMES):
        ret, frame = cap.read()

        if ret:
            first_frame = frame

    if first_frame is None:
        cap.release()
        raise RuntimeError("Could not read the first webcam frame.")

    actual_height, actual_width = first_frame.shape[:2]

    scale_x = actual_width / float(calib_width)
    scale_y = actual_height / float(calib_height)

    scaled_matrix = camera_matrix.copy()
    scaled_matrix[0, 0] *= scale_x
    scaled_matrix[0, 2] *= scale_x
    scaled_matrix[1, 1] *= scale_y
    scaled_matrix[1, 2] *= scale_y

    new_matrix, _ = cv2.getOptimalNewCameraMatrix(
        scaled_matrix,
        dist_coeffs,
        (actual_width, actual_height),
        1,
        (actual_width, actual_height),
    )

    map_x, map_y = cv2.initUndistortRectifyMap(
        scaled_matrix,
        dist_coeffs,
        None,
        new_matrix,
        (actual_width, actual_height),
        cv2.CV_32FC1,
    )

    pose_matrix = new_matrix.astype(np.float64)

    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    detectors = build_aruco_detectors(build_aruco_parameters())

    cv2.namedWindow(CONTROL_WINDOW, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(CONTROL_WINDOW, 420, 320)

    for name, default, maximum in TRACKBARS:
        cv2.createTrackbar(name, CONTROL_WINDOW, default, maximum, noop)

    blue_tracker = Tracker("blue")
    yellow_tracker = Tracker("yellow")

    show_masks = False
    masks_open = False
    last_time = time.time()
    fps = 0.0

    try:
        while True:
            ret, frame = cap.read()

            if not ret:
                print("Could not read webcam frame.")
                break

            clean = cv2.remap(frame, map_x, map_y, cv2.INTER_LINEAR)
            display = clean.copy()

            height, width = clean.shape[:2]
            canvas = np.zeros((height, width, 3), dtype=np.uint8)

            markers = detect_markers(clean, detectors, clahe, pose_matrix)

            yellow_lower, yellow_upper, blue_lower, blue_upper = read_hsv_bounds()

            blurred = cv2.GaussianBlur(clean, (7, 7), 0)
            hsv = cv2.cvtColor(blurred, cv2.COLOR_BGR2HSV)

            blue_detections, blue_mask = detect_blue(hsv, blue_lower, blue_upper)
            yellow_detections, yellow_mask = detect_yellow(hsv, yellow_lower, yellow_upper)

            associate_tags(blue_detections + yellow_detections, markers)

            blue_tracker.update(blue_detections)
            yellow_tracker.update(yellow_detections)

            draw_markers(display, markers, pose_matrix)
            draw_tracks(display, canvas, blue_tracker, pose_matrix)
            draw_tracks(display, canvas, yellow_tracker, pose_matrix)
            draw_links(display, canvas, blue_tracker, yellow_tracker, pose_matrix)

            now = time.time()
            elapsed = now - last_time
            last_time = now

            if elapsed > 0:
                fps = 0.9 * fps + 0.1 * (1.0 / elapsed)

            draw_text(display, f"Blue: {len(blue_tracker.visible_confirmed())}", (10, 25), (255, 120, 0), 0.6, 2)
            draw_text(display, f"Yellow: {len(yellow_tracker.visible_confirmed())}", (10, 50), (0, 255, 255), 0.6, 2)
            draw_text(display, f"Tags: {len(markers)}", (10, 75), (0, 255, 0), 0.6, 2)
            draw_text(display, f"FPS: {fps:.0f}", (10, 100), (255, 255, 255), 0.6, 2)
            draw_text(display, "q quit | m masks | s print HSV | r reset", (10, height - 12), (255, 255, 255), 0.5)

            cv2.imshow("Blue Object & Yellow Ball Tracker", display)
            cv2.imshow("Centroid Tracking (Pitch Black)", canvas)

            if show_masks:
                combined = np.hstack((blue_mask, yellow_mask))
                combined = cv2.resize(combined, None, fx=0.5, fy=0.5)
                cv2.imshow("Masks (blue | yellow)", combined)
                masks_open = True
            elif masks_open:
                cv2.destroyWindow("Masks (blue | yellow)")
                masks_open = False

            key = cv2.waitKey(1) & 0xFF

            if key == ord("q"):
                break

            if key == ord("m"):
                show_masks = not show_masks

            if key == ord("r"):
                blue_tracker.reset()
                yellow_tracker.reset()

            if key == ord("s"):
                print("yellow lower", yellow_lower.tolist(), "upper", yellow_upper.tolist())
                print("blue lower", blue_lower.tolist(), "upper", blue_upper.tolist())

    finally:
        cap.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
