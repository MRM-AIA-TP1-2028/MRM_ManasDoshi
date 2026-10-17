import cv2
import numpy as np
import os

CALIB_FILE = "camera_calibration.yaml"  
MARKER_SIZE_METERS = 0.034              

ARUCO_DICT = {
    "DICT_4X4_100": cv2.aruco.DICT_4X4_100,
    "DICT_5X5_100": cv2.aruco.DICT_5X5_100,
    "DICT_6X6_100": cv2.aruco.DICT_6X6_100,
}

def load_camera_calibration(calib_path):
    if not os.path.exists(calib_path):
        print(f"Error: Calibration file '{calib_path}' not found.")
        print("Please calibrate your camera using a chessboard pattern and save the parameters to an OpenCV YAML file.")
        exit(1)
        
    cv_file = cv2.FileStorage(calib_path, cv2.FILE_STORAGE_READ)
    camera_matrix = cv_file.getNode("camera_matrix").mat()
    dist_coeffs = cv_file.getNode("dist_coeff").mat()
    cv_file.release()
    
    return camera_matrix, dist_coeffs

camera_matrix, dist_coeffs = load_camera_calibration(CALIB_FILE)

cap = cv2.VideoCapture(0)  
if not cap.isOpened():
    print("Error: Could not open webcam.")
    exit()

print("Webcam started with real calibration data! Press 'q' to quit.")

while True:
    ret, frame = cap.read()
    if not ret:
        print("Failed to grab frame.")
        break

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    for dict_name, dict_id in ARUCO_DICT.items():
        aruco_dict = cv2.aruco.getPredefinedDictionary(dict_id)
        parameters = cv2.aruco.DetectorParameters()
        detector = cv2.aruco.ArucoDetector(aruco_dict, parameters)

        corners, ids, rejected = detector.detectMarkers(gray)

        if ids is not None and len(corners) > 0:
            cv2.aruco.drawDetectedMarkers(frame, corners, ids)

            half_size = MARKER_SIZE_METERS / 2.0
            obj_points = np.array([
                [-half_size,  half_size, 0],
                [ half_size,  half_size, 0],
                [ half_size, -half_size, 0],
                [-half_size, -half_size, 0]
            ], dtype=np.float32)

            for i in range(len(ids)):
                success, rvec, tvec = cv2.solvePnP(
                    obj_points, 
                    corners[i][0], 
                    camera_matrix, 
                    dist_coeffs, 
                    flags=cv2.SOLVEPNP_ITERATIVE
                )

                if success:
                    cv2.drawFrameAxes(frame, camera_matrix, dist_coeffs, rvec, tvec, 0.03)

                    depth_m = tvec[2][0]
                    depth_cm = depth_m * 100

                    text = f"ID: {ids[i][0]} | Depth: {depth_cm:.1f} cm"
                    
                    corner_point = tuple(corners[i][0][0].astype(int))
                    cv2.putText(
                        frame, text, (corner_point[0], corner_point[1] - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2
                    )

    cv2.imshow("Real-World ArUco Pose & Depth Estimation", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()