import cv2
import numpy as np

CHECKBOARD = (7, 5)  
criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)

objp = np.zeros((CHECKBOARD[0] * CHECKBOARD[1], 3), np.float32)
objp[:, :2] = np.mgrid[0:CHECKBOARD[0], 0:CHECKBOARD[1]].T.reshape(-1, 2)

objpoints = []  
imgpoints = []  

cap = cv2.VideoCapture(0)
print("=== CAMERA CALIBRATION TOOL ===")
print("1. Hold your printed chessboard in front of the camera.")
print("2. Press 'c' to capture a frame (try different angles and distances).")
print("3. Aim for 15-20 captures.")
print("4. Press 's' to calculate and save. Press 'q' to quit.")

capture_count = 0

while True:
    ret, frame = cap.read()
    if not ret:
        break

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    
    ret_corners, corners = cv2.findChessboardCorners(gray, CHECKBOARD, None)

    display_frame = frame.copy()
    if ret_corners:
        corners2 = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)
        cv2.drawChessboardCorners(display_frame, CHECKBOARD, corners2, ret_corners)
        cv2.putText(display_frame, f"Captures: {capture_count} | Press 'c' to capture", (20, 40), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
    else:
        cv2.putText(display_frame, "Chessboard not found!", (20, 40), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

    cv2.imshow("Calibration", display_frame)
    key = cv2.waitKey(1) & 0xFF

    if key == ord('q'):
        break
    elif key == ord('c') and ret_corners:
        objpoints.append(objp)
        imgpoints.append(corners2)
        capture_count += 1
        print(f"Captured frame {capture_count} successfully!")
    elif key == ord('s'):
        if capture_count < 10:
            print("Warning: Recommended to have at least 10-15 captures for accurate results. Capturing more...")
            continue
        print("Calculating camera matrix and distortion coefficients...")
        ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, gray.shape[::-1], None, None)
        
        cv_file = cv2.FileStorage("camera_calibration.yaml", cv2.FILE_STORAGE_WRITE)
        cv_file.write("camera_matrix", mtx)
        cv_file.write("dist_coeff", dist)
        cv_file.release()
        
        print("Calibration successful! Saved to 'camera_calibration.yaml'.")
        break

cap.release()
cv2.destroyAllWindows()