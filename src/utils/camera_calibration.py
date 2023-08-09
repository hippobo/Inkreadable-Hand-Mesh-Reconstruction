

import numpy as np
import cv2
import os


def calibrate(dirpath, square_size, width, height):
    """ Apply camera calibration operation for images in the given directory path. """

   
    
    if not os.path.isdir(dirpath):
        raise Exception(f'Invalid directory: {dirpath}')
    # termination criteria
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)

    # prepare object points, like (0,0,0), (1,0,0), (2,0,0) ....,(8,6,0)
    objp = np.zeros((height*width, 3), np.float32)
    objp[:, :2] = np.mgrid[0:width, 0:height].T.reshape(-1, 2)

    objp = objp * square_size

    # Arrays to store object points and image points from all the images.
    objpoints = []  # 3d point in real world space
    imgpoints = []  # 2d points in image plane.

    images = os.listdir(dirpath)
    total_images = len(images)
    num_chessboards_detected = 0
    for fname in images:
        img = cv2.imread(os.path.join(dirpath, fname))
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        # Find the chess board corners
        ret, corners = cv2.findChessboardCorners(gray, (width, height), None)

        # If found, add object points, image points (after refining them)
        if ret:
            objpoints.append(objp)
            

            corners2 = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)
            imgpoints.append(corners2)

            # Draw and display the corners
            img = cv2.drawChessboardCorners(img, (width, height), corners2, ret)
            num_chessboards_detected += 1 
            print(f"Chessboard detected in {fname}")
        else:
            print(f"Chessboard not detected in {fname}")
            


    if num_chessboards_detected != total_images:  # If chessboard was not found in any image
        raise ValueError("Could not find chessboard in all images.")
    
    ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, gray.shape[::-1], None, None)

    print("Calibration Done")
    return [ret, mtx, dist, rvecs, tvecs]


