#!/home/kevin/.pyenv/versions/pyenv_py385/bin/python
import cv2
import glob
import numpy as np
from cv2 import aruco

import argparse
import sys
import os
import math

aruco_dict = aruco.Dictionary_get(aruco.DICT_6X6_250)
board = aruco.CharucoBoard_create(7, 5, 1, .8, aruco_dict)
# cam = cv2.VideoCapture(0)
cam = cv2.VideoCapture(0)         # usb cam
cam.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cam.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

def cal():
    allCorners = []
    allIds = []
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100, 0.00001)
    images = glob.glob('/home/kevin/workspace/workspace_aruco/2021_aruco/camera_cal_img/cal*.png')
    #imsize = None
    for im in images:
        print("=> Processing image {0}".format(im))
        frame = cv2.imread(im)
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        corners, ids, rejectedImgPoints = cv2.aruco.detectMarkers(gray, aruco_dict)

        if len(corners) > 0:
            for corner in corners:
                cv2.cornerSubPix(gray, corner,
                                 winSize=(3, 3),
                                 zeroZone=(-1, -1),
                                 criteria=criteria)
            res2 = cv2.aruco.interpolateCornersCharuco(corners, ids, gray, board)
            if res2[1] is not None and res2[2] is not None and len(res2[1]) > 3:
                allCorners.append(res2[1])
                allIds.append(res2[2])

        imsize = gray.shape

    return allCorners, allIds, imsize


def calibrate_charuco(allCorners, allIds, imsize):
    '''
    cameraMatrixInit = np.array([[2714., 0., imsize[1] / 2],            #카메라 초점거리
                                 [0., 2714., imsize[0] / 2.],
                                 [0., 0., 1.]])
    '''
    cameraMatrixInit = np.array([[2000., 0., imsize[0] / 2.],
                                 [0., 2000., imsize[1] / 2.],
                                 [0., 0., 1.]])
    distCoeffsInit = np.zeros((5, 1))
    flags = (cv2.CALIB_USE_INTRINSIC_GUESS + cv2.CALIB_RATIONAL_MODEL + cv2.CALIB_FIX_ASPECT_RATIO)
    (ret, camera_matrix, distortion_coefficients0,
     rotation_vectors, translation_vectors,
     stdDeviationsIntrinsics, stdDeviationsExtrinsics,
     perViewErrors) = cv2.aruco.calibrateCameraCharucoExtended(
        charucoCorners=allCorners,
        charucoIds=allIds,
        board=board,
        imageSize=imsize,
        cameraMatrix=cameraMatrixInit,
        distCoeffs=distCoeffsInit,
        flags=flags,
        criteria=(cv2.TERM_CRITERIA_EPS & cv2.TERM_CRITERIA_COUNT, 10000, 1e-9))

    return ret, camera_matrix, distortion_coefficients0, rotation_vectors, translation_vectors


def detect_marker(mtx, dist):
    param = cv2.aruco.DetectorParameters_create()
    if cam.isOpened():
        while True:
            _, frame = cam.read()
            gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            coners, ids, point = cv2.aruco.detectMarkers(gray_frame, aruco_dict, parameters=param)
            if np.all(ids != None):
                # rvecs, tvecs, _objPoints = cv2.aruco.estimatePoseSingleMarkers(coners, 0.04, mtx, dist)
                # frame = cv2.aruco.drawAxis(frame, mtx, dist, rvecs[0], tvecs[0], 0.04)
                rvecs, tvecs, _objPoints = cv2.aruco.estimatePoseSingleMarkers(coners, 0.05, mtx, dist)
                frame = cv2.aruco.drawAxis(frame, mtx, dist, rvecs[0], tvecs[0], 0.05)
                rvecs_msg = rvecs.tolist()
                tvecs_msg = tvecs.tolist()
                rvecs_msg_x = rvecs_msg[0][0][0]
                rvecs_msg_y = rvecs_msg[0][0][1]
                rvecs_msg_z = rvecs_msg[0][0][2]
                tvecs_msg_x = tvecs_msg[0][0][0]
                tvecs_msg_y = tvecs_msg[0][0][1]
                tvecs_msg_z = tvecs_msg[0][0][2]
                cv2.putText(frame, "%.1f cm / %.0f deg" % ((tvecs[0][0][2] * 100), (rvecs[0][0][2] / math.pi * 180)),
                            (0, 450), cv2.FONT_HERSHEY_PLAIN, 2.0, (0, 0, 255), 2)
            frame = cv2.aruco.drawDetectedMarkers(frame, coners, ids)
            cv2.imshow('video', frame)
            k = cv2.waitKey(30) & 0xff
            if k == 27:
                break
        cam.release()

def main():
    allCorners, allIds, imsize = cal()
    ret, mtx, dist, rvec, tvec = calibrate_charuco(allCorners, allIds, imsize)
    detect_marker(mtx, dist)


if __name__ == "__main__":
    main()


