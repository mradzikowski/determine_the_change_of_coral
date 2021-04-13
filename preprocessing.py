import cv2
import numpy as np
import imutils


def align_photos(img1, img2, img1_color):
    height, width = img2.shape

    # Create ORB detector with 5000 features.
    orb_detector = cv2.ORB_create(5000)

    # Find keypoints and descriptors.
    # The first arg is the image, second arg is the mask
    # (which is not required in this case).
    kp1, d1 = orb_detector.detectAndCompute(img1, None)
    kp2, d2 = orb_detector.detectAndCompute(img2, None)

    # Match features between the two images.
    # We create a Brute Force matcher with
    # Hamming distance as measurement mode.
    matcher = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)

    # Match the two sets of descriptors.
    matches = matcher.match(d1, d2)

    # Sort matches on the basis of their Hamming distance.
    matches.sort(key=lambda x: x.distance)

    # Take the top 90 % matches forward.
    matches = matches[:int(len(matches) * 90)]
    no_of_matches = len(matches)

    # Define empty matrices of shape no_of_matches * 2.
    p1 = np.zeros((no_of_matches, 2))
    p2 = np.zeros((no_of_matches, 2))

    for i in range(len(matches)):
        p1[i, :] = kp1[matches[i].queryIdx].pt
        p2[i, :] = kp2[matches[i].trainIdx].pt

    # Find the homography matrix.
    homography, mask = cv2.findHomography(p1, p2, cv2.RANSAC)

    # Use this matrix to transform the
    # colored image wrt the reference image.
    transformed_img = cv2.warpPerspective(img1_color,
                                          homography, (width, height))
    return transformed_img


def get_white_masked_photo(img_bgr):

    lower_white = np.array([180, 180, 80])
    upper_white = np.array([255, 255, 255])

    mask = cv2.inRange(img_bgr, lower_white, upper_white)

    kernel = np.ones((3, 3), np.uint8)
    closing = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel=kernel, iterations=3)

    return closing


def get_pink_masked_photo(img_bgr):
    lower_pink = np.array([100, 50, 130])
    upper_pink = np.array([240, 200, 230])

    img_hsv = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV)

    mask = cv2.inRange(img_hsv, lower_pink, upper_pink)
    kernel = np.ones((3, 3), np.uint8)
    closing = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel=kernel, iterations=5)

    return closing


def find_contours(image_now):
    cnts = cv2.findContours(image_now, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cnts = imutils.grab_contours(cnts)
    array_of_bounding_boxes = []
    for c in cnts:
        area = cv2.contourArea(c)
        area_min = 800
        area_max = 4000
        if area_min <= area <= area_max:
            (x, y, w, h) = cv2.boundingRect(c)
            array_of_bounding_boxes.append((x, y, w, h))
    return array_of_bounding_boxes