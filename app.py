from skimage.metrics import structural_similarity as ssim
import argparse
import imutils
import cv2
from skimage import data, img_as_float
import numpy as np


def find_contours(image_now):
    # thresh = cv2.threshold(diff, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)[1]
    cnts = cv2.findContours(image_now, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cnts = imutils.grab_contours(cnts)
    array_of_bounding_boxes = []
    for c in cnts:
        area = cv2.contourArea(c)
        area_min = 800
        area_max = 4000
        if area_min <= area <= area_max:
            (x, y, w, h) = cv2.boundingRect(c)
            # cv2.rectangle(image_now, (x, y), (x + w, y + h), colour, 2)
            array_of_bounding_boxes.append((x, y, w, h))

    return array_of_bounding_boxes


def calculate_ssim(img1, img2):
    (score, diff) = ssim(img1, img2, full=True)
    diff = (diff * 255).astype("uint8")
    return diff, score


def allign_photos(img1, img2, img1_color):
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
    res = cv2.bitwise_and(img_bgr, img_bgr, mask=mask)

    kernel = np.ones((3, 3), np.uint8)
    closing = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel=kernel, iterations=3)

    return closing


def get_pink_masked_photo(img_bgr):
    lower_pink = np.array([100, 50, 130])
    upper_pink = np.array([240, 200, 230])

    img_hsv = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV)

    mask = cv2.inRange(img_hsv, lower_pink, upper_pink)
    res = cv2.bitwise_and(img_bgr, img_bgr, mask=mask)
    kernel = np.ones((3, 3), np.uint8)
    closing = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel=kernel, iterations=5)

    return closing


def get_bounding_boxes(RECOVERY, GROWTH, DEATH, BLEACHING, image_now):
    counter = 0
    recovery_bounding_boxes = find_contours(RECOVERY)
    growth_bounding_boxes = find_contours(GROWTH)
    death_bounding_boxes = find_contours(DEATH)
    bleaching_bounding_boxes = find_contours(BLEACHING)
    print("RECOVER", recovery_bounding_boxes)
    print("GROWTH", growth_bounding_boxes)
    print("DEATH", death_bounding_boxes)
    print("BLEACHING", bleaching_bounding_boxes)

    if len(recovery_bounding_boxes) > 0:
        counter += 1
        for bounding_box in recovery_bounding_boxes:
            x = bounding_box[0]
            y = bounding_box[1]
            w = bounding_box[2]
            h = bounding_box[3]
            cv2.rectangle(image_now, (x, y), (x + w, y + h), (255, 0, 0), 2)

    if len(growth_bounding_boxes) > 0:
        counter += 1
        for bounding_box in growth_bounding_boxes:
            x = bounding_box[0]
            y = bounding_box[1]
            w = bounding_box[2]
            h = bounding_box[3]
            cv2.rectangle(image_now, (x, y), (x + w, y + h), (0, 255, 0), 2)

    if len(death_bounding_boxes) > 0:
        if counter < 2:
            counter += 1
            for bounding_box in death_bounding_boxes:
                x = bounding_box[0]
                y = bounding_box[1]
                w = bounding_box[2]
                h = bounding_box[3]
                cv2.rectangle(image_now, (x, y), (x + w, y + h), (0, 255, 255), 2)

    if len(bleaching_bounding_boxes) > 0:
        if counter < 2:
            counter += 1
            for bounding_box in bleaching_bounding_boxes:
                x = bounding_box[0]
                y = bounding_box[1]
                w = bounding_box[2]
                h = bounding_box[3]
                cv2.rectangle(image_now, (x, y), (x + w, y + h), (0, 0, 255), 2)
    return image_now


def main():
    image_before = cv2.imread("photos/before.png")
    image_now = cv2.imread("photos/now.png")
    # image_before = cv2.resize(image_before, (300, 300))
    # image_now = cv2.resize(image_now, (300, 300))
    # image_before = cv2.bilateralFilter(image_before, 3, 75, 75)
    # image_now = cv2.bilateralFilter(image_now, 3, 75, 75)
    grayA = cv2.cvtColor(image_before, cv2.COLOR_BGR2GRAY)
    grayB = cv2.cvtColor(image_now, cv2.COLOR_BGR2GRAY)

    transformed_image = allign_photos(grayA, grayB, image_before)
    # transformed_image_gray = cv2.cvtColor(transformed_image, cv2.COLOR_BGR2GRAY) # BEFORE IMAGE ALIGNED TO NOW
    #
    # cv2.imshow("BEFORE", image_before)
    # cv2.imshow("Modified", image_now)
    # cv2.imshow("Trans",transformed_image_gray)
    # cv2.waitKey(0)
    #
    # img_now_float = img_as_float(grayB) # B - now
    # before_alligned_float = img_as_float(transformed_image_gray) # A - before
    #
    #
    # img_now_float = cv2.resize(img_now_float, (480, 360))
    # before_alligned_float = cv2.resize(before_alligned_float, (480, 360))

    img_masked_white = get_white_masked_photo(image_now)
    img_masked_white_before_alligned = get_white_masked_photo(transformed_image)

    pink_masked_now = get_pink_masked_photo(image_now)
    pink_masked_alligned = get_pink_masked_photo(transformed_image)

    masked_now_merged = cv2.bitwise_or(img_masked_white, pink_masked_now)
    masked_before_merged = cv2.bitwise_or(img_masked_white_before_alligned, pink_masked_alligned)
    #
    # masked_differences_white = cv2.bitwise_xor(img_masked_white, img_masked_white_before_alligned)
    #
    # masked_or_white = cv2.bitwise_or(img_masked_white, img_masked_white_before_alligned)

    # now_difference = img_masked_white_before_alligned - img_masked_white
    # cv2.imshow("DIFFRENCE BETWEEN WHITES", now_difference)

    RECOVERY = cv2.bitwise_and(pink_masked_now, img_masked_white_before_alligned)
    cv2.imshow("RECOVERY", RECOVERY)

    BLEACHING = cv2.bitwise_and(pink_masked_alligned, img_masked_white)
    cv2.imshow("BLEACHING", BLEACHING)

    kernel = np.ones((7, 7), np.uint8)

    GROWTH = pink_masked_now - pink_masked_alligned - RECOVERY + BLEACHING
    GROWTH = cv2.morphologyEx(GROWTH, cv2.MORPH_OPEN, kernel, 10)
    cv2.imshow("GROWTH", GROWTH)

    DEATH = masked_before_merged - masked_now_merged - GROWTH
    DEATH = cv2.morphologyEx(DEATH, cv2.MORPH_OPEN, kernel, 10)
    cv2.imshow("DEATH", DEATH)
    #
    # cv2.imshow("Masked white", img_masked_white)
    # cv2.imshow("Alligned white", img_masked_white_before_alligned)
    #
    # cv2.imshow("MASKED PINK NOW", pink_masked_now)
    # cv2.imshow("MASKED PINK ALLIGNED", pink_masked_alligned)

    image_now_with_areas = get_bounding_boxes(RECOVERY, GROWTH, DEATH, BLEACHING, image_now)
    cv2.imshow("OUTLINED IMAGE", image_now_with_areas)
    cv2.imshow("BEFORE IMAGE", image_before)
    cv2.imshow("MERGED NOW", masked_now_merged)
    cv2.imshow("MERGED BEFORE", masked_before_merged)

    # cv2.imshow("DIFFERENCE WHITE", masked_differences_white)
    # cv2.imshow("OR WHITE", masked_or_white)
    cv2.waitKey(0)


if __name__ == "__main__":
    main()
