import cv2
import numpy as np

from preprocessing import align_photos, get_white_masked_photo, get_pink_masked_photo, find_contours

BLUE = (255, 0, 0)
GREEN = (0, 255, 0)
YELLOW = (0, 255, 255)
RED = (0, 0, 255)


def draw_bounding_boxes(boxes, colour, image_now):
    for box in boxes:
        x = box[0]
        y = box[1]
        w = box[2]
        h = box[3]
        cv2.rectangle(image_now, (x, y), (x + w, y + h), colour, 2)
    return image_now


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
        image_now = draw_bounding_boxes(recovery_bounding_boxes, BLUE, image_now)

    if len(growth_bounding_boxes) > 0:
        counter += 1
        image_now = draw_bounding_boxes(growth_bounding_boxes, GREEN, image_now)

    if len(death_bounding_boxes) > 0:
        if counter < 2:
            counter += 1
            image_now = draw_bounding_boxes(death_bounding_boxes, YELLOW, image_now)

    if len(bleaching_bounding_boxes) > 0:
        if counter < 2:
            counter += 1
            image_now = draw_bounding_boxes(bleaching_bounding_boxes, RED, image_now)
    return image_now


def main():
    image_before = cv2.imread("photos/before.png")
    image_now = cv2.imread("photos/now.png")
    image_before = cv2.resize(image_before, (480, 360))
    image_now = cv2.resize(image_now, (480, 360))
    image_before = cv2.bilateralFilter(image_before, 3, 75, 75)
    image_now = cv2.bilateralFilter(image_now, 3, 75, 75)
    gray_before = cv2.cvtColor(image_before, cv2.COLOR_BGR2GRAY)
    gray_now = cv2.cvtColor(image_now, cv2.COLOR_BGR2GRAY)

    transformed_image = align_photos(gray_before, gray_now, image_before)

    img_masked_white = get_white_masked_photo(image_now)
    img_masked_white_before_aligned = get_white_masked_photo(transformed_image)

    pink_masked_now = get_pink_masked_photo(image_now)
    pink_masked_aligned = get_pink_masked_photo(transformed_image)

    masked_now_merged = cv2.bitwise_or(img_masked_white, pink_masked_now)
    masked_before_merged = cv2.bitwise_or(img_masked_white_before_aligned, pink_masked_aligned)

    RECOVERY = cv2.bitwise_and(pink_masked_now, img_masked_white_before_aligned)
    cv2.imshow("RECOVERY", RECOVERY)

    BLEACHING = cv2.bitwise_and(pink_masked_aligned, img_masked_white)
    cv2.imshow("BLEACHING", BLEACHING)

    kernel = np.ones((7, 7), np.uint8)

    GROWTH = pink_masked_now - pink_masked_aligned - RECOVERY + BLEACHING
    GROWTH = cv2.morphologyEx(GROWTH, cv2.MORPH_OPEN, kernel, 10)
    cv2.imshow("GROWTH", GROWTH)

    DEATH = masked_before_merged - masked_now_merged
    DEATH = cv2.morphologyEx(DEATH, cv2.MORPH_OPEN, kernel, 10)
    cv2.imshow("DEATH", DEATH)

    image_now_with_areas = get_bounding_boxes(RECOVERY, GROWTH, DEATH, BLEACHING, image_now)
    cv2.imshow("OUTLINED IMAGE", image_now_with_areas)
    cv2.imshow("BEFORE IMAGE", image_before)
    cv2.imshow("MERGED NOW", masked_now_merged)
    cv2.imshow("MERGED BEFORE", masked_before_merged)
    cv2.waitKey(0)


if __name__ == "__main__":
    main()
