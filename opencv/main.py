import cv2
import numpy as np

def calculate_distance(x1, y1, x2, y2):
    return np.sqrt((x2 - x1)**2 + (y2 - y1)**2)
def process_image(image_path):
    # 1. Load the image
    image = cv2.imread(image_path)
    original_image = image.copy()

    # 2. Convert the image to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    cv2.imwrite('gray_1.jpg', gray)

    # 3. Apply Gaussian Blur to the grayscale image
    blurred = cv2.GaussianBlur(gray, (9, 9), 0)
    cv2.imwrite('filter_1.jpg', blurred)

    # 4. Edge detection using Canny (automatic thresholding)
    edges = cv2.Canny(blurred, 100, 140)
    cv2.imwrite('edge_1.jpg', edges)

    # 5. Optional Binarization (If not using Canny, you could use thresholding)
    _, binary = cv2.threshold(gray, 160, 255, cv2.THRESH_BINARY)
    cv2.imwrite('bin_1.jpg', binary)

    # 6. Morphological transformations (e.g., Dilation to fill gaps in edges)
    kernel = np.ones((3, 3), np.uint8)
    morphed = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, kernel)
    cv2.imwrite('morphology_1.jpg', morphed)

    # 7. Line detection using Hough Transform
    lines = cv2.HoughLinesP(morphed, 1, np.pi / 180, 100, minLineLength=110, maxLineGap=40)

    # 8. Draw the detected lines on the original image
    # Define a minimum distance threshold between lines
    MIN_DISTANCE_THRESHOLD = 118

    # List to keep track of drawn lines
    drawn_lines = []

    if lines is not None:
        for line in lines:
            x1, y1, x2, y2 = line[0]

            # Calculate the distance between the current line and all previously drawn lines
            should_draw = True
            for drawn_line in drawn_lines:
                dx1, dy1, dx2, dy2 = drawn_line
                if calculate_distance(x1, y1, dx1, dy1) < MIN_DISTANCE_THRESHOLD and calculate_distance(x2, y2, dx2,
                                                                                                        dy2) < MIN_DISTANCE_THRESHOLD:
                    should_draw = False
                    break

            # Only draw the line if it's not too close to previously drawn lines
            if should_draw:
                cv2.line(original_image, (x1, y1), (x2, y2), (0, 255, 0), 5)
                drawn_lines.append((x1, y1, x2, y2))

    # Save the image with filtered lane lines
    cv2.imwrite('line_1.jpg', original_image)

    # Display the final output
    cv2.imshow('Lane Lines', original_image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


# Example usage with one of your images
process_image('img/1.jpg')
