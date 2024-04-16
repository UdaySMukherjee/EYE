import cv2
import os

# Function to find the centroid of a contour
def find_centroid(contour):
    M = cv2.moments(contour)
    if M["m00"] != 0:
        cx = int(M["m10"] / M["m00"])
        cy = int(M["m01"] / M["m00"])
        return (cx, cy)
    else:
        return None

# Path to the folder containing images
folder_path = "diabetic_retinopathy_manualsegm/"

# Iterate over each image in the folder
for filename in os.listdir(folder_path):
    if filename.endswith(".tif") or filename.endswith(".png"):  # Adjust file extensions as needed
        # Read the image
        image_path = os.path.join(folder_path, filename)
        image = cv2.imread(image_path)

        # Convert the image to grayscale
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        # Threshold the image to create a binary mask
        _, mask = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY)

        # Find contours in the binary mask
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        # Find the largest contour
        largest_contour = max(contours, key=cv2.contourArea)

        # Find the centroid of the largest contour
        centroid = find_centroid(largest_contour)

        if centroid is not None:
            # Draw a line through the centroid to represent the middle line
            cv2.line(image, (centroid[0], 0), (centroid[0], image.shape[0]), (0, 255, 0), 2)

            # Display the image with the middle line
            cv2.imshow("Middle Line", image)
            cv2.waitKey(0)

# Close all OpenCV windows
cv2.destroyAllWindows()
