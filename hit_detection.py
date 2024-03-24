import cv2
import numpy as np
import pickle  # Pickle library for serializing Python objects
import cv2  # OpenCV library for computer vision tasks
import numpy as np  # NumPy library for numerical operations

 
######################################
cam_id = 0
width, height = 1920, 1080
map_file_path = "./map.p"

######################################
 
file_obj = open(map_file_path, 'rb')
map_points = pickle.load(file_obj)
file_obj.close()
print(f"Loaded map coordinates.")

# Open a connection to the webcam
 # For Webcam
# Set the width and height of the webcam frame

# Counter to keep track of how many polygons have been created
counter = 0

# Initialize video capture (0 for webcam, or provide video file path)
# Uncomment the line below for webcam
#cap = cv2.VideoCapture(1)
# Comment the line above and uncomment the line below for a video file
cap = cv2.VideoCapture('video.mp4')
cap.set(3, width)
cap.set(4, height)
# Counter to keep track of how many polygons have been created
counter = 0
# Define the lower and upper bounds of the yellow color in HSV
lower_yellow = np.array([20, 100, 100])
upper_yellow = np.array([30, 255, 255])

# Initialize variables
prev_position = None
is_ball_in_motion = False
distance_threshold = 55

# List to store hit points
hit_points = []

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Convert the frame to HSV
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    # Create a mask for the yellow color
    mask = cv2.inRange(hsv, lower_yellow, upper_yellow)

    # Apply morphological operations to reduce noise
    kernel = np.ones((5, 5), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)

    # Find contours in the mask
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Process each contour
    for contour in contours:
        # Filter contours based on area
        area = cv2.contourArea(contour)
        if area > 100:  # Adjust this threshold based on the size of your ball
            # Calculate the center of the contour
            M = cv2.moments(contour)
            if M["m00"] != 0:
                cx = int(M["m10"] / M["m00"])
                cy = int(M["m01"] / M["m00"])

                # Draw a circle at the center of the contour
                cv2.circle(frame, (cx, cy), 10, (0, 255, 0), -1)

                # Draw a circle around the detected ball
                cv2.circle(frame, (cx, cy), 20, (0, 0, 255), 2)

                # Check if the ball is in motion
                if prev_position is not None:
                    distance = np.linalg.norm(np.array([cx, cy]) - prev_position)
                    if distance > distance_threshold and not is_ball_in_motion:
                        is_ball_in_motion = True
                        print("Ball is in motion!")
                    elif distance <= distance_threshold and is_ball_in_motion:
                        is_ball_in_motion = False

                        # Record the hit point
                        print("Ball hit detected on the wall at ({}, {})".format(cx, cy))
                        hit_points.append((cx, cy))

                # Update the previous position
                prev_position = np.array([cx, cy])

    # Draw hit points on the frame
    for hit_point in hit_points:
        cv2.circle(frame, hit_point, 5, (255, 0, 0), -1)

    # Display the frame
    cv2.imshow('Frame', frame)

    # Break the loop if 'q' is pressed
    if cv2.waitKey(30) & 0xFF == ord('q'):
        break

# Release the video capture object and close all windows
cap.release()
cv2.destroyAllWindows()
