# -*- coding: utf-8 -*-
"""
Created on Sat Jul  6 19:03:38 2024

@author: Suneera Wahab
"""
import cv2
import numpy as np

def get_quadrant(x, y, frame_width, frame_height):
    adjusted_width = int(frame_width * 0.7)  # Only consider the right 70% of the frame
    if x >= frame_width * 0.3:  # Only consider x >= 30% of the frame width
        x -= int(frame_width * 0.3)  # Adjust x to the new frame of reference
        if x < adjusted_width / 2 and y < frame_height / 2:
            return 1
        elif x >= adjusted_width / 2 and y < frame_height / 2:
            return 2
        elif x < adjusted_width / 2 and y >= frame_height / 2:
            return 3
        else:
            return 4
    return None  # Outside of the quadrants area

def detect_balls(frame):
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    # Define color ranges for detection (HSV format)
    colors = {
        'orange': [(5, 100, 100), (15, 255, 255)],
        'yellow': [(20, 100, 100), (30, 255, 255)],
        'dark green': [(35, 40, 40), (85, 255, 255)],
        'white': [(10, 10, 200), (180, 40, 255)]
    }

    detections = []

    for color, (lower, upper) in colors.items():
        mask = cv2.inRange(hsv, np.array(lower), np.array(upper))
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((5, 5), np.uint8))
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((5, 5), np.uint8))
        
        contours, _ = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

        for contour in contours:
            area = cv2.contourArea(contour)
            if area > 400:  # Adjust threshold as needed
                perimeter = cv2.arcLength(contour, True)
                circularity = 4 * np.pi * (area / (perimeter * perimeter))
                if circularity > 0.7:  # Circularity threshold to filter out non-circular objects
                    (x, y), radius = cv2.minEnclosingCircle(contour)
                    center = (int(x), int(y))
                    detections.append((center, color))

    return detections

def main(video_path, output_video_path, output_txt_path):
    cap = cv2.VideoCapture(video_path)
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    event_log = []

    tracked_objects = {}

    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    out = cv2.VideoWriter(output_video_path, fourcc, fps, (frame_width, frame_height))

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        detections = detect_balls(frame)
        current_time = cap.get(cv2.CAP_PROP_POS_MSEC) / 1000  # Current time in seconds

        new_tracked_objects = {}

        for (center, color) in detections:
            x, y = center
            quadrant = get_quadrant(x, y, frame_width, frame_height)
            if quadrant is None:
                continue
            obj_id = f"{color}_{quadrant}"

            if obj_id not in tracked_objects:
                event_log.append((current_time, quadrant, color, 'Entry'))
                cv2.putText(frame, f"Entry {current_time:.2f}s", (x, y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
            new_tracked_objects[obj_id] = (center, quadrant, color, 'Entry')

        # Check for exits
        for obj_id, (prev_center, prev_quadrant, prev_color, prev_event) in tracked_objects.items():
            if obj_id not in new_tracked_objects:
                event_log.append((current_time, prev_quadrant, prev_color, 'Exit'))
                cv2.putText(frame, f"Exit {current_time:.2f}s", prev_center, cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)

        for (center, color) in detections:
            cv2.circle(frame, center, 10, (255, 255, 255), 2)
            cv2.putText(frame, color, center, cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)

        tracked_objects = new_tracked_objects

        out.write(frame)

    cap.release()
    out.release()
    cv2.destroyAllWindows()

    # Write event log to file
    with open(output_txt_path, 'w') as f:
        for event in event_log:
            f.write(f"{event[0]:.2f}, {event[1]}, {event[2]}, {event[3]}\n")



if __name__ == "__main__":
    video_path = 'C:/Users/Suneera Wahab/Downloads/assignment3_v1.0/input_video.mp4'  # Replace with your video file path
    output_video_path = 'C:/Users/Suneera Wahab/Downloads/assignment3_v1.0/output_video.avi'  # Replace with desired output video file path
    output_txt_path = 'C:/Users/Suneera Wahab/Downloads/assignment3_v1.0/event_log.txt'  # Replace with desired output text file path
    main(video_path, output_video_path, output_txt_path)
