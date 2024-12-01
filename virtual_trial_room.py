import cv2
import mediapipe as mp
import numpy as np

# Initialize Mediapipe Pose
mp_drawing = mp.solutions.drawing_utils
mp_pose = mp.solutions.pose

# Known width of the reference object in centimeters (e.g., a credit card is 8.5 cm)
REFERENCE_WIDTH_CM = 8.5

def compute_real_size(landmark1, landmark2, frame_shape, ref_pixel_width):
    # Calculate distance in pixels between two landmarks
    landmark1_px = (int(landmark1.x * frame_shape[1]), int(landmark1.y * frame_shape[0]))
    landmark2_px = (int(landmark2.x * frame_shape[1]), int(landmark2.y * frame_shape[0]))
    pixel_distance = np.linalg.norm(np.array(landmark2_px) - np.array(landmark1_px))
    
    # Calculate real-world distance in centimeters
    cm_per_pixel = REFERENCE_WIDTH_CM / ref_pixel_width
    real_distance_cm = pixel_distance * cm_per_pixel
    
    return real_distance_cm

def estimate_tshirt_size(shoulder_width_cm, waist_width_cm):
    # Example size estimation logic based on measurements
    if shoulder_width_cm < 40 and waist_width_cm < 70:
        return "Small"
    elif shoulder_width_cm < 45 and waist_width_cm < 80:
        return "Medium"
    elif shoulder_width_cm < 50 and waist_width_cm < 90:
        return "Large"
    else:
        return "Extra Large"

# Capture Video from Webcam
cap = cv2.VideoCapture(0)

with mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5) as pose:
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        
        # Convert the frame to RGB for Mediapipe processing
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = pose.process(frame_rgb)
        
        if results.pose_landmarks:
            # Draw pose landmarks on the frame
            mp_drawing.draw_landmarks(frame, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)
            
            # Assume reference object is visible and measured in pixels
            ref_pixel_width = 150  # Example value; replace with actual measurement

            # Compute body measurements in centimeters
            shoulder_width_cm = compute_real_size(results.pose_landmarks.landmark[11], results.pose_landmarks.landmark[12], frame.shape, ref_pixel_width)
            waist_width_cm = compute_real_size(results.pose_landmarks.landmark[23], results.pose_landmarks.landmark[24], frame.shape, ref_pixel_width)
            
            # Estimate T-shirt size
            tshirt_size = estimate_tshirt_size(shoulder_width_cm, waist_width_cm)
            
            # Display measurements and estimated size on the frame
            cv2.putText(frame, f'Shoulder Width: {shoulder_width_cm:.2f} cm', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
            cv2.putText(frame, f'Waist Width: {waist_width_cm:.2f} cm', (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
            cv2.putText(frame, f'T-Shirt Size: {tshirt_size}', (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)

        # Show the frame with annotations
        cv2.imshow('Virtual Trial Room', frame)
        
        # Exit the loop when 'q' or 'ESC' is pressed
        if cv2.waitKey(1) & 0xFF in [27, ord('q')]:
            break

# Release resources
cap.release()
cv2.destroyAllWindows()