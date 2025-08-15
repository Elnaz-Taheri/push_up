import cv2
import mediapipe as mp
import numpy as np

mp_drawing = mp.solutions.drawing_utils
mp_pose = mp.solutions.pose

# Function to calculate the angle between three points
def calculate_angle(a, b, c):
    a = np.array(a)
    b = np.array(b)
    c = np.array(c)
    
    radians = np.arctan2(c[1] - b[1], c[0] - b[0]) - np.arctan2(a[1] - b[1], a[0] - b[0])
    angle = np.abs(radians * 180.0 / np.pi)
    if angle > 180.0:
        angle = 360 - angle
    return angle

cap = cv2.VideoCapture("push_up.mp4")

# Push-up counter variables
counter = 0
stage = None

# Set up Mediapipe Pose instance
with mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5) as pose:
    while cap.isOpened():
        ret, frame = cap.read()

        # Convert the frame to RGB
        image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        image.flags.writeable = False

        # Process the frame to detect pose landmarks
        results = pose.process(image)
       
        # Convert back to BGR for OpenCV display
        image.flags.writeable = True
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        
        try:
            landmarks = results.pose_landmarks.landmark
            
            # Get coordinates for left shoulder, elbow, and wrist
            shoulder = [landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].x,
                        landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].y]
            elbow = [landmarks[mp_pose.PoseLandmark.LEFT_ELBOW.value].x,
                     landmarks[mp_pose.PoseLandmark.LEFT_ELBOW.value].y]
            wrist = [landmarks[mp_pose.PoseLandmark.LEFT_WRIST.value].x,
                     landmarks[mp_pose.PoseLandmark.LEFT_WRIST.value].y]
            
            # Calculate elbow angle
            angle = calculate_angle(shoulder, elbow, wrist)
            
            # Display the angle on the frame
            cv2.putText(image, str(int(angle)),
                        tuple(np.multiply(elbow, [900, 500]).astype(int)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,0, 0), 2)

            # Push-up counting logic
            if angle > 160:
                stage = "up"
            if angle < 90 and stage == 'up':
                stage = "down"
                counter += 1
                print("Push-ups:", counter)
                
        except:
            pass
        
        # Render curl counter
        # Setup status box
        cv2.rectangle(image, (0,0), (225,73), (245,117,16), -1)
        cv2.rectangle(image, (image.shape[1] - (image.shape[1]//10), (image.shape[0]//10)),(image.shape[1] - (image.shape[1]//200), (image.shape[0] - image.shape[0]//10)),(255,100,100),3)
        # Draw progress bar for angle
        bar_x1 = (image.shape[1] - (image.shape[1] // 10)) + 5
        bar_y1 = (image.shape[0] // 10) + 5
        bar_x2 = image.shape[1] - (image.shape[1] // 200) - 5
        bar_y2 = image.shape[0] - image.shape[0] // 10 - 5

        # Calculate filled bar height based on angle
        bar_height = bar_y2 - bar_y1
        filled_height = bar_y2 - int(angle * (bar_height / 180.0))
        # filled_y = bar_y2 - filled_height
        bar_y1 = bar_y2 - filled_height
        cv2.rectangle(image, (bar_x1, bar_y1), (bar_x2, bar_y2), (0, 200, 0), -1)
        # Rep data
        cv2.putText(image, 'REPS', (15,12), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,0,0), 1, cv2.LINE_AA)
        cv2.putText(image, str(counter), 
                    (10,60), 
                    cv2.FONT_HERSHEY_SIMPLEX, 2, (255,255,255), 2, cv2.LINE_AA)
        
        # Stage data
        cv2.putText(image, 'STAGE', (65,12), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,0,0), 1, cv2.LINE_AA)
        cv2.putText(image, stage, 
                    (60,60), 
                    cv2.FONT_HERSHEY_SIMPLEX, 2, (255,255,255), 2, cv2.LINE_AA)

        # Draw pose landmarks
        mp_drawing.draw_landmarks(image, results.pose_landmarks, mp_pose.POSE_CONNECTIONS,
                                  mp_drawing.DrawingSpec(color=(245,117,66), thickness=2, circle_radius=5),
                                  mp_drawing.DrawingSpec(color=(245,66,230), thickness=2, circle_radius=5))
        
        # Display the frame
        cv2.imshow('Push-up Counter', image)

        if cv2.waitKey(10) & 0xFF == ord('q'):
            break

cap.release()
cv2.destroyAllWindows()
