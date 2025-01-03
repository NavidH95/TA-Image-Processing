import cv2
import numpy as np

def display_optical_flow(video_path, frame1_idx, frame2_idx):
    # Open the video file
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print("Cannot open video")
        return
    
    # Access the specified frames
    cap.set(cv2.CAP_PROP_POS_FRAMES, frame1_idx)
    ret, frame1 = cap.read()
    if not ret:
        print(f"Cannot read frame {frame1_idx}")
        return

    cap.set(cv2.CAP_PROP_POS_FRAMES, frame2_idx)
    ret, frame2 = cap.read()
    if not ret:
        print(f"Cannot read frame {frame2_idx}")
        return

    # Convert frames to grayscale
    gray1 = cv2.cvtColor(frame1, cv2.COLOR_BGR2GRAY)
    gray2 = cv2.cvtColor(frame2, cv2.COLOR_BGR2GRAY)

    # Calculate optical flow
    flow = cv2.calcOpticalFlowFarneback(gray1, gray2, None, 0.5, 3, 15, 3, 5, 1.2, 0)
    # Visualize the optical flow
    h, w = gray1.shape
    step = 16  # Distance between arrows
    y, x = np.mgrid[step//2:h:step, step//2:w:step].astype(int)
        
    # Match array dimensions
    fx, fy = flow[y, x, 0], flow[y, x, 1]  # Extract optical flow for specified coordinates
    lines = np.vstack([x.ravel(), y.ravel(), (x + fx).ravel(), (y + fy).ravel()]).T.reshape(-1, 2, 2)
    lines = np.int32(lines)

    # Draw arrows on the image
    vis = cv2.cvtColor(gray1, cv2.COLOR_GRAY2BGR)
    for (x1, y1), (x2, y2) in lines:
        cv2.arrowedLine(vis, (x1, y1), (x2, y2), (0, 255, 0), 1, tipLength=0.3)

    # Display the image
    cv2.imshow('Optical Flow', vis)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    cap.release()

# Use the function
if __name__ == "__main__":
    video_path = 'walking.mp4'  # Path to the video file
    frame1_idx = 50  # Index of the first frame
    frame2_idx = 60  # Index of the second frame
    display_optical_flow(video_path, frame1_idx, frame2_idx)
