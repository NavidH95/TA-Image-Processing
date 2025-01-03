import cv2
import numpy as np

# Path to the video file
vid_path = 'walking.mp4'

# Play the video
cap = cv2.VideoCapture(vid_path)
if not cap.isOpened():
    print("Error: Unable to open video file")
    exit()

# Video information
frame_rate = int(cap.get(cv2.CAP_PROP_FPS))
total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

# Read the first frame as the reference frame
ret, ref_frame = cap.read()

if not ret:
    print("Error: Unable to read the first frame")
    exit()

# Display the reference frame
cv2.imshow("Reference Frame", ref_frame)
cv2.waitKey(0)
cv2.destroyAllWindows()

# Convert the reference frame to grayscale
ref_frame_gray = cv2.cvtColor(ref_frame, cv2.COLOR_BGR2GRAY)

# Threshold parameter
threshold_param = 0.1 

# Set up the video writer for the output video
fourcc = cv2.VideoWriter_fourcc(*'XVID')
out = cv2.VideoWriter('myVideo.avi', fourcc, frame_rate, (ref_frame.shape[1] * 2, ref_frame.shape[0]))

# Process each frame
for _ in range(total_frames):
    ret, cur_frame = cap.read()
    if not ret:
        break

    # Convert the current frame to grayscale
    cur_frame_gray = cv2.cvtColor(cur_frame, cv2.COLOR_BGR2GRAY)
    
    # Compute the difference between the current frame and the reference frame
    diff_img = cv2.absdiff(cur_frame_gray, ref_frame_gray)
    
    # Binarize the difference
    _, binarised_mask = cv2.threshold(diff_img, int(threshold_param * 255), 255, cv2.THRESH_BINARY)
    
    # Convert the binary mask to three channels
    binarised_mask_colored = cv2.merge([binarised_mask] * 3)
    
    # Combine the current frame and the binary mask
    combined_frame = np.hstack((cur_frame, binarised_mask_colored))
    
    # Write the combined frame to the output video
    out.write(combined_frame)

# Release resources
cap.release()
out.release()

# Play the output video
cap_out = cv2.VideoCapture('myVideo.avi')
while cap_out.isOpened():
    ret, frame = cap_out.read()
    if not ret:
        break
    cv2.imshow("Output Video", frame)
    if cv2.waitKey(30) & 0xFF == ord('q'):
        break

cap_out.release()
cv2.destroyAllWindows()
