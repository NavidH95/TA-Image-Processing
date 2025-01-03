import cv2
import numpy as np


def draw_optical_flow(flow):
    h, w = flow.shape[:2]
    hsv = np.zeros((h, w, 3), dtype=np.uint8)
    hsv[..., 1] = 255  # Set saturation to maximum

    # Calculate flow magnitude and angle
    mag, ang = cv2.cartToPolar(flow[..., 0], flow[..., 1])
    print(mag)
    hsv[..., 0] = ang * 180 / np.pi / 2  # Hue corresponds to direction
    hsv[..., 2] = cv2.normalize(mag, None, 0, 255, cv2.NORM_MINMAX)  # Value corresponds to magnitude

    # Convert HSV to BGR for visualization
    bgr = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
    return bgr


def optical_flow_demo(video_path):
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print("Error: Cannot open video.")
        return

    ret, prev_frame = cap.read()
    if not ret:
        print("Error: Cannot read the first frame.")
        return

    prev_gray = cv2.cvtColor(prev_frame, cv2.COLOR_BGR2GRAY)

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # Calculate dense optical flow using Farneback method
        flow = cv2.calcOpticalFlowFarneback(prev_gray, gray, None,
                                            pyr_scale=0.5, levels=3, winsize=15,
                                            iterations=3, poly_n=5, poly_sigma=1.2,
                                            flags=0)

        # Visualize the optical flow
        flow_color = draw_optical_flow(flow)
        cv2.imshow('Optical Flow', flow_color)

        # Update previous frame
        prev_gray = gray

        if cv2.waitKey(20) & 0xFF == 27:  # Press 'ESC' to exit
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    video_path = "test.mp4"  
    optical_flow_demo(video_path)
