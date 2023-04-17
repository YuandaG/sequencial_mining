import cv2
import numpy as np
import matplotlib.pyplot as plt

# Load the video
cap = cv2.VideoCapture('clip.mp4')

# Read the first frame
ret, prev_frame = cap.read()

# Convert the frame to grayscale
prev_gray = cv2.cvtColor(prev_frame, cv2.COLOR_BGR2GRAY)

# Define the parameters for the optical flow algorithm
flow_params = dict(pyr_scale=0.5,
                   levels=2,
                   winsize=50,
                   iterations=3,
                   poly_n=7,
                   poly_sigma=1.5,
                   flags=0)

# Define the threshold for detecting a turn
threshold_mag = 0.7
threshold_ang = 2.2
count = 0
# ma = 5
# i = 0
plt_ang = []
plt_mag = []
# m = []
# a = []

while(cap.isOpened()):
    # Read the next frame
    ret, frame = cap.read()
    if not ret:
        break
    cv2.imshow('output', frame)

    if count % 1 == 0:
    # Convert the frame to grayscale
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # Calculate the optical flow using Farneback's algorithm
        flow = cv2.calcOpticalFlowFarneback(prev_gray, gray, None, **flow_params)
        flow[:,:,1] = 0
        # Convert the flow vectors from Cartesian to polar coordinates
        mag, ang = cv2.cartToPolar(flow[..., 0], flow[..., 1])

        # Compute the average direction and magnitude of the flow vectors
        avg_mag = np.mean(mag)
        avg_ang = np.mean(ang)

        # # Compare the average direction and magnitude with the previous frame
        # if 'prev_ang' in locals() and 'prev_mag' in locals():
        #     ang_diff = np.abs(avg_ang - prev_ang)
        #     mag_diff = np.abs(avg_mag - prev_mag)

        # Check if the vehicle is turning based on the threshold
        if avg_ang > threshold_ang and avg_mag > threshold_mag:
            print('The vehicle is turning!')

        # Update the previous frame and points for the next iteration
        prev_gray = gray
        prev_ang = avg_ang
        prev_mag = avg_mag
        plt_ang.append(avg_ang)
        plt_mag.append(avg_mag)

        # try:
        #     avg_mag_smoothed = np.mean(plt_ang[i-ma:i])
        #     avg_ang_smoothed = np.mean(plt_mag[i-ma:i])
        # except:
        #     pass
        
        # m.append(avg_mag_smoothed)
        # a.append(avg_ang_smoothed)

        # i += 1

    count += 1
    
    if count % 4000 == 0:
        plt.plot(plt_ang, label='ang')
        plt.plot(plt_mag, label='mag')
        plt.legend()
        plt.show()

    if cv2.waitKey(1) & 0xFF == ord('Q'):
      break

# Release the video capture and close all windows
cap.release()
cv2.destroyAllWindows()
