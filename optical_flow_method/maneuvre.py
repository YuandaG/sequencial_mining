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

'''
prev：	first 8-bit single-channel input image.
next：	second input image of the same size and the same type as prev.
flow：	computed flow image that has the same size as prev and type CV_32FC2.
pyr_scale：	parameter, specifying the image scale (<1) to build pyramids for each image; pyr_scale=0.5 means a classical pyramid, where each next layer is twice smaller than the previous one.
levels：	number of pyramid layers including the initial image; levels=1 means that no extra layers are created and only the original images are used.
winsize：	averaging window size; larger values increase the algorithm robustness to image noise and give more chances for fast motion detection, but yield more blurred motion field.
iterations：	number of iterations the algorithm does at each pyramid level.
poly_n：	size of the pixel neighborhood used to find polynomial expansion in each pixel; larger values mean that the image will be approximated with smoother surfaces, yielding more robust algorithm and more blurred motion field, typically poly_n =5 or 7.
poly_sigma：	standard deviation of the Gaussian that is used to smooth derivatives used as a basis for the polynomial expansion; for poly_n=5, you can set poly_sigma=1.1, for poly_n=7, a good value would be poly_sigma=1.5.
flags：	operation flags that can be a combination of the following:
'''



flow_params = dict(pyr_scale=0.5,
                   levels=1,
                   winsize=10,
                   iterations=3,
                   poly_n=7,
                   poly_sigma=1.5,
                   flags=0)

# Initialize the sum gradient
sum_gradient = []
sum_gradient_lr = []

i = 0


# vedio_width = int(cap.get(3))
# vedio_hight = int(cap.get(4))
# vedio_fps = int(cap.get(5))
 
# video_cod = cv2.VideoWriter_fourcc(*'MP4V')
# video_output= cv2.VideoWriter('flow_day.mp4',
#                       video_cod,
#                       vedio_fps,
#                       (vedio_width,vedio_hight))


count = 0

while(cap.isOpened()):
    # Read the next frame
    ret, frame = cap.read()
    if not ret:
        break

    # Process every 3 frame.
    if count % 3 == 0:

        # Convert the frame to grayscale
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # Calculate the dense optical flow using Farneback's algorithm
        flow = cv2.calcOpticalFlowFarneback(prev_gray, gray, None, **flow_params)
        flow[:,:,1] = 0

        # Compute the gradient magnitude and direction for each pixel
        gx = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
        gy = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)
        mag, ang = cv2.cartToPolar(gx, gy)

        # gx_l = gx[:,0:int(gx.shape[1]/2)]
        # gx_r = gx[:,int(-gx.shape[1]/2):]
        # gx = np.append(gx_l, gx_r, axis=1)
        # gy_l = gy[:,0:int(gx.shape[1]/2)]*0
        # gy_r = gy[:,int(gx.shape[1]/2):]*0
        # gx = np.append(gy_l, gy_r, axis=1)

        # Compute the dot product of the optical flow vectors and gradient vectors
        dot_product = np.sum(flow * np.dstack((gx, gy)), axis=2)

        # Filter the dot product using the gradient magnitude
        mask = np.logical_and(mag > 10, mag < 15)
        # mask = mag > 10
        dot_product_filtered = dot_product[mask]
        mag_filtered = mag[mask]
        # Compute the sum gradient
        sum_gradient.append(np.sum(dot_product_filtered / mag_filtered))

        # Update the previous frame and points for the next iteration
        prev_gray = gray

        i += 1
        # print('Sum gradient:', sum_gradient[-1])

        # Display the footage
        cv2.imshow('output', frame)


        hsv = np.zeros_like(frame)
        hsv[..., 1] = 255
        mag, ang = cv2.cartToPolar(flow[..., 0], flow[..., 1])
        hsv[..., 0] = ang * 180 / np.pi / 2
        hsv[..., 2] = cv2.normalize(mag, None, 0, 255, cv2.NORM_MINMAX)
        rgb = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
        # cv2.imshow('Dense Optical Flow', rgb)

        try:
            avg = np.mean(sum_gradient[i-3:i])
            print('Sum gradient:', avg)
        except:
            pass

    count += 1

    if count % 4000 == 0:
        plt.plot(sum_gradient)
        plt.show()
    # Press Q on keyboard to stop recording
    if cv2.waitKey(1) & 0xFF == ord('Q'):
      break




    # try:
    #     if i % 3 == 0:
    #         avg = np.mean(sum_gradient_lr[i-3:i])
    #         print('Sum gradient:', avg)
    # except:
    #     pass

    # video_output.write(frame)



# Release the video capture and close all windows
cap.release()
cv2.destroyAllWindows()

