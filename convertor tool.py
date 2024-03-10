#conversion of sd video to hd video by increasing resolution/no of pixels
#using cv,diffusion and incrasing rsolution frame by frame

import cv2
import numpy as np
import time
import concurrent.futures

def process_frame(frame, scale_factor):
    # Resize the frame
    resized_frame = cv2.resize(frame, None, fx=scale_factor, fy=scale_factor)

    # Convert frames to grayscale
    gray_old = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    gray_new = cv2.cvtColor(resized_frame, cv2.COLOR_BGR2GRAY)

    # Ensure both frames have the same size
    gray_old = cv2.resize(gray_old, resized_frame.shape[:2][::-1])

    # Calculate dense optical flow
    flow = cv2.calcOpticalFlowFarneback(gray_old, gray_new, None, 0.5, 3, 15, 3, 5, 1.2, 0)

    # Use the flow to generate intermediate frames
    h, w = flow.shape[:2]
    flow_map = np.column_stack((np.repeat(np.arange(h), w), np.tile(np.arange(w), h)))
    displacement = flow_map + flow.reshape(-1, 2)
    displacement = np.clip(displacement, 0, (h - 1, w - 1))
    intermediate_frame = resized_frame[displacement[:, 0].astype(int), displacement[:, 1].astype(int)].reshape(h, w, 3)

    return intermediate_frame.astype(np.uint8)

def upscale_video_with_diffusion(input_path, output_path, scale_factor):
    start_time = time.time()  # Record the start time

    # Open the video file
    cap = cv2.VideoCapture(input_path)

    # Get the original video's properties
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    # Calculate the new frame size
    new_width = int(width * scale_factor)
    new_height = int(height * scale_factor)

    # Define the codec and create VideoWriter object for MP4
    fourcc = cv2.VideoWriter_fourcc(*'H264')
    out = cv2.VideoWriter(output_path, fourcc, fps, (new_width, new_height))

    def process_video():
        while True:
            ret, frame = cap.read()

            if not ret:
                break  # Break the loop if there are no more frames

            intermediate_frame = process_frame(frame, scale_factor)

            # Write the intermediate frame to the output video file
            out.write(intermediate_frame)

    # Set up threading for video processing
    with concurrent.futures.ThreadPoolExecutor() as executor:
        executor.submit(process_video)

    # Release the video capture and writer objects
    cap.release()
    out.release()

    end_time = time.time()  # Record the end time
    time_taken = end_time - start_time  # Calculate time taken

    print(f"Time taken to process the video: {time_taken:.2f} seconds")

# Example usage
upscale_video_with_diffusion('b1.mp4', 'output_video_diffusion.mp4', 2.0)
