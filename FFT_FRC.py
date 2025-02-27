import cv2
import numpy as np
import time
import matplotlib.pyplot as plt
import os  # For getting file sizes

# Apply FFT to a frame
def apply_fft(frame):
    return cv2.dft(np.float32(frame), flags=cv2.DFT_COMPLEX_OUTPUT)

# Apply Inverse FFT to a frequency frame
def apply_ifft(freq_frame):
    return cv2.idft(freq_frame, flags=cv2.DFT_SCALE)[:,:,0]

# Interpolate frames using a spatial domain interpolation
def interpolate_frames_spatial(prev_frame, next_frame, alpha):
    return cv2.addWeighted(prev_frame, 1 - alpha, next_frame, alpha, 0)

# Extract signal (intensity variation over time)
def extract_signal(video_path):
    cap = cv2.VideoCapture(video_path)
    frame_intensities = []
    frame_indices = []

    frame_idx = 0
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)  # Convert to grayscale
        avg_intensity = np.mean(gray_frame)  # Compute mean intensity
        frame_intensities.append(avg_intensity)
        frame_indices.append(frame_idx)
        frame_idx += 1
    
    cap.release()
    return frame_indices, frame_intensities

# Function to handle video input and output
def process_video(input_path, output_path, target_fps=90):
    # Open the video and get properties
    cap = cv2.VideoCapture(input_path)
    if not cap.isOpened():
        raise ValueError(f"Error: Unable to open the video file {input_path}")

    fps = cap.get(cv2.CAP_PROP_FPS)
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    # Set up the output video writer (using H264 codec for better quality)
    out = cv2.VideoWriter(output_path, cv2.VideoWriter_fourcc(*"H264"), target_fps, (frame_width, frame_height))

    # Start overall processing timer
    start_time = time.time()

    prev_frame = None
    frame_count = 0
    frame_process_times = []
    frame_sizes = []  # To store frame sizes

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        
        # Time the frame processing
        frame_start_time = time.time()

        # Calculate frame size (in bytes)
        frame_size = frame_width * frame_height * frame.shape[2]  # Width x Height x Channels
        frame_sizes.append(frame_size)

        if prev_frame is not None:
            # Interpolate frames
            for alpha in np.linspace(0, 1, num=2):  # Interpolate 2x (to target FPS)
                interpolated_frame = interpolate_frames_spatial(prev_frame, frame, alpha)
                out.write(interpolated_frame)  # Save the interpolated frame

        prev_frame = frame  # Update the previous frame
        frame_count += 1

        # Track processing time for the frame
        frame_end_time = time.time()
        frame_process_times.append(frame_end_time - frame_start_time)

    cap.release()
    out.release()

    # Total processing time
    end_time = time.time()
    total_processing_time = end_time - start_time
    avg_frame_process_time = np.mean(frame_process_times)
    total_frames_processed = frame_count * 2  # Because we add interpolated frames

    print(f"Total video processing completed in {total_processing_time:.2f} seconds.")
    print(f"Average time per frame: {avg_frame_process_time:.4f} seconds")
    print(f"Total frames processed (including interpolated): {total_frames_processed}")

    # Frame rate analysis
    cap_in = cv2.VideoCapture(input_path)
    input_fps = cap_in.get(cv2.CAP_PROP_FPS)
    print(f"Input Video FPS: {input_fps}")

    cap_out = cv2.VideoCapture(output_path)
    output_fps = cap_out.get(cv2.CAP_PROP_FPS)
    print(f"Output Video FPS: {output_fps}")

    # Show video sizes before and after conversion
    input_size = os.path.getsize(input_path) / (1024 * 1024)  # Size in MB
    output_size = os.path.getsize(output_path) / (1024 * 1024)  # Size in MB
    print(f"Input Video Size: {input_size:.2f} MB")
    print(f"Output Video Size: {output_size:.2f} MB")

    # Visualization of frame processing times and frame sizes
    fig, ax1 = plt.subplots()

    # Plot frame processing time on the primary y-axis
    ax1.plot(frame_process_times, color='blue', label='Frame Processing Time')
    ax1.set_xlabel('Frame Index')
    ax1.set_ylabel('Processing Time (seconds)', color='blue')
    ax1.tick_params(axis='y', labelcolor='blue')

    # Create a secondary y-axis to plot the frame sizes
    ax2 = ax1.twinx()
    ax2.plot(frame_sizes, color='red', label='Frame Size (bytes)')
    ax2.set_ylabel('Frame Size (bytes)', color='red')
    ax2.tick_params(axis='y', labelcolor='red')

    # Title and show the plot
    plt.title('Frame Processing Time and Frame Size')
    plt.show()

    # Signal analysis - Plot frame intensity over time
    input_frame_indices, input_signal = extract_signal(input_path)
    output_frame_indices, output_signal = extract_signal(output_path)

    plt.figure(figsize=(10, 5))
    plt.plot(input_frame_indices, input_signal, label="Input Video Signal (Original FPS)", color='blue')
    plt.plot(output_frame_indices, output_signal, label="Output Video Signal (Interpolated FPS)", color='red', linestyle='dashed')

    plt.xlabel("Frame Index")
    plt.ylabel("Average Frame Intensity")
    plt.title("Video Frame Signal Comparison")
    plt.legend()
    plt.show()

# Example usage
input_video_path = r"C:\Users\ADMIN\OneDrive\Documents\Documents\RUPP\code\Python\Frame Rate Conversion FFT\Test_video-uhd_2560_1440_30fps.mp4"
output_video_path = "output_60fps_video.mp4"

try:
    process_video(input_video_path, output_video_path, target_fps=60)
except Exception as e:
    print(str(e))
