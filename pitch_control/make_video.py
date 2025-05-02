import cv2
import glob
import os

# Set your folder path
img_folder = "pitch_control/plots"
output_video = "pitch_control/videos/pitch_control_video.mp4"
fps = 5  # frames per second

# Get all images in sorted order
img_files = sorted(glob.glob(os.path.join(img_folder, "pitch_control_frame_*.png")))

# Load first image to get frame size
first_frame = cv2.imread(img_files[0])
height, width, _ = first_frame.shape

# Set up the video writer
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
video_writer = cv2.VideoWriter(output_video, fourcc, fps, (width, height))

# Write each image as a frame
for img_file in img_files:
    frame = cv2.imread(img_file)
    video_writer.write(frame)

video_writer.release()
print(f"✅ Video saved as {output_video}")