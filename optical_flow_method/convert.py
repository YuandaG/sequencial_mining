from moviepy.editor import *

# Replace the file paths with the actual file paths
input_path = "/Users/yuandagao/Downloads/output_3-0.avi"
output_path = "/Users/yuandagao/Downloads/demo_3.mov"

# Load the input video file
video = VideoFileClip(input_path)

# Write the output video file in the MOV format
video.write_videofile(output_path, codec="libx264", audio_codec="aac", preset="ultrafast")
