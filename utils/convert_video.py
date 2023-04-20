import moviepy.editor as mp

def gif_to_mp4(gif_path, mp4_path):
    clip = mp.VideoFileClip(gif_path)
    clip.write_videofile(mp4_path)