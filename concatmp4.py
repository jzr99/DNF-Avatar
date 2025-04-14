from moviepy.editor import VideoFileClip, clips_array

# Load your 4 video clips
clip1 = VideoFileClip("./resource/m11.mp4")
clip2 = VideoFileClip("./resource/m12.mp4")
clip3 = VideoFileClip("./resource/m13.mp4")
clip4 = VideoFileClip("./resource/m14.mp4")

# Resize all clips to the same height (to avoid alignment issues)
target_height = min(clip1.h, clip2.h, clip3.h, clip4.h)
clip1 = clip1.resize(height=target_height)
clip2 = clip2.resize(height=target_height)
clip3 = clip3.resize(height=target_height)
clip4 = clip4.resize(height=target_height)

# Trim to the shortest clip duration
min_duration = min(clip1.duration, clip2.duration, clip3.duration, clip4.duration)
clip1 = clip1.subclip(0, min_duration)
clip2 = clip2.subclip(0, min_duration)
clip3 = clip3.subclip(0, min_duration)
clip4 = clip4.subclip(0, min_duration)

# Concatenate horizontally
final_clip = clips_array([[clip1, clip2, clip3, clip4]])

# Export to GIF
# final_clip.write_gif("f1.gif", fps=25)  # You can change fps for smoother/slower animation
final_clip.write_videofile("m1.mp4", codec="libx264", audio_codec="aac")