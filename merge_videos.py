import imageio_ffmpeg
import subprocess
import os

ffmpeg_exe = imageio_ffmpeg.get_ffmpeg_exe()

def merge_av(vid_in, aud_in, out_file):
    if os.path.exists(vid_in) and os.path.exists(aud_in):
        print(f"Merging {vid_in} and {aud_in} into {out_file}...")
        # Re-encode video to h264 for maximum compatibility, copy audio
        cmd = [
            ffmpeg_exe, '-y',
            '-i', vid_in,
            '-i', aud_in,
            '-c:v', 'libx264',
            '-pix_fmt', 'yuv420p',
            '-c:a', 'copy',
            out_file
        ]
        subprocess.run(cmd, check=True)
        print(f"Successfully created {out_file}")
    else:
        print(f"Missing inputs for {out_file}")

# 1. Drop tutorial
merge_av(
    r"cv_model\training_data\side_videos\drop_tutorial_insight.f398.mp4",
    r"cv_model\training_data\side_videos\drop_tutorial_insight.f140.m4a",
    r"cv_model\training_data\side_videos\drop_tutorial_insight.mp4"
)

# 2. Side video 1
merge_av(
    r"cv_model\training_data\side_videos\side_video_1.mp4",
    r"cv_model\training_data\side_videos\side_video_1.f140.m4a",
    r"cv_model\training_data\side_videos\side_video_1_fixed.mp4"
)
