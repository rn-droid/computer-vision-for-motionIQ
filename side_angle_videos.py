#!/usr/bin/env python3
"""
Download side-angle badminton training videos from YouTube.
These videos show players from a side/perpendicular view, ideal for
biomechanical analysis of shot types (smash, clear, drop).

Uses yt-dlp to download curated coaching/tutorial videos that demonstrate
specific shot techniques from a side angle.
"""
import os
import subprocess
import sys

VIDEO_DIR = os.path.join(os.path.dirname(__file__), "training_data", "side_videos")

# Curated list of side-angle badminton videos
# Format: (youtube_url, output_name, description)
SIDE_ANGLE_VIDEOS = [
    # ── All three shots compared (side-by-side view) ──
    (
        "https://youtu.be/J62n_y84WkY",
        "overhead_clear_smash_drop_comparison",
        "Badminton Famly: Overhead shots comparison (clear, smash, drop) - side angle",
    ),
    # ── Smash tutorials (side angle demonstrations) ──
    (
        "https://youtu.be/004r4w6577U",
        "smash_tutorial_insight",
        "Badminton Insight: Smash tutorial with side-view biomechanics",
    ),
    (
        "https://youtu.be/Qy-lD21c-P0",
        "smash_biomechanics_famly",
        "Badminton Famly: 5 biomechanics tips for smash - side angle",
    ),
    # ── Clear shot tutorials (side angle demonstrations) ──
    (
        "https://youtu.be/T4n0Vd15t4A",
        "clear_6steps_famly",
        "Badminton Famly: Perfect clear in 6 steps - side view",
    ),
    (
        "https://youtu.be/d4pYn-r5U-M",
        "forehand_clear_insight",
        "Badminton Insight: Forehand clear tutorial - side angle",
    ),
    # ── Drop shot tutorials (side angle demonstrations) ──
    (
        "https://youtu.be/u--taRfMoTs",
        "drop_tutorial_insight",
        "Badminton Insight: Drop shot technique and practices - side view",
    ),
    (
        "https://youtu.be/4e96N0n_W-c",
        "drop_shot_basic_feather",
        "Basic Feather: Drop shot do's and don'ts - side angle",
    ),
]


def download_side_videos(max_videos: int = None):
    """Download side-angle training videos from YouTube."""
    os.makedirs(VIDEO_DIR, exist_ok=True)

    videos_to_download = SIDE_ANGLE_VIDEOS
    if max_videos:
        videos_to_download = videos_to_download[:max_videos]

    print(f"Side-angle video download")
    print(f"  Output dir: {VIDEO_DIR}")
    print(f"  Videos to download: {len(videos_to_download)}")
    print()

    downloaded = 0
    for url, name, description in videos_to_download:
        output_path = os.path.join(VIDEO_DIR, f"{name}.mp4")

        if os.path.exists(output_path):
            print(f"  [SKIP] {name} (already downloaded)")
            downloaded += 1
            continue

        print(f"  [DOWNLOAD] {name}")
        print(f"    {description}")
        print(f"    URL: {url}")

        try:
            subprocess.run(
                [
                    "yt-dlp",
                    "-f", "bestvideo[height<=720][ext=mp4]+bestaudio[ext=m4a]/best[height<=720][ext=mp4]/best",
                    "--merge-output-format", "mp4",
                    "-o", output_path,
                    "--no-playlist",
                    url,
                ],
                check=True,
                timeout=600,
            )
            downloaded += 1
            print(f"    [SUCCESS] Saved: {output_path}")
        except subprocess.CalledProcessError as e:
            print(f"    [FAILED] {e}")
        except FileNotFoundError:
            print("    [FAILED] yt-dlp not found! Install with: pip install yt-dlp")
            sys.exit(1)

    print(f"\nDownloaded {downloaded}/{len(videos_to_download)} videos to {VIDEO_DIR}")
    return downloaded


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Download side-angle badminton training videos")
    parser.add_argument("--max", type=int, default=None, help="Max videos to download (default: all)")
    args = parser.parse_args()
    download_side_videos(max_videos=args.max)
