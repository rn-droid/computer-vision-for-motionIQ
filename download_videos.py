#!/usr/bin/env python3
"""
Download BWF match videos referenced by the ShuttleSet dataset.
Uses yt-dlp to download from YouTube URLs listed in match.csv.
"""
import csv
import os
import subprocess
import sys

MATCH_CSV = os.path.join(os.path.dirname(__file__), "..", "temp_coachAI", "ShuttleSet", "set", "match.csv")
VIDEO_DIR = os.path.join(os.path.dirname(__file__), "training_data", "videos")


def download_videos(max_videos=5):
    """Download match videos from YouTube. Start with a small subset."""
    os.makedirs(VIDEO_DIR, exist_ok=True)

    with open(MATCH_CSV, "r", encoding="utf-8-sig") as f:
        matches = list(csv.DictReader(f))

    # Deduplicate by video name (multiple sets share same video)
    seen = set()
    unique_matches = []
    for m in matches:
        vid_name = m["video"]
        if vid_name not in seen:
            seen.add(vid_name)
            unique_matches.append(m)

    print(f"Found {len(unique_matches)} unique match videos in ShuttleSet")
    print(f"Downloading first {max_videos} videos...")

    downloaded = 0
    for m in unique_matches[:max_videos]:
        vid_name = m["video"]
        url = m["url"]
        output_path = os.path.join(VIDEO_DIR, f"{vid_name}.mp4")

        if os.path.exists(output_path):
            print(f"  [SKIP] {vid_name} (already downloaded)")
            downloaded += 1
            continue

        print(f"  [DOWNLOAD] {vid_name}")
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
                timeout=600,  # 10 min max per video
            )
            downloaded += 1
            print(f"    Saved: {output_path}")
        except subprocess.CalledProcessError as e:
            print(f"    Failed: {e}")
        except FileNotFoundError:
            print("    yt-dlp not found! Install with: pip install yt-dlp")
            sys.exit(1)

    print(f"\nDownloaded {downloaded}/{max_videos} videos to {VIDEO_DIR}")
    return downloaded


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Download ShuttleSet match videos")
    parser.add_argument("--max", type=int, default=5, help="Max videos to download (default 5)")
    args = parser.parse_args()
    download_videos(max_videos=args.max)
