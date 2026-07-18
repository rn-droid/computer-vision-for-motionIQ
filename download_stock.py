import requests
import re
import urllib.request
import os
import ssl

ssl._create_default_https_context = ssl._create_unverified_context

url = 'https://pixabay.com/videos/search/badminton/'
headers = {'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'}
print("Searching Pixabay for badminton videos...")
response = requests.get(url, headers=headers)
html = response.text

# Pixabay embeds mp4 urls in the page source
matches = re.findall(r'https://cdn\.pixabay\.com/video/[^"\'\s]+\.mp4', html)

if matches:
    video_url = matches[0]
    print(f"Found video URL: {video_url}")
    out_path = os.path.join('cv_model', 'training_data', 'side_videos', 'stock_badminton.mp4')
    print(f"Downloading to {out_path}...")
    req = urllib.request.Request(video_url, headers=headers)
    with urllib.request.urlopen(req) as response, open(out_path, 'wb') as out_file:
        data = response.read()
        out_file.write(data)
    print("Download complete!")
else:
    print("No video found on Pixabay.")

# Try pexels if pixabay failed
if not matches:
    url = 'https://www.pexels.com/search/videos/badminton/'
    print("Searching Pexels for badminton videos...")
    response = requests.get(url, headers=headers)
    matches = re.findall(r'https://player\.vimeo\.com/external/[^"\'\s]+\.mp4[^"\'\s]*', response.text)
    if matches:
        video_url = matches[0]
        print(f"Found video URL: {video_url}")
        out_path = os.path.join('cv_model', 'training_data', 'side_videos', 'stock_badminton.mp4')
        print(f"Downloading to {out_path}...")
        req = urllib.request.Request(video_url, headers=headers)
        with urllib.request.urlopen(req) as response, open(out_path, 'wb') as out_file:
            data = response.read()
            out_file.write(data)
        print("Download complete!")
    else:
        print("No video found on Pexels.")
