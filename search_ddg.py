import requests
import re

html = requests.get('https://html.duckduckgo.com/html/?q=badminton+side+angle+mp4+site%3Aarchive.org', headers={'User-Agent': 'Mozilla/5.0'}).text
urls = re.findall(r'https?://[^\s"\'<>]+(?:\.mp4|\.webm)', html)
for u in urls:
    print(u)
