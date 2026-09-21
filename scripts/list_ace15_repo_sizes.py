# -*- coding: utf-8 -*-
import json
import urllib.request

for repo in ["ACE-Step/Ace-Step1.5", "ACE-Step/acestep-v15-xl-turbo"]:
    req = urllib.request.Request(
        f"https://hf-mirror.com/api/models/{repo}?blobs=true",
        headers={"User-Agent": "Mozilla/5.0"},
    )
    d = json.load(urllib.request.urlopen(req, timeout=60))
    print("===", repo)
    for s in d.get("siblings", []):
        sz = s.get("size") or 0
        print(f"  {s['rfilename']}  {sz:,}")
