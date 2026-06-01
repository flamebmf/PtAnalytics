# Copyright (c) 2026 PlurumTech.com
# SPDX-License-Identifier: GPL-3.0-only
#!/usr/bin/env python3
"""Check RTSP camera connection using OpenCV"""
import sys
import cv2
import time

def check_camera(url, test_seconds=5):
    print(f"Testing camera: {url}")
    
    cap = cv2.VideoCapture(url, cv2.CAP_FFMPEG)
    
    if not cap.isOpened():
        print("FAILED: Cannot open camera")
        return False
    
    print("SUCCESS: Camera opened")
    
    frames = 0
    start_time = time.time()
    
    while time.time() - start_time < test_seconds:
        ret, frame = cap.read()
        if not ret or frame is None:
            print(f"  Frame read failed (frame #{frames})")
            break
        
        h, w = frame.shape[:2]
        frames += 1
        print(f"  Frame #{frames}: {w}x{h} OK")
    
    cap.release()
    
    elapsed = time.time() - start_time
    print(f"\nDuration: {elapsed:.1f}s, Frames: {frames}")
    
    return frames > 0

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python check-camera.py <rtsp_url>")
        sys.exit(1)
    
    url = sys.argv[1]
    success = check_camera(url)
    sys.exit(0 if success else 1)
