import os
import torch
import cv2
import numpy as np
from PIL import Image

# Check the control video file
control_video_path = "asset/src_video_depth_49_896x512_8fps.mp4"

if os.path.exists(control_video_path):
    cap = cv2.VideoCapture(control_video_path)
    
    # Get video properties
    fps = cap.get(cv2.CAP_PROP_FPS)
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    
    print(f"Control Video Properties:")
    print(f"  FPS: {fps}")
    print(f"  Total Frames: {frame_count}")
    print(f"  Resolution: {width}x{height} (W x H)")
    
    cap.release()
    
    # Now simulate what get_video_to_video_latent does
    sample_size = [512, 896]  # height, width
    video_length = 49
    fps_input = 8
    
    print(f"\nProcessing Parameters:")
    print(f"  Sample Size: {sample_size} (H x W)")
    print(f"  Video Length: {video_length}")
    print(f"  Desired FPS: {fps_input}")
    
    # Simulate frame reading
    cap = cv2.VideoCapture(control_video_path)
    original_fps = cap.get(cv2.CAP_PROP_FPS)
    frame_skip = 1 if fps_input is None else max(1, int(original_fps // fps_input))
    
    print(f"  Original FPS: {original_fps}")
    print(f"  Frame Skip: {frame_skip}")
    
    input_video = []
    frame_count = 0
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        if frame_count % frame_skip == 0:
            frame = cv2.resize(frame, (sample_size[1], sample_size[0]))
            input_video.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        
        frame_count += 1
    
    cap.release()
    
    # Convert to tensor (same as get_video_to_video_latent)
    input_video = torch.from_numpy(np.array(input_video))[:video_length]
    print(f"\nAfter reading and truncating:")
    print(f"  Input video shape (F, H, W, C): {input_video.shape}")
    
    # Permute to (C, F, H, W) and add batch dimension
    input_video = input_video.permute([3, 0, 1, 2]).unsqueeze(0) / 255
    print(f"  After permute and batch add (B, C, F, H, W): {input_video.shape}")
    
    # Now check VAE encoding expectations
    print(f"\nVAE Encoding Expectations:")
    print(f"  For video_length={video_length} frames with temporal_compression_ratio=4:")
    
    # From predict script
    video_length_adjusted = int((video_length - 1) // 4 * 4) + 1 if video_length != 1 else 1
    latent_frames = (video_length_adjusted - 1) // 4 + 1
    
    print(f"    Adjusted video_length: {video_length_adjusted}")
    print(f"    Expected latent_frames: {latent_frames}")
    
else:
    print(f"Control video file not found: {control_video_path}")
    print(f"Current directory: {os.getcwd()}")
    print(f"Files in asset/:")
    if os.path.exists("asset"):
        for f in os.listdir("asset"):
            print(f"  {f}")
