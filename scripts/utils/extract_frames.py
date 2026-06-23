"""
从视频中提取帧, 用于多视角重建
用法: python extract_frames.py --video input.mp4 --output frames/ --fps 3
"""
import argparse
import cv2
import os


def extract_frames(video_path: str, output_dir: str, fps: int = 3):
    os.makedirs(output_dir, exist_ok=True)
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video: {video_path}")

    video_fps = cap.get(cv2.CAP_PROP_FPS)
    interval = max(1, int(video_fps / fps))
    frame_count = 0
    saved_count = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        if frame_count % interval == 0:
            out_path = os.path.join(output_dir, f"{saved_count:05d}.jpg")
            cv2.imwrite(out_path, frame)
            saved_count += 1
        frame_count += 1

    cap.release()
    print(f"Extracted {saved_count} frames from {frame_count} total, "
          f"video FPS={video_fps:.1f}, target FPS={fps}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--video", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--fps", type=int, default=3)
    args = parser.parse_args()
    extract_frames(args.video, args.output, args.fps)


if __name__ == "__main__":
    main()
