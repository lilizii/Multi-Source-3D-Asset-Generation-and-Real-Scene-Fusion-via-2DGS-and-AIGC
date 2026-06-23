"""
图片去背景工具 - 提供两种方式:
1. 使用 rembg (AI自动去背景)
2. 使用 OpenCV + 手工处理

输出: 4通道 RGBA PNG 图片 (用于 Magic123 / Zero123 输入)
用法: python remove_bg.py --input photo.jpg --output photo_rgba.png
"""
import argparse
import cv2
import numpy as np
import os
import sys


def remove_bg_rembg(input_path: str, output_path: str, model: str = "u2net"):
    """使用 rembg 自动去背景, 支持多种模型
    model: u2net, isnet-general-use, silueta, u2net_cloth
    """
    try:
        from rembg import remove, new_session
        with open(input_path, "rb") as f:
            data = f.read()
        session = new_session(model)
        result = remove(data, session=session)
        with open(output_path, "wb") as f:
            f.write(result)
        print(f"Background removed (rembg, model={model}): {output_path}")
        return True
    except ImportError:
        return False


def remove_bg_manual(input_path: str, output_path: str,
                     lower_hsv=(0, 0, 0), upper_hsv=(180, 255, 200)):
    """
    手工去背景 - 使用 HSV 颜色阈值
    适用于简单背景 (如白底/纯色背景)
    """
    img = cv2.imread(input_path)
    if img is None:
        raise RuntimeError(f"Cannot read image: {input_path}")

    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(hsv, np.array(lower_hsv), np.array(upper_hsv))
    mask = cv2.bitwise_not(mask)

    # 形态学操作
    kernel = np.ones((5, 5), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)

    # 创建 RGBA 图片
    b, g, r = cv2.split(img)
    alpha = mask
    rgba = cv2.merge([b, g, r, alpha])
    cv2.imwrite(output_path, rgba)
    print(f"Background removed (manual HSV): {output_path}")
    return True


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--method", default="auto",
                        choices=["auto", "rembg", "manual"],
                        help="auto=先尝试rembg再fallback到manual")
    parser.add_argument("--model", default="u2net",
                        choices=["u2net", "isnet-general-use", "silueta", "u2net_cloth"],
                        help="rembg 模型: u2net(默认), isnet(更准), silueta(边缘), u2net_cloth(布料)")
    args = parser.parse_args()

    if not os.path.exists(args.input):
        print(f"Error: input file not found: {args.input}")
        sys.exit(1)

    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)

    if args.method in ("auto", "rembg"):
        if remove_bg_rembg(args.input, args.output, model=args.model):
            return

    if args.method in ("auto", "manual"):
        remove_bg_manual(args.input, args.output)
        return

    print("Error: no valid method succeeded")
    sys.exit(1)


if __name__ == "__main__":
    main()
