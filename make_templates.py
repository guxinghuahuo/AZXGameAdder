# make_templates.py
import os
import cv2
import numpy as np

BOARD_ROWS = 14
BOARD_COLS = 8
IMAGE_PATH = "1.png"   # 你的截图文件名

TEMPLATES_DIR = "templates"

os.makedirs(TEMPLATES_DIR, exist_ok=True)


def prepare_digit_image(cell_img):
    """和识别时一致的预处理：转灰度+裁掉格子边框一点"""
    if len(cell_img.shape) == 3:
        gray = cv2.cvtColor(cell_img, cv2.COLOR_BGR2GRAY)
    else:
        gray = cell_img.copy()

    h, w = gray.shape
    margin_h = int(h * 0.15)
    margin_w = int(w * 0.15)
    gray = gray[margin_h:h - margin_h, margin_w:w - margin_w]

    return gray


def main():
    img = cv2.imread(IMAGE_PATH)
    if img is None:
        print(f"[ERROR] 无法读取图片: {IMAGE_PATH}")
        return

    cv2.namedWindow("Select board", cv2.WINDOW_NORMAL)
    roi = cv2.selectROI("Select board", img, showCrosshair=True, fromCenter=False)
    cv2.destroyWindow("Select board")

    x, y, w, h = roi
    if w == 0 or h == 0:
        print("[ERROR] ROI 无效，请重新运行")
        return

    board_img = img[y:y + h, x:x + w]

    bh, bw, _ = board_img.shape
    cell_h = bh / BOARD_ROWS
    cell_w = bw / BOARD_COLS

    # 每个数字的计数
    save_count = {str(d): 0 for d in range(1, 10)}

    print("开始标注模板：")
    print(" - 按 1~9：保存为对应数字模板")
    print(" - 按 0 或 空格：跳过该格子")
    print(" - 按 ESC：提前结束")

    for r in range(BOARD_ROWS):
        for c in range(BOARD_COLS):
            y1 = int(r * cell_h)
            y2 = int((r + 1) * cell_h)
            x1 = int(c * cell_w)
            x2 = int((c + 1) * cell_w)

            cell = board_img[y1:y2, x1:x2]
            show = cv2.resize(cell, None, fx=4, fy=4, interpolation=cv2.INTER_NEAREST)

            while True:
                cv2.imshow("Label digit (1-9), 0/space 跳过, ESC 结束", show)
                key = cv2.waitKey(0)

                # ESC
                if key == 27:
                    cv2.destroyAllWindows()
                    print("\n标注结束。")
                    return

                # 空格或 0：跳过
                if key in (32, ord('0')):
                    break

                # 1~9：保存模板
                if ord('1') <= key <= ord('9'):
                    d = chr(key)
                    idx = save_count[d]
                    save_count[d] += 1

                    proc = prepare_digit_image(cell)
                    filename = f"{d}_{idx}.png"
                    filepath = os.path.join(TEMPLATES_DIR, filename)
                    cv2.imwrite(filepath, proc)
                    print(f"保存模板: {filepath}")
                    break

                print("请按 1~9 标记数字，0/空格跳过，ESC 结束。")

    cv2.destroyAllWindows()
    print("\n全部格子标注完成。当前每个数字模板数量：")
    for d in range(1, 10):
        print(f"{d}: {save_count[str(d)]} 张")


if __name__ == "__main__":
    main()
