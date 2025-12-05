import cv2
import numpy as np
import pytesseract
from mss import mss
import time

# ----------------- 配置区 -----------------
# 棋盘行列数
BOARD_ROWS = 14
BOARD_COLS = 8

# 目标和
TARGET_SUM = 10

# Tesseract 安装路径（请根据你本机修改）
# Windows 示例：
# pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"

# 如果已经在系统 PATH 中，可以注释掉上面这一行

# 画框颜色和线宽
RECT_COLOR = (0, 0, 255)  # BGR 红色
RECT_THICKNESS = 2

# 显示全部矩形还是只显示一个推荐解
SHOW_ALL_RECTANGLES = True    # True: 显示所有可消除矩形
# -------------------------------------------------


def grab_screen_rect(monitor_dict):
    """使用 mss 截取屏幕的一块区域，返回 BGR 图像"""
    with mss() as sct:
        img = np.array(sct.grab(monitor_dict))[:, :, :3]
    # mss 得到的是 BGRA，这里去掉 A
    img = cv2.cvtColor(img, cv2.COLOR_BGRA2BGR)
    return img


def select_board_region_once():
    """
    第一次运行时：截一张全屏图，让你手动框选棋盘区域。
    返回 (x, y, w, h)
    """
    with mss() as sct:
        monitor = sct.monitors[1]  # 主显示器
        screenshot = np.array(sct.grab(monitor))[:, :, :3]
        screenshot = cv2.cvtColor(screenshot, cv2.COLOR_BGRA2BGR)

    cv2.namedWindow("Select board (拖动选择棋盘, 回车确认, ESC取消)", cv2.WINDOW_NORMAL)
    roi = cv2.selectROI("Select board (拖动选择棋盘, 回车确认, ESC取消)",
                        screenshot, showCrosshair=True, fromCenter=False)
    cv2.destroyWindow("Select board (拖动选择棋盘, 回车确认, ESC取消)")

    x, y, w, h = roi
    if w == 0 or h == 0:
        raise RuntimeError("没有选择有效区域，请重新运行程序")

    print(f"[INFO] 已选择棋盘区域: x={x}, y={y}, w={w}, h={h}")
    return x, y, w, h

def preprocess_digit(cell_img):
    """对单个格子图像做预处理，返回适合 OCR 的二值图"""
    gray = cv2.cvtColor(cell_img, cv2.COLOR_BGR2GRAY)

    # 1. 边缘裁剪少一点，避免把数字切掉
    h, w = gray.shape
    margin_h = int(h * 0.05)   # 原来是 0.15，这里改成 0.05
    margin_w = int(w * 0.05)
    gray = gray[margin_h:h - margin_h, margin_w:w - margin_w]

    # 2. 放大，提高分辨率
    gray = cv2.resize(gray, None, fx=4, fy=4, interpolation=cv2.INTER_LINEAR)

    # 3. 均衡化，拉开对比度
    gray = cv2.equalizeHist(gray)

    # 4. OTSU 二值化
    _, th = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    # 5. 确保“字是黑色，背景是白色”
    num_black = np.sum(th == 0)
    num_white = np.sum(th == 255)
    if num_black < num_white:
        th = 255 - th

    # 6. 做一点闭运算，把笔画连起来
    kernel = np.ones((3, 3), np.uint8)
    th = cv2.morphologyEx(th, cv2.MORPH_CLOSE, kernel)

    return th


def ocr_cell(cell_img):
    """识别单个格子的数字：1~9，识别失败返回 -1"""
    img = preprocess_digit(cell_img)

    # 尝试两套配置
    configs = [
        r'--psm 10 --oem 1 -c tessedit_char_whitelist=123456789 -c classify_bln_numeric_mode=1',
        r'--psm 13 --oem 1 -c tessedit_char_whitelist=123456789 -c classify_bln_numeric_mode=1',
    ]

    for config in configs:
        text = pytesseract.image_to_string(img, config=config)
        digits = [ch for ch in text if ch.isdigit()]
        if digits:
            return int(digits[0])

    # 实在识别不了，就返回 -1（表示“未知”）
    return -1



def ocr_board(board_img):
    """
    对整块棋盘图像做 14x8 切分，返回:
    - nums: 数字矩阵 (int)，未知位置先填 0
    - unknown_mask: 同样大小的 bool 矩阵，True 表示该格子识别失败
    """
    h, w, _ = board_img.shape
    cell_h = h / BOARD_ROWS
    cell_w = w / BOARD_COLS

    nums = np.zeros((BOARD_ROWS, BOARD_COLS), dtype=int)
    unknown_mask = np.zeros((BOARD_ROWS, BOARD_COLS), dtype=bool)

    for r in range(BOARD_ROWS):
        for c in range(BOARD_COLS):
            y1 = int(r * cell_h)
            y2 = int((r + 1) * cell_h)
            x1 = int(c * cell_w)
            x2 = int((c + 1) * cell_w)

            cell = board_img[y1:y2, x1:x2]
            digit = ocr_cell(cell)

            if digit == -1:
                unknown_mask[r, c] = True
                nums[r, c] = 0
            else:
                nums[r, c] = digit

    return nums, unknown_mask


def find_rectangles_sum_k(board, unknown_mask, target=10):
    """
    枚举所有矩形(行列均连续) 使得内部元素和为 target。
    如果矩形中包含未知格子(unknown_mask=True)，则跳过。
    返回列表 [(top, left, bottom, right), ...]
    """
    rows, cols = board.shape
    res = []

    for top in range(rows):
        col_sums = np.zeros(cols, dtype=int)
        col_unknown = np.zeros(cols, dtype=int)
        for bottom in range(top, rows):
            col_sums += board[bottom]
            col_unknown += unknown_mask[bottom].astype(int)

            for left in range(cols):
                s = 0
                u = 0
                for right in range(left, cols):
                    s += col_sums[right]
                    u += col_unknown[right]

                    # 有未知格子，直接跳过这个矩形
                    if u > 0:
                        continue

                    area = (bottom - top + 1) * (right - left + 1)
                    if s == target and area >= 2:
                        res.append((top, left, bottom, right))

                    if s >= target:
                        break
    return res


def choose_best_rectangle(rectangles):
    """从所有矩形中挑一个“推荐解”（这里选择面积最大的）"""
    if not rectangles:
        return None
    # 面积最大；若面积相同，优先靠上的、靠左的
    best = max(rectangles,
               key=lambda r: ((r[2] - r[0] + 1) * (r[3] - r[1] + 1),
                              -r[0], -r[1]))
    return best


def draw_rectangles_on_board(board_img, rectangles):
    """在棋盘图像上画矩形框，返回新图像"""
    h, w, _ = board_img.shape
    cell_h = h / BOARD_ROWS
    cell_w = w / BOARD_COLS

    vis = board_img.copy()

    for (top, left, bottom, right) in rectangles:
        x1 = int(left * cell_w)
        y1 = int(top * cell_h)
        x2 = int((right + 1) * cell_w)
        y2 = int((bottom + 1) * cell_h)

        cv2.rectangle(vis, (x1, y1), (x2, y2), RECT_COLOR, RECT_THICKNESS)

    return vis


def main_live():
    # 1. 选择棋盘区域
    x, y, w, h = select_board_region_once()
    monitor = {"left": x, "top": y, "width": w, "height": h}

    print("[INFO] 开始实时识别，按 ESC 退出")

    with mss() as sct:
        while True:
            frame = np.array(sct.grab(monitor))[:, :, :3]
            frame = cv2.cvtColor(frame, cv2.COLOR_BGRA2BGR)

            # 2. OCR 识别数字矩阵
            nums, unknown_mask = ocr_board(frame)
            rectangles = find_rectangles_sum_k(nums, unknown_mask, TARGET_SUM)

            #（可选）打印一次看看
            print(nums)

            # 3. 找所有矩形和为10的组合
            rectangles = find_rectangles_sum_k(nums, TARGET_SUM)

            if not rectangles:
                msg = "当前无可消除组合"
            else:
                msg = f"可消除组合数量: {len(rectangles)}"

            # 4. 画框
            if SHOW_ALL_RECTANGLES:
                draw_rects = rectangles
            else:
                best = choose_best_rectangle(rectangles)
                draw_rects = [best] if best else []

            vis = draw_rectangles_on_board(frame, draw_rects)

            # 在左上角写点文字
            cv2.putText(vis, msg, (10, 20),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

            cv2.imshow("Ten Helper (ESC 退出)", vis)

            key = cv2.waitKey(1) & 0xFF
            if key == 27:  # ESC
                break

    cv2.destroyAllWindows()


def main_single_image(image_path):
    """
    如果你只想用你给的那张截图测试，可以调用这个函数：
    例如: main_single_image("1.png")
    """
    img = cv2.imread(image_path)
    if img is None:
        print(f"[ERROR] 无法打开图片: {image_path}")
        return

    # 让你在这张图片上选棋盘区域
    cv2.namedWindow("Select board", cv2.WINDOW_NORMAL)
    roi = cv2.selectROI("Select board", img, showCrosshair=True, fromCenter=False)
    cv2.destroyWindow("Select board")

    x, y, w, h = roi
    board_img = img[y:y + h, x:x + w]

    nums, unknown_mask = ocr_board(board_img)
    rectangles = find_rectangles_sum_k(nums, unknown_mask, TARGET_SUM)

    print("识别的数字矩阵：")
    print(nums)

    rectangles = find_rectangles_sum_k(nums, TARGET_SUM)
    print(f"找到 {len(rectangles)} 个可消除矩形")

    if SHOW_ALL_RECTANGLES:
        draw_rects = rectangles
    else:
        best = choose_best_rectangle(rectangles)
        draw_rects = [best] if best else []

    vis = draw_rectangles_on_board(board_img, draw_rects)

    cv2.imshow("Result", vis)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


if __name__ == "__main__":
    # 默认跑实时模式，如果你想先在单张图上测试，
    # 可以把 main_live() 注释掉，改成 main_single_image("1.png")
    # main_live()
    main_single_image("1.png")
