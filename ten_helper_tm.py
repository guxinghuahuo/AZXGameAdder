# ten_helper_tm.py
import cv2
import numpy as np
import os
import glob
from mss import mss

BOARD_ROWS = 14
BOARD_COLS = 8
TARGET_SUM = 10

TEMPLATES_DIR = "templates"

RECT_COLOR = (0, 0, 255)  # BGR 红色
RECT_THICKNESS = 2
SHOW_ALL_RECTANGLES = True


def prepare_digit_image(cell_img):
    """识别时的预处理：转灰度 + 裁掉边框一点"""
    if len(cell_img.shape) == 3:
        gray = cv2.cvtColor(cell_img, cv2.COLOR_BGR2GRAY)
    else:
        gray = cell_img.copy()

    h, w = gray.shape
    margin_h = int(h * 0.15)
    margin_w = int(w * 0.15)
    gray = gray[margin_h:h - margin_h, margin_w:w - margin_w]

    return gray


class DigitTemplates:
    def __init__(self, templates_dir=TEMPLATES_DIR):
        self.templates = []  # list of (digit, img)
        self._load(templates_dir)

    def _load(self, templates_dir):
        paths = glob.glob(os.path.join(templates_dir, "*.png"))
        if not paths:
            raise RuntimeError(
                f"在 {templates_dir} 目录下没有找到任何模板，请先运行 make_templates.py 生成模板。"
            )

        for path in paths:
            name = os.path.basename(path)  # 如 "5_0.png"
            try:
                d = int(name.split("_")[0])
            except Exception:
                print(f"[WARN] 模板文件名无法解析数字，已跳过: {name}")
                continue

            img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
            if img is None:
                print(f"[WARN] 无法读取模板文件: {path}")
                continue

            self.templates.append((d, img))

        if not self.templates:
            raise RuntimeError("模板加载失败：没有可用模板。")

        print(f"[INFO] 已加载 {len(self.templates)} 个模板。")

    def match_digit(self, cell_img):
        """
        对单个格子进行模板匹配。
        返回 (digit, best_score)，digit=-1 表示匹配度太低，认为未知。
        """
        cell = prepare_digit_image(cell_img)
        h, w = cell.shape

        best_digit = -1
        best_score = -1.0

        for d, tpl in self.templates:
            th, tw = tpl.shape
            if h < th or w < tw:
                continue

            res = cv2.matchTemplate(cell, tpl, cv2.TM_CCOEFF_NORMED)
            score = res.max()

            if score > best_score:
                best_score = score
                best_digit = d

        # 阈值可以按效果调整，一般 0.5~0.7
        if best_score < 0.5:
            return -1, best_score

        return best_digit, best_score


def select_board_region_once(image):
    """让用户在给定图像上选中棋盘区域，返回 (x, y, w, h)"""
    cv2.namedWindow("Select board", cv2.WINDOW_NORMAL)
    roi = cv2.selectROI("Select board", image, showCrosshair=True, fromCenter=False)
    cv2.destroyWindow("Select board")

    x, y, w, h = roi
    if w == 0 or h == 0:
        raise RuntimeError("没有选择有效区域，请重新运行程序")
    print(f"[INFO] 已选择棋盘区域: x={x}, y={y}, w={w}, h={h}")
    return x, y, w, h


def ocr_board(board_img, digit_templates: DigitTemplates):
    """
    对棋盘区域做 14x8 切分并识别数字。
    返回:
    - nums: int 矩阵 (BOARD_ROWS, BOARD_COLS)，未知格子填 0
    - unknown_mask: bool 矩阵，同样大小，True 表示该格子未知
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
            digit, score = digit_templates.match_digit(cell)

            if digit == -1:
                unknown_mask[r, c] = True
                nums[r, c] = 0
            else:
                nums[r, c] = digit

    return nums, unknown_mask


def find_rectangles_sum_k(board, unknown_mask, target=10):
    """
    找到所有矩形(行列连续) 使得内部数字之和为 target，
    且矩形内不包含 unknown_mask 为 True 的格子。
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

                    # 含有未知格子，跳过
                    if u > 0:
                        continue

                    area = (bottom - top + 1) * (right - left + 1)
                    if s == target and area >= 2:
                        res.append((top, left, bottom, right))

                    if s > target:
                        break
    return res


def choose_best_rectangle(rectangles):
    """从所有矩形中选一个“推荐解”，这里选面积最大的。"""
    if not rectangles:
        return None
    return max(
        rectangles,
        key=lambda r: ((r[2] - r[0] + 1) * (r[3] - r[1] + 1), -r[0], -r[1]),
    )


def draw_rectangles_on_board(board_img, rectangles):
    """在棋盘图像上画出矩形框"""
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


def main_single_image(image_path="1.png"):
    """
    在单张截图上测试：
    1. 读取图片
    2. 让你选棋盘 ROI
    3. 模板匹配识别矩阵
    4. 找出和为10的矩形并画框
    """
    img = cv2.imread(image_path)
    if img is None:
        print(f"[ERROR] 无法读取图片: {image_path}")
        return

    digit_templates = DigitTemplates(TEMPLATES_DIR)

    x, y, w, h = select_board_region_once(img)
    board_img = img[y:y + h, x:x + w]

    nums, unknown_mask = ocr_board(board_img, digit_templates)

    print("识别的数字矩阵：")
    print(nums)

    rectangles = find_rectangles_sum_k(nums, unknown_mask, TARGET_SUM)
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


def main_live():
    """
    实时模式：
    1. 截一张当前屏幕，选棋盘 ROI
    2. 使用 mss 持续截取该区域
    3. 识别矩阵 + 找所有和为10的矩形并画框，实时显示
    """
    digit_templates = DigitTemplates(TEMPLATES_DIR)

    with mss() as sct:
        monitor = sct.monitors[1]
        screenshot = np.array(sct.grab(monitor))[:, :, :3]
        screenshot = cv2.cvtColor(screenshot, cv2.COLOR_BGRA2BGR)

    x, y, w, h = select_board_region_once(screenshot)
    monitor_rect = {"left": x, "top": y, "width": w, "height": h}

    print("[INFO] 开始实时识别，按 ESC 退出")

    with mss() as sct:
        while True:
            frame = np.array(sct.grab(monitor_rect))[:, :, :3]
            frame = cv2.cvtColor(frame, cv2.COLOR_BGRA2BGR)

            nums, unknown_mask = ocr_board(frame, digit_templates)
            rectangles = find_rectangles_sum_k(nums, unknown_mask, TARGET_SUM)

            if not rectangles:
                msg = "当前无可消除组合"
            else:
                msg = f"可消除组合数量: {len(rectangles)}"

            if SHOW_ALL_RECTANGLES:
                draw_rects = rectangles
            else:
                best = choose_best_rectangle(rectangles)
                draw_rects = [best] if best else []

            vis = draw_rectangles_on_board(frame, draw_rects)
            cv2.putText(vis, msg, (10, 20),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

            cv2.imshow("Ten Helper (ESC 退出)", vis)
            key = cv2.waitKey(1) & 0xFF
            if key == 27:
                break

    cv2.destroyAllWindows()


if __name__ == "__main__":
    # 先在单张截图上测试：
    main_single_image("1.png")
    # 实时模式请改成：
    # main_live()
