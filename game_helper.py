import os
import time
import argparse
from datetime import datetime

import cv2
import numpy as np
from PIL import Image

try:
    import pyautogui
except Exception:
    pyautogui = None

try:
    import keyboard
except Exception:
    keyboard = None

# =============== 基本配置 ===================

SCREENSHOT_DIR = "Screenshots"
TEMPLATE_DIR = "digit_templates"

GRID_ROWS = 14
GRID_COLS = 8

# 你已经调好的棋盘区域比例
BOARD_LEFT_RATIO = 0.378
BOARD_RIGHT_RATIO = 0.622
BOARD_TOP_RATIO = 0.20
BOARD_BOTTOM_RATIO = 0.97

# 每个格子内部取一个“中心区域”来识别数字（避免边框干扰）
CELL_INNER_SCALE_W = 0.5
CELL_INNER_SCALE_H = 0.5

# Canny 边缘检测阈值（可以根据效果微调）
CANNY_LOW = 50
CANNY_HIGH = 150

# 模板匹配时的最小相似度，低于这个就认为识别失败/空格
MIN_SIMILARITY = 0.55

# 矩形高度 -> 最大宽度（格数）：1×9, 2×4, 3×3
HEIGHT_MAX_WIDTH = {1: 9, 2: 4, 3: 3}


# =============== 工具函数 ===================

def ensure_dir(path: str):
    os.makedirs(path, exist_ok=True)


def take_screenshot() -> str:
    """截一张全屏保存到 Screenshots 目录。"""
    if pyautogui is None:
        raise RuntimeError("当前环境没有安装 pyautogui，无法自动截屏。")
    ensure_dir(SCREENSHOT_DIR)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = os.path.join(SCREENSHOT_DIR, f"screenshot_{ts}.png")
    img = pyautogui.screenshot()
    img.save(filename)
    print(f"[INFO] 截图已保存: {filename}")
    return filename


def load_image(path: str) -> Image.Image:
    if not os.path.exists(path):
        raise FileNotFoundError(path)
    return Image.open(path)


def get_board_bbox(image: Image.Image):
    """根据比例算出棋盘区域的像素坐标。"""
    w, h = image.size
    left = int(w * BOARD_LEFT_RATIO)
    right = int(w * BOARD_RIGHT_RATIO)
    top = int(h * BOARD_TOP_RATIO)
    bottom = int(h * BOARD_BOTTOM_RATIO)
    return left, top, right, bottom


# =============== 灰度 + Canny 边缘模板匹配 ===================

def preprocess_edge_image(pil_img: Image.Image, size=(32, 32)) -> np.ndarray:
    """
    模板 / 单个格子统一的预处理流程：
      1. 转灰度
      2. CLAHE 自适应直方图均衡（提升对比度）
      3. Canny 边缘提取
      4. 缩放到固定大小
      5. 去均值 + L2 归一化（得到单位向量）
    """
    gray = cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2GRAY)
    # 提升对比度
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    gray = clahe.apply(gray)
    # 边缘提取
    edges = cv2.Canny(gray, CANNY_LOW, CANNY_HIGH)
    # 缩放到统一大小
    resized = cv2.resize(edges, size, interpolation=cv2.INTER_AREA)
    v = resized.astype("float32") / 255.0
    # 去均值 + 归一化
    mean = float(v.mean())
    v = v - mean
    norm = float(np.linalg.norm(v))
    if norm > 1e-6:
        v /= norm
    return v


def load_digit_templates():
    """
    从 TEMPLATE_DIR 中加载数字模板。
    支持文件名：
        1.png  2.png ... 9.png
    如果额外放了 0.png，会被当做“空格模板”使用。
    """
    templates: dict[int, np.ndarray] = {}

    if not os.path.isdir(TEMPLATE_DIR):
        print(f"[WARN] 模板目录 {TEMPLATE_DIR} 不存在，请先准备 1.png ~ 9.png。")
        return templates

    for name in os.listdir(TEMPLATE_DIR):
        if not name.lower().endswith(".png"):
            continue
        stem = os.path.splitext(name)[0]
        if not stem.isdigit():
            continue
        digit = int(stem)
        path = os.path.join(TEMPLATE_DIR, name)
        img = Image.open(path)
        templates[digit] = preprocess_edge_image(img)

    if not templates:
        print(f"[WARN] 在 {TEMPLATE_DIR} 中没找到任何数字模板 png 文件。")
    else:
        print(f"[INFO] 共加载到 {len(templates)} 个模板数字: {sorted(templates.keys())}")
    return templates


def classify_cell_from_templates(cell_img: Image.Image,
                                 templates: dict[int, np.ndarray],
                                 min_similarity: float = MIN_SIMILARITY):
    """
    使用“边缘特征 + 余弦相似度”的方式，判断 cell_img 是哪个数字。
    返回 (digit or None, best_score)。
    """
    if not templates:
        return None, 0.0

    v = preprocess_edge_image(cell_img)
    best_digit = None
    best_score = -1.0

    for d, temp_v in templates.items():
        score = float((v * temp_v).sum())  # 余弦相似度
        if score > best_score:
            best_score = score
            best_digit = d

    if best_score < min_similarity:
        # 相似度太低，当作空格（None）
        return None, best_score
    return best_digit, best_score


# =============== 提取棋盘数字矩阵 ===================

def extract_grid_with_templates(image: Image.Image,
                                templates: dict[int, np.ndarray],
                                debug: bool = False):
    """
    用模板匹配（灰度 + 边缘）识别整张棋盘的数字，返回 14×8 的矩阵。
    """
    if not templates:
        raise RuntimeError("没有数字模板，无法识别，请先在 digit_templates 中放 1.png~9.png")

    w, h = image.size
    left, top, right, bottom = get_board_bbox(image)
    board_w = right - left
    board_h = bottom - top
    cell_w = board_w / GRID_COLS
    cell_h = board_h / GRID_ROWS

    grid: list[list[int | None]] = []

    debug_img = None
    if debug:
        debug_img = cv2.cvtColor(np.array(image.copy()), cv2.COLOR_RGB2BGR)

    for r in range(GRID_ROWS):
        row_vals = []
        for c in range(GRID_COLS):
            cx = left + (c + 0.5) * cell_w
            cy = top + (r + 0.5) * cell_h
            bw = cell_w * CELL_INNER_SCALE_W
            bh = cell_h * CELL_INNER_SCALE_H
            l = int(cx - bw / 2)
            t = int(cy - bh / 2)
            rr = int(cx + bw / 2)
            bb = int(cy + bh / 2)

            cell_img = image.crop((l, t, rr, bb))
            digit, score = classify_cell_from_templates(cell_img, templates)
            row_vals.append(digit)

            if debug and debug_img is not None:
                cv2.rectangle(debug_img, (l, t), (rr, bb), (0, 0, 255), 1)
                label = "." if digit is None else str(digit)
                cv2.putText(debug_img, label, (l, t - 2),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 0), 1)

        grid.append(row_vals)

    print("[INFO] 识别到的数字矩阵：")
    for r in grid:
        print(" ".join(str(x) if x is not None else "." for x in r))

    if debug and debug_img is not None:
        dbg_path = "debug_grid_v3.png"
        cv2.imwrite(dbg_path, debug_img)
        print(f"[DEBUG] 调试图已保存: {dbg_path}")

    return grid


# =============== 游戏规则 & 贪心算法 ===================

def build_prefix_sum(grid):
    rows = len(grid)
    cols = len(grid[0])
    ps = [[0] * (cols + 1) for _ in range(rows + 1)]
    for i in range(rows):
        for j in range(cols):
            v = grid[i][j]
            val = 0 if v is None else v
            ps[i + 1][j + 1] = (
                ps[i][j + 1] + ps[i + 1][j] - ps[i][j] + val
            )
    return ps


def rect_sum(ps, r1, c1, r2, c2):
    return (ps[r2 + 1][c2 + 1]
            - ps[r1][c2 + 1]
            - ps[r2 + 1][c1]
            + ps[r1][c1])


def find_valid_rectangles(grid):
    """
    找出所有符合规则的矩形：
      - 高度只能是 1、2、3
      - 对应最大宽度：1→9，2→4，3→3
      - 矩形内数字和为 10
      - None 视为 0，可以被框进去
    返回列表：(r1,c1,r2,c2,area)，按面积从小到大排序。
    """
    rows = len(grid)
    cols = len(grid[0])
    ps = build_prefix_sum(grid)

    moves = []
    for top in range(rows):
        for left in range(cols):
            for h in (1, 2, 3):
                bottom = top + h - 1
                if bottom >= rows:
                    break
                max_w = min(HEIGHT_MAX_WIDTH[h], cols - left)
                for w in range(1, max_w + 1):
                    right = left + w - 1
                    area = h * w
                    if area < 2:
                        continue
                    s = rect_sum(ps, top, left, bottom, right)
                    if s == 10:
                        moves.append((top, left, bottom, right, area))

    moves.sort(key=lambda x: (x[4], x[0], x[1]))
    return moves


def greedy_sequence(grid, max_steps=1000):
    """
    贪心多步模拟：每一步选面积最小、最靠上的矩形；
    选中格子数字全部置为 0（表示被消除）。
    """
    work = [row[:] for row in grid]
    steps = []

    for _ in range(max_steps):
        moves = find_valid_rectangles(work)
        if not moves:
            break
        r1, c1, r2, c2, area = moves[0]
        steps.append(moves[0])
        for i in range(r1, r2 + 1):
            for j in range(c1, c2 + 1):
                v = work[i][j]
                if v is not None and v != 0:
                    work[i][j] = 0

    return steps, work


def pretty_print_single_best(moves):
    if not moves:
        print("[RESULT] 当前没有可消除的矩形（和为 10）")
        return None
    r1, c1, r2, c2, area = moves[0]
    print(f"[RESULT] 推荐本步选择：行 {r1+1}-{r2+1}，列 {c1+1}-{c2+1} （格子数: {area}）")
    if len(moves) > 1:
        print("        其他可选方案数量：", len(moves) - 1)
    return moves[0]


def pretty_print_sequence(steps):
    if not steps:
        print("[RESULT] 贪心模拟：没有任何可选矩形。")
        return
    print(f"[RESULT] 贪心模拟共找到 {len(steps)} 步：")
    for idx, (r1, c1, r2, c2, area) in enumerate(steps, 1):
        print(f"  第{idx:2d}步：行 {r1+1}-{r2+1}，列 {c1+1}-{c2+1} （格子数: {area}）")


# =============== 一次分析 & 快捷键模式 ===================

def analyze_once(templates,
                 img_path: str | None = None,
                 debug: bool = False,
                 show_sequence: bool = False):
    """完整执行一次：截/读图 → 识别矩阵 → 输出一步 / 整局策略。"""
    if img_path is None:
        img_path = take_screenshot()
    else:
        print(f"[INFO] 使用指定图片: {img_path}")

    image = load_image(img_path)
    grid = extract_grid_with_templates(image, templates, debug=debug)

    if show_sequence:
        steps, final_grid = greedy_sequence(grid)
        pretty_print_sequence(steps)
        print("\n[INFO] 贪心模拟结束后的棋盘（0 或 . 表示空格）：")
        for r in final_grid:
            print(" ".join(str(x) if x not in (None, 0) else "." for x in r))
    else:
        moves = find_valid_rectangles(grid)
        pretty_print_single_best(moves)

    return grid


def hotkey_loop(templates, hotkey="ctrl+shift+q", debug=False):
    """监听快捷键，每按一次就分析当前一步。"""
    if keyboard is None:
        raise RuntimeError("当前环境没有安装 keyboard 库，无法进入快捷键模式。")

    print("=" * 60)
    print(f"[INFO] 已进入快捷键模式，按 {hotkey} 进行【当前一步】的推荐。")
    print("       每次都会重新截屏并分析当前棋盘。")
    print("       按 Ctrl+C 或关闭窗口退出。")
    print("=" * 60)

    def _on_hotkey():
        print("\n[HOTKEY] 捕捉到按键，开始分析当前屏幕...")
        analyze_once(templates, img_path=None, debug=debug, show_sequence=False)
        print("[HOTKEY] 本次分析结束，可继续游戏或再次按快捷键。")

    keyboard.add_hotkey(hotkey, _on_hotkey)

    try:
        while True:
            time.sleep(0.5)
    except KeyboardInterrupt:
        print("\n[INFO] 用户中断，退出。")


# =============== 命令行入口 ===================

def main():
    parser = argparse.ArgumentParser(
        description="小游戏辅助：灰度+边缘模板匹配 + 和为10贪心"
    )
    parser.add_argument("--image", "-i", help="使用已有图片分析（不截屏）")
    parser.add_argument("--once", action="store_true",
                        help="只执行一次分析并退出")
    parser.add_argument("--debug", action="store_true",
                        help="调试模式：输出 debug_grid_v3.png")
    parser.add_argument("--sequence", action="store_true",
                        help="一次性计算整局贪心消除序列（仅分析用）")
    parser.add_argument("--hotkey", default="ctrl+shift+q",
                        help="快捷键（默认 ctrl+shift+q）")

    args = parser.parse_args()

    templates = load_digit_templates()
    if not templates:
        print("\n[ERROR] 没有加载到任何数字模板，无法继续。")
        print(f"       请在 {TEMPLATE_DIR} 目录下放置 1.png~9.png（可选 0.png 代表空格）。")
        return

    if args.once:
        analyze_once(templates, img_path=args.image,
                     debug=args.debug, show_sequence=args.sequence)
    else:
        if args.image:
            analyze_once(templates, img_path=args.image,
                         debug=args.debug, show_sequence=args.sequence)
            print("\n[INFO] 之后会使用实时截屏进行分析。")
        hotkey_loop(templates, hotkey=args.hotkey, debug=args.debug)


if __name__ == "__main__":
    main()
