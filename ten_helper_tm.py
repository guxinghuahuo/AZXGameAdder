# ten_helper_tm.py
import cv2
import numpy as np
import os
import glob
import threading
import time
import keyboard
from mss import mss

BOARD_ROWS = 16
BOARD_COLS = 10
TARGET_SUM = 10

TEMPLATES_DIR = "templates"

RECT_COLOR = (0, 0, 255)  # BGR 红色
RECT_THICKNESS = 2
SHOW_ALL_RECTANGLES = True

# 全局变量，控制实时识别的运行状态
live_mode_active = False
live_mode_thread = None
# 【新增】控制刷新的全局标志位
refresh_requested = False

def trigger_global_refresh():
    """全局热键的回调函数，只负责修改标志位"""
    global refresh_requested
    refresh_requested = True
    print("[INFO] 收到全局刷新请求 (Ctrl+Shift+Z)")

def prepare_digit_image(cell_img):
    """识别时的预处理：转灰度 + 裁掉边框一点"""
    if len(cell_img.shape) == 3:
        gray = cv2.cvtColor(cell_img, cv2.COLOR_BGR2GRAY)
    else:
        gray = cell_img.copy()
    # h, w = gray.shape
    # margin_h = int(h * 0.15)
    # margin_w = int(w * 0.15)
    # gray = gray[margin_h:h - margin_h, margin_w:w - margin_w]

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


def is_empty_cell(cell_img, threshold=20):
    """
    判断格子是否为空白（背景）。
    原理：空白格子颜色单一，标准差(std)很小；有数字的格子标准差很大。
    """
    if len(cell_img.shape) == 3:
        gray = cv2.cvtColor(cell_img, cv2.COLOR_BGR2GRAY)
    else:
        gray = cell_img
    
    # 计算标准差
    score = np.std(gray)
    
    # 如果标准差小于阈值，认为是空白格
    return score < threshold


def ocr_board(board_img, digit_templates: DigitTemplates):
    """
    对棋盘区域做 14x8 切分并识别数字。
    修改后：支持识别空白格为 0。
    """
    h, w, _ = board_img.shape
    cell_h = h / BOARD_ROWS
    cell_w = w / BOARD_COLS
    # print(f"[INFO] 棋盘切分: 每个格子大小 {cell_h:.2f}x{cell_w:.2f}")

    nums = np.zeros((BOARD_ROWS, BOARD_COLS), dtype=int)
    unknown_mask = np.zeros((BOARD_ROWS, BOARD_COLS), dtype=bool)

    for r in range(BOARD_ROWS):
        for c in range(BOARD_COLS):
            y1 = int(r * cell_h)
            y2 = int((r + 1) * cell_h)
            x1 = int(c * cell_w)
            x2 = int((c + 1) * cell_w)

            cell = board_img[y1:y2, x1:x2]
            
            # 1. 尝试匹配数字
            digit, score = digit_templates.match_digit(cell)

            if digit != -1:
                # 匹配到了数字
                nums[r, c] = digit
            else:
                # 2. 没匹配到数字，检查是不是空白格
                if is_empty_cell(cell):
                    nums[r, c] = 0  # 空白格视为 0
                    # unknown_mask 保持 False，表示这是一个有效的“0”
                else:
                    # 既不是数字也不是空白，才是真正的未知
                    unknown_mask[r, c] = True
                    nums[r, c] = 0

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
    """
    从所有矩形中选一个“推荐解”。
    策略修改：优先选择【面积最小】的矩形。
    如果面积相同，则优先选择【最靠上】、然后【最靠左】的。
    """
    if not rectangles:
        return None
    
    # 计算面积的辅助函数
    def get_area(r):
        return (r[2] - r[0] + 1) * (r[3] - r[1] + 1)

    # 使用 min 寻找最小值
    # key 的比较顺序：
    # 1. 面积 (越小越好)
    # 2. r[0] 即 top 行号 (越小越好，即越靠上)
    # 3. r[1] 即 left 列号 (越小越好，即越靠左)
    return min(
        rectangles,
        key=lambda r: (get_area(r), r[0], r[1]),
    )

def get_non_overlapping_rectangles(rectangles):
    """
    贪心策略：
    1. 先对所有矩形排序（优先面积小，其次靠上，其次靠左）。
    2. 依次选取，如果当前矩形与已选的矩形不重叠（不共用格子），则选中。
    返回一个不重叠的矩形列表。
    """
    if not rectangles:
        return []

    # 1. 排序：策略与之前相同，优先消除小的
    # (面积, Top行, Left列)
    sorted_rects = sorted(
        rectangles,
        key=lambda r: ((r[2] - r[0] + 1) * (r[3] - r[1] + 1), r[0], r[1])
    )

    selected = []
    # 建立一个掩膜标记已被占用的格子
    occupied = np.zeros((BOARD_ROWS, BOARD_COLS), dtype=bool)

    for (top, left, bottom, right) in sorted_rects:
        # 检查当前矩形范围内是否有格子已被占用
        is_clash = False
        for r in range(top, bottom + 1):
            for c in range(left, right + 1):
                if occupied[r, c]:
                    is_clash = True
                    break
            if is_clash:
                break
        
        # 如果没有冲突，则选中该矩形，并标记占用
        if not is_clash:
            selected.append((top, left, bottom, right))
            for r in range(top, bottom + 1):
                for c in range(left, right + 1):
                    occupied[r, c] = True

    return selected


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


def toggle_live_mode():
    """切换实时识别模式的开关函数"""
    global live_mode_active, live_mode_thread
    
    if live_mode_active:
        print("[INFO] 实时识别模式已在运行中，按 ESC 退出当前模式")
        return
    
    if live_mode_thread is not None and live_mode_thread.is_alive():
        print("[INFO] 实时识别模式正在启动中，请稍候...")
        return
    
    print("[INFO] 启动实时识别模式...")
    live_mode_thread = threading.Thread(target=main_live)
    live_mode_thread.daemon = True
    live_mode_thread.start()


def main_live():
    global live_mode_active, refresh_requested  # 【修改】引入 refresh_requested
    
    digit_templates = DigitTemplates(TEMPLATES_DIR)

    # --- 1. 获取全屏截图用于框选 ---
    with mss() as sct:
        monitor = sct.monitors[1] 
        screenshot_bgra = np.array(sct.grab(monitor))
        screenshot = cv2.cvtColor(screenshot_bgra, cv2.COLOR_BGRA2BGR)

    # --- 2. 用户框选区域 ---
    print("[INFO] 请框选【仅包含数字】的区域！")
    x, y, w, h = select_board_region_once(screenshot)
    
    monitor_rect = {
        "left": monitor["left"] + x,
        "top": monitor["top"] + y,
        "width": w,
        "height": h
    }

    print(f"[INFO] 锁定区域: {monitor_rect}")
    print("[INFO] 操作指南：")
    print("  - 程序会自动每 5 秒刷新一次结果")
    print("  - 全局热键 【Ctrl+Shift+Z】：立即刷新")
    print("  - 按 【ESC】 键：退出")
    
    live_mode_active = True
    
    # 初始化状态
    last_process_time = 0
    REFRESH_INTERVAL = 3.0 
    current_vis = np.zeros((h, w, 3), dtype=np.uint8)

    # --- 3. 主循环 ---
    with mss() as sct:
        while live_mode_active:
            current_time = time.time()
            
            # 这里的 waitKey 主要是为了响应 ESC 退出和维持窗口刷新
            key = cv2.waitKey(50) & 0xFF 
            
            if key == 27: # ESC
                live_mode_active = False
                break
            
            # 【关键修改】判断刷新条件：
            # 1. 窗口处于焦点时按了 'R'
            # 2. 或者 全局热键标志位被置为 True
            # 3. 或者 时间间隔到了
            needs_refresh = (key == ord('r')) or \
                            refresh_requested or \
                            (current_time - last_process_time > REFRESH_INTERVAL)

            if needs_refresh:
                # 【关键修改】如果是全局热键触发的，处理完后要复位，防止无限刷新
                if refresh_requested:
                    refresh_requested = False
                
                # ... 下面是原有的识别逻辑，保持不变 ...
                img_bgra = np.array(sct.grab(monitor_rect))
                frame = img_bgra[:, :, :3]
                frame = np.ascontiguousarray(frame)

                nums, unknown_mask = ocr_board(frame, digit_templates)
                all_rectangles = find_rectangles_sum_k(nums, unknown_mask, TARGET_SUM)
                best_batch = get_non_overlapping_rectangles(all_rectangles)

                vis = frame.copy()
                
                # 画网格
                cell_h = vis.shape[0] / BOARD_ROWS
                cell_w = vis.shape[1] / BOARD_COLS
                for r in range(BOARD_ROWS + 1):
                    cv2.line(vis, (0, int(r * cell_h)), (vis.shape[1], int(r * cell_h)), (200, 200, 200), 1)
                for c in range(BOARD_COLS + 1):
                    cv2.line(vis, (int(c * cell_w), 0), (int(c * cell_w), vis.shape[0]), (200, 200, 200), 1)

                # 画数字
                for r in range(BOARD_ROWS):
                    for c in range(BOARD_COLS):
                        digit = nums[r, c]
                        cx = int(c * cell_w + cell_w * 0.3)
                        cy = int(r * cell_h + cell_h * 0.7)
                        
                        if unknown_mask[r, c]:
                            cv2.putText(vis, "?", (cx, cy), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)
                        elif digit == 0:
                            cv2.circle(vis, (int(c * cell_w + cell_w/2), int(r * cell_h + cell_h/2)), 2, (100, 100, 100), -1)
                        else:
                            cv2.putText(vis, str(digit), (cx, cy), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)

                # 画红框
                if best_batch:
                    for (top, left, bottom, right) in best_batch:
                        x1 = int(left * cell_w)
                        y1 = int(top * cell_h)
                        x2 = int((right + 1) * cell_w)
                        y2 = int((bottom + 1) * cell_h)
                        cv2.rectangle(vis, (x1, y1), (x2, y2), RECT_COLOR, 3)
                    msg = f"Batch: {len(best_batch)}"
                else:
                    msg = "No Solution"

                cv2.rectangle(vis, (0, 0), (vis.shape[1], 5), (0, 0, 0), -1) 
                current_vis = vis
                last_process_time = time.time()

            # --- 显示部分 ---
            display_img = current_vis.copy()
            elapsed = current_time - last_process_time
            ratio = 1.0 - (elapsed / REFRESH_INTERVAL)
            if ratio < 0: ratio = 0
            
            bar_w = int(display_img.shape[1] * ratio)
            cv2.rectangle(display_img, (0, 0), (bar_w, 5), (0, 255, 0), -1)

            cv2.imshow("Ten Helper", display_img)

    cv2.destroyAllWindows()


if __name__ == "__main__":
    # main_single_image()
    # main_single_image("debug_live_frame.png")
    print("[INFO] Ten Helper 已启动")
    print("[INFO] 按 Ctrl+Shift+X 启动/停止实时识别模式")
    print("[INFO] 按 Ctrl+C 退出程序")
    
    # 注册全局热键 Ctrl+Shift+X
    keyboard.add_hotkey('ctrl+shift+x', toggle_live_mode)
    # 注册全局热键 Ctrl+Shift+Z
    keyboard.add_hotkey('ctrl+shift+z', trigger_global_refresh)

    try:
        # 保持程序运行，等待热键触发
        while True:
            time.sleep(0.1)
    except KeyboardInterrupt:
        print("[INFO] 程序已退出")
    finally:
        # 清理资源
        if live_mode_active:
            live_mode_active = False
            if live_mode_thread is not None and live_mode_thread.is_alive():
                live_mode_thread.join(timeout=1.0)