import cv2
import numpy as np
import pytesseract
import pyautogui
import time
import tkinter as tk
from mss import mss
import threading

# ================= 配置区域 =================
# 请确保路径正确
pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'

ROWS = 14
COLS = 8
TARGET_SUM = 10
CONFIG_OCR = r'--oem 3 --psm 6 -c tessedit_char_whitelist=0123456789'

class GameAssistant:
    def __init__(self):
        self.root = tk.Tk()
        self.root.title("Game Overlay")
        self.root.attributes("-alpha", 0.3)
        self.root.attributes("-topmost", True)
        self.root.attributes("-fullscreen", True)
        self.root.attributes("-transparentcolor", "white")
        self.root.config(bg="white")
        
        self.canvas = tk.Canvas(self.root, width=self.root.winfo_screenwidth(), height=self.root.winfo_screenheight(), bg="white", highlightthickness=0)
        self.canvas.pack()
        
        # [修改点 1]：这里删除了 self.sct = mss()，因为不能跨线程使用
        
        self.running = True
        self.matrix = np.zeros((ROWS, COLS), dtype=int)
        self.cells_rects = []

    def capture_screen_and_detect_grid(self):
        """
        截取屏幕并尝试定位游戏区域。
        """
        # [修改点 2]：在截图函数内部初始化 mss，使用 with 语句自动管理资源
        with mss() as sct:
            # 获取主显示器
            monitor = sct.monitors[1]
            img = np.array(sct.grab(monitor))
        
        img_gray = cv2.cvtColor(img, cv2.COLOR_BGRA2GRAY)

        # 图像预处理
        _, thresh = cv2.threshold(img_gray, 100, 255, cv2.THRESH_BINARY_INV)

        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        valid_cells = []
        for c in contours:
            x, y, w, h = cv2.boundingRect(c)
            # 宽松一点的过滤条件
            aspect_ratio = w / float(h)
            # 这里的 30-100 像素是基于一般 1080p 屏幕的估算，如果是 2k/4k 屏可能要调大
            if 30 < w < 120 and 30 < h < 120 and 0.8 < aspect_ratio < 1.3:
                valid_cells.append((x, y, w, h))

        if len(valid_cells) < ROWS * COLS:
            # 这里的print可能会刷屏，实际使用可以注释掉
            # print(f"识别中... 当前找到疑似格子数: {len(valid_cells)}")
            return None, None

        # 排序：行优先(y)，列次之(x)
        # 增加容错：y坐标差异在15像素以内视为同一行
        valid_cells.sort(key=lambda k: (k[1] // 15, k[0]))
        
        # 截取前 14*8 个格子
        grid_cells = valid_cells[:ROWS * COLS]
        
        matrix = np.zeros((ROWS, COLS), dtype=int)
        
        for i, (x, y, w, h) in enumerate(grid_cells):
            r, c = divmod(i, COLS)
            if r >= ROWS: break
            
            roi = img_gray[y:y+h, x:x+w]
            # 增强对比度，专门针对白色数字
            _, roi_thresh = cv2.threshold(roi, 160, 255, cv2.THRESH_BINARY)
            
            # 增加 padding (白边) 可以提高 OCR 识别率
            roi_thresh = cv2.copyMakeBorder(roi_thresh, 2, 2, 2, 2, cv2.BORDER_CONSTANT, value=0)

            text = pytesseract.image_to_string(roi_thresh, config=CONFIG_OCR).strip()
            
            try:
                # 针对常见识别错误的简单修正
                text = text.replace('l', '1').replace('O', '0').replace('B', '8')
                num = int(text)
                matrix[r][c] = num
            except:
                matrix[r][c] = 0

        return matrix, grid_cells

    def find_solutions(self, matrix):
        solutions = []
        for r1 in range(ROWS):
            for c1 in range(COLS):
                for r2 in range(r1, ROWS):
                    # 优化：如果当前矩形宽度过大其实没必要算，一般框选不会跨越太远
                    # 这里为了全量解，暂时保留全遍历
                    sub_matrix = matrix[r1:r2+1, c1:COLS] # 先取行切片
                    
                    # 利用 numpy 快速计算前缀和或者滑动窗口会更快，但这里直接暴力遍历列
                    for c2 in range(c1, COLS):
                        # 重新切片获取精确矩形
                        current_rect = matrix[r1:r2+1, c1:c2+1]
                        s = np.sum(current_rect)
                        
                        if s == TARGET_SUM:
                            solutions.append((r1, c1, r2, c2))
                        elif s > TARGET_SUM:
                            # 剪枝：如果当前这一行加进去已经超了，对于这个 r1,r2 来说，c2再往右也没用了
                            # 注意：这个剪枝只在全正数矩阵有效（游戏里没有负数）
                            break
        return solutions

    def draw_box(self, solutions, grid_cells):
        self.canvas.delete("all") 
        if not solutions:
            return

        # 限制绘制数量，防止卡顿，只画前 20 个解
        for (r1, c1, r2, c2) in solutions[:30]:
            idx_tl = r1 * COLS + c1
            idx_br = r2 * COLS + c2
            
            if idx_tl < len(grid_cells) and idx_br < len(grid_cells):
                x1, y1, _, _ = grid_cells[idx_tl]
                x2, y2, w2, h2 = grid_cells[idx_br]
                
                # 计算中心点
                center_x = (x1 + x2 + w2) / 2
                center_y = (y1 + y2 + h2) / 2
                
                # 画框
                self.canvas.create_rectangle(x1-2, y1-2, x2+w2+2, y2+h2+2, outline="#FF0000", width=3)
                # 画个半透明填充提示
                # tkinter canvas 不支持直接 rgba fill，这里只画框

    def loop(self):
        print("后台扫描线程已启动...")
        while self.running:
            try:
                # 1. 识别
                matrix, grid_cells = self.capture_screen_and_detect_grid()
                
                if matrix is not None:
                    # 简单校验：如果矩阵全是0，说明识别出了问题
                    if np.sum(matrix) == 0:
                        print("警告：矩阵全为0，可能 OCR 阈值需要调整")
                        time.sleep(1)
                        continue

                    # 2. 计算
                    solutions = self.find_solutions(matrix)
                    if len(solutions) > 0:
                        print(f"找到 {len(solutions)} 个解")
                    
                    # 3. 绘图 (必须在主线程执行 UI 更新，使用 after)
                    # 虽然 tkinter 非线程安全，但 after 是线程安全的通信机制
                    self.root.after(0, self.draw_box, solutions, grid_cells)
                else:
                    # 未找到格子时，清空画布
                    self.root.after(0, self.canvas.delete, "all")
                
                time.sleep(0.5) # 0.5秒刷新一次，太快会闪烁且占CPU

            except Exception as e:
                print(f"Loop 发生错误: {e}")
                time.sleep(1)

    def start(self):
        t = threading.Thread(target=self.loop)
        t.daemon = True
        t.start()
        self.root.mainloop()

if __name__ == "__main__":
    print("程序启动中...")
    print("请确保游戏在前台，并且没有被其他窗口遮挡。")
    bot = GameAssistant()
    bot.start()