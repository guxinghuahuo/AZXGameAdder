import cv2
import numpy as np
import pytesseract

# ================= 配置区域 =================
pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'
IMAGE_PATH = '1.png' 

ROWS = 14
COLS = 8
TARGET_SUM = 10
# PSM 10: 单个字符模式
CONFIG_OCR = r'--oem 3 --psm 10 -c tessedit_char_whitelist=123456789' # 注意：去掉了0，因为游戏里没有0

def get_grayscale_stats(img):
    """辅助函数：分析图片亮度分布，用于自动确定阈值"""
    return np.mean(img), np.std(img)

def solve_puzzle_static(image_path):
    img = cv2.imread(image_path)
    if img is None:
        print(f"错误：找不到图片 {image_path}")
        return

    # 1. 预处理
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    # 这一步是为了找格子位置，阈值保持 v2/v3 的设定即可
    _, thresh_grid = cv2.threshold(gray, 100, 255, cv2.THRESH_BINARY_INV)

    # 2. 轮廓检测 (找格子)
    contours, _ = cv2.findContours(thresh_grid, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    
    valid_cells = []
    for c in contours:
        x, y, w, h = cv2.boundingRect(c)
        aspect_ratio = w / float(h)
        # 4K 宽容度过滤
        if 40 < w < 400 and 40 < h < 400 and 0.8 < aspect_ratio < 1.4:
            valid_cells.append((x, y, w, h))

    if not valid_cells:
        print("未找到任何格子")
        return

    # === 排序与筛选 (沿用 v3 的逻辑) ===
    avg_h = np.mean([c[3] for c in valid_cells])
    valid_cells.sort(key=lambda k: k[1]) # 先按Y排
    
    sorted_cells = []
    current_row = [valid_cells[0]]
    for c in valid_cells[1:]:
        if c[1] < current_row[0][1] + (avg_h / 2):
            current_row.append(c)
        else:
            current_row.sort(key=lambda k: k[0]) # 行内按X排
            sorted_cells.extend(current_row)
            current_row = [c]
    current_row.sort(key=lambda k: k[0])
    sorted_cells.extend(current_row)
    
    if len(sorted_cells) >= ROWS * COLS:
        # 取最后 112 个
        grid_cells = sorted_cells[-(ROWS*COLS):]
    else:
        print(f"❌ 警告：只找到 {len(sorted_cells)} 个格子，少于 {ROWS*COLS}")
        grid_cells = sorted_cells

    # === OCR 识别 (全新逻辑) ===
    matrix = np.zeros((ROWS, COLS), dtype=int)
    print("开始智能 OCR 识别...")

    debug_images = [] # 用于拼图调试

    for i, (x, y, w, h) in enumerate(grid_cells):
        r, c = divmod(i, COLS)
        if r >= ROWS: break

        # 1. 提取原始 ROI
        roi = gray[y:y+h, x:x+w]
        
        # 2. 第一次粗略裁切 (去掉边缘的桶壁)
        # 上下左右各去掉 15%
        pad_x = int(w * 0.15)
        pad_y = int(h * 0.15)
        roi_inner = roi[pad_y:h-pad_y, pad_x:w-pad_x]
        
        # 3. 高阈值二值化 (提取白色数字)
        # 游戏数字是纯白(255)，桶是灰的。我们提高阈值只留白色。
        # 180 是个安全值，任何灰色都会变黑，只有白色字保留
        _, digit_mask = cv2.threshold(roi_inner, 180, 255, cv2.THRESH_BINARY)
        
        # 4. 寻找数字的精确包围盒 (Bounding Box)
        # 这样可以忽略周围所有干扰，只聚焦数字本身
        digit_contours, _ = cv2.findContours(digit_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        final_roi = None
        
        # 找到最大的白色块（通常就是数字）
        if digit_contours:
            max_c = max(digit_contours, key=cv2.contourArea)
            dx, dy, dw, dh = cv2.boundingRect(max_c)
            
            # 如果找到的白色块太小（噪点），就还是用整个图
            if dw > 5 and dh > 10:
                # 裁剪出纯数字部分
                final_roi = digit_mask[dy:dy+dh, dx:dx+dw]
            else:
                final_roi = digit_mask
        else:
            # 没找到白色块？可能是全黑，或者阈值太高
            final_roi = digit_mask

        # 5. 反色 (Invert) -> 变成 白底黑字
        # 此时 final_roi 是 黑底白字 (数字是255)
        # 我们要 OCR 识别，通常需要 白底黑字
        final_roi = cv2.bitwise_not(final_roi)

        # 6. 统一尺寸 (Resize)
        # 调整到固定高度 40px，保持比例
        target_h = 40
        scale = target_h / final_roi.shape[0]
        target_w = int(final_roi.shape[1] * scale)
        final_roi = cv2.resize(final_roi, (target_w, target_h), interpolation=cv2.INTER_LINEAR)
        
        # 7. 再次二值化清理缩放产生的灰度
        _, final_roi = cv2.threshold(final_roi, 127, 255, cv2.THRESH_BINARY)

        # 8. 加边框 (Padding)
        # Tesseract 需要字周围有留白
        final_roi = cv2.copyMakeBorder(final_roi, 10, 10, 10, 10, cv2.BORDER_CONSTANT, value=255)

        # 收集调试图
        debug_img_small = cv2.resize(final_roi, (50, 50))
        debug_images.append(debug_img_small)

        # 9. 识别
        text = pytesseract.image_to_string(final_roi, config=CONFIG_OCR).strip()
        
        # 映射表：处理像素字体常见误认
        replace_map = {
            'l': '1', 'I': '1', '|': '1', ']': '1', '[': '1',
            'O': '0', 'Q': '0', 'D': '0',
            'Z': '2',
            'S': '5', '$': '5',
            'B': '8',
            'A': '4',
            'G': '6',
            'q': '9'
        }
        for k, v in replace_map.items():
            text = text.replace(k, v)

        try:
            val = int(text)
            matrix[r][c] = val
        except:
            # 识别失败时，记为 -1 方便我们在日志里看到
            matrix[r][c] = 0
            
        # 标注
        cv2.putText(img, str(matrix[r][c]), (x+w//2-10, y+h//2), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    # 保存调试拼图
    if debug_images:
        rows_img = []
        for r in range(ROWS):
            chunk = debug_images[r*COLS : (r+1)*COLS]
            if not chunk: break
            rows_img.append(np.hstack(chunk))
        if rows_img:
            montage = np.vstack(rows_img)
            cv2.imwrite("debug_ocr_final.jpg", montage)
            print("已保存 OCR 最终输入图: debug_ocr_final.jpg (请务必查看这张图！)")

    print("\n=== 识别结果矩阵 ===")
    print(matrix)

    # 简单校验
    zero_count = np.sum(matrix == 0)
    if zero_count > 5:
        print(f"⚠️ 警告: 还有 {zero_count} 个格子识别为 0 (失败)。")
        print("请打开 debug_ocr_final.jpg 查看这些格子是否变成了全白或全黑。")

    # 计算解
    solutions = []
    for r1 in range(ROWS):
        for c1 in range(COLS):
            for r2 in range(r1, ROWS):
                for c2 in range(c1, COLS):
                    s = np.sum(matrix[r1:r2+1, c1:c2+1])
                    if s == TARGET_SUM:
                        solutions.append((r1, c1, r2, c2))
                    elif s > TARGET_SUM:
                        break

    print(f"\n找到 {len(solutions)} 个解")
    
    # 画图
    for (r1, c1, r2, c2) in solutions:
        idx_tl = r1 * COLS + c1
        idx_br = r2 * COLS + c2
        if idx_tl < len(grid_cells) and idx_br < len(grid_cells):
            x1, y1, _, _ = grid_cells[idx_tl]
            x2, y2, w2, h2 = grid_cells[idx_br]
            cv2.rectangle(img, (x1, y1), (x2+w2, y2+h2), (0, 0, 255), 4)

    cv2.imwrite("result_v4.jpg", img)
    print("最终结果已保存为 result_v4.jpg")

if __name__ == "__main__":
    solve_puzzle_static(IMAGE_PATH)