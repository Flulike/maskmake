import cv2
import numpy as np

# 全局变量
all_points = []     # 存储所有区域的坐标点
current_points = [] # 当前正在标注的区域点
img_orig = None     # 原始图像
scale = 1.0         # 图像缩放比例
drawing_mode = True # 标注模式状态

def click_event(event, x_display, y_display, flags, param):
    """ 鼠标回调函数，处理点击事件 """
    global current_points, img_orig, scale, drawing_mode
    
    if not drawing_mode:
        return
    
    if event == cv2.EVENT_LBUTTONDOWN and len(current_points) < 4:
        # 转换坐标到原图尺寸
        x_orig = int(x_display / scale)
        y_orig = int(y_display / scale)
        current_points.append((x_orig, y_orig))
        
        # 更新显示图像
        update_display()

def update_display():
    """ 更新显示图像 """
    display_h = int(img_orig.shape[0] * scale)
    display_w = int(img_orig.shape[1] * scale)
    img_display = cv2.resize(img_orig.copy(), (display_w, display_h))
    
    # 绘制所有已完成的区域
    for points in all_points:
        # 绘制点
        for i, (x, y) in enumerate(points):
            x_disp = int(x * scale)
            y_disp = int(y * scale)
            cv2.circle(img_display, (x_disp, y_disp), 5, (0, 0, 255), -1)
            if i > 0:
                x_prev = int(points[i-1][0] * scale)
                y_prev = int(points[i-1][1] * scale)
                cv2.line(img_display, (x_prev, y_prev), (x_disp, y_disp), (0, 255, 0), 2)
        
        # 绘制闭合框
        if len(points) == 4:
            pts_display = np.array([[int(x*scale), int(y*scale)] for (x,y) in points], np.int32)
            cv2.polylines(img_display, [pts_display], True, (0, 0, 255), 2)
    
    # 绘制当前正在标注的区域
    for i, (x, y) in enumerate(current_points):
        x_disp = int(x * scale)
        y_disp = int(y * scale)
        cv2.circle(img_display, (x_disp, y_disp), 5, (255, 255, 255), -1)
        if i > 0:
            x_prev = int(current_points[i-1][0] * scale)
            y_prev = int(current_points[i-1][1] * scale)
            cv2.line(img_display, (x_prev, y_prev), (x_disp, y_disp), (0, 255, 0), 2)
    
    cv2.imshow('image', img_display)

def calculate_boxes():
    """ 计算所有区域的归一化坐标 """
    h, w = img_orig.shape[:2]
    boxes = []
    
    for points in all_points:
        x_coords = [p[0] for p in points]
        y_coords = [p[1] for p in points]
        x_min, x_max = min(x_coords), max(x_coords)
        y_min, y_max = min(y_coords), max(y_coords)
        
        box = [
            round(x_min / w, 4),
            round(y_min / h, 4),
            round(x_max / w, 4),
            round(y_max / h, 4)
        ]
        boxes.append(box)
    
    return boxes

def apply_masks():
    """ 应用所有遮罩到图像 """
    result = img_orig.copy()
    alpha = 1  # 透明度
    
    for points in all_points:
        # 创建遮罩
        mask = np.zeros(img_orig.shape[:2], dtype=np.uint8)
        pts = np.array(points, dtype=np.int32).reshape((-1,1,2))
        cv2.fillPoly(mask, [pts], 255)
        
        # 创建遮罩层
        overlay = img_orig.copy()
        cv2.fillPoly(overlay, [pts], (255, 255, 255))
        masked_img = cv2.addWeighted(img_orig, 1 - alpha, overlay, alpha, 0)
        
        # 应用遮罩
        result[mask == 255] = masked_img[mask == 255]
    
    return result

def main():
    global img_orig, scale, current_points, all_points, drawing_mode
    
    # 读取图像
    #img_path = input("请输入图像路径: ").strip('"')
    img_path = "upocr.png"
    img_orig = cv2.imread(img_path)
    if img_orig is None:
        print("错误：无法加载图像")
        return
    
    # 计算缩放比例
    h, w = img_orig.shape[:2]
    max_display_size = 800
    scale = min(max_display_size/h, max_display_size/w)
    display_w, display_h = int(w*scale), int(h*scale)
    
    # 创建窗口
    cv2.namedWindow('image')
    cv2.setMouseCallback('image', click_event)
    
    # 初始显示
    update_display()
    
    # 主循环
    while True:
        key = cv2.waitKey(20)
        
        # ESC键退出
        if key == 27:
            break
        
        # 回车键完成当前区域
        if key == 13 and len(current_points) == 4:
            all_points.append(current_points.copy())
            current_points.clear()
            update_display()
            print(f"已保存区域 {len(all_points)}，按ESC结束标注")
        
        # 退格键删除上一个点
        if key == 8 and len(current_points) > 0:
            current_points.pop()
            update_display()
    
    # 处理结果
    if len(all_points) > 0:
        # 应用所有遮罩
        result = apply_masks()
        
        # 显示并保存结果
        cv2.imshow('Result', result)
        cv2.waitKey(0)
        cv2.imwrite('masked_output.jpg', result)
        
        # 计算并保存boxes坐标
        boxes = calculate_boxes()
        print("\n生成的归一化boxes坐标：")
        print("boxes = [")
        for box in boxes:
            print(f"    {box},")
        print("]")
        
        with open('boxes.txt', 'w') as f:
            f.write("boxes = [\n")
            for box in boxes:
                f.write(f"    {box},\n")
            f.write("]")
    
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()