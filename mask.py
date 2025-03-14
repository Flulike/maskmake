import cv2
import numpy as np

# 全局变量
points_orig = []  # 存储原图坐标的四个点
img_orig = None    # 原始图像
scale = 1.0        # 图像缩放比例

def click_event(event, x_display, y_display, flags, param):
    """ 鼠标回调函数，处理点击事件 """
    global points_orig, img_orig, scale
    
    if event == cv2.EVENT_LBUTTONDOWN and len(points_orig) < 4:
        # 将显示坐标转换为原图坐标
        x_orig = int(x_display / scale)
        y_orig = int(y_display / scale)
        points_orig.append((x_orig, y_orig))
        
        # 在显示图像上绘制点击效果
        display_h, display_w = int(img_orig.shape[0] * scale), int(img_orig.shape[1] * scale)
        img_display = cv2.resize(img_orig.copy(), (display_w, display_h))
        
        # 绘制所有已有点和连线
        for i, (x, y) in enumerate(points_orig):
            # 转换回显示坐标进行绘制
            x_disp = int(x * scale)
            y_disp = int(y * scale)
            cv2.circle(img_display, (x_disp, y_disp), 5, (0, 0, 255), -1)
            if i > 0:
                x_prev = int(points_orig[i-1][0] * scale)
                y_prev = int(points_orig[i-1][1] * scale)
                cv2.line(img_display, (x_prev, y_prev), (x_disp, y_disp), (0, 255, 0), 2)
        
        # 如果已满四个点，绘制闭合多边形
        if len(points_orig) == 4:
            pts_display = np.array([[int(x*scale), int(y*scale)] for (x,y) in points_orig], np.int32)
            cv2.polylines(img_display, [pts_display], True, (0, 255, 255), 2)
        
        cv2.imshow('image', img_display)

def main():
    global img_orig, scale
    
    # 读取图像
    img_path = input("请输入图像路径: ").strip('"')  # 处理拖放文件可能包含的引号
    img_orig = cv2.imread(img_path)
    if img_orig is None:
        print("错误：无法加载图像")
        return
    
    # 计算缩放比例以适应显示
    h, w = img_orig.shape[:2]
    max_display_size = 800
    scale = min(max_display_size/h, max_display_size/w)
    display_w, display_h = int(w*scale), int(h*scale)
    
    # 创建窗口并设置回调
    cv2.namedWindow('image')
    cv2.setMouseCallback('image', click_event)
    
    # 初始显示缩放后的图像
    img_display = cv2.resize(img_orig, (display_w, display_h))
    cv2.imshow('image', img_display)
    
    # 等待用户选择四个点
    while len(points_orig) < 4:
        key = cv2.waitKey(20)
        if key == 27:  # ESC键退出
            cv2.destroyAllWindows()
            return
    
    # 生成遮罩
    mask = np.zeros(img_orig.shape[:2], dtype=np.uint8)
    pts = np.array(points_orig, dtype=np.int32).reshape((-1,1,2))
    cv2.fillPoly(mask, [pts], 255)
    
    # 创建半透明红色遮罩层
    overlay = img_orig.copy()
    cv2.fillPoly(overlay, [pts], (255, 255, 255))
    alpha = 1  # 透明度
    
    # 合并图像
    masked_img = cv2.addWeighted(img_orig, 1 - alpha, overlay, alpha, 0)
    
    # 应用遮罩
    result = img_orig.copy()
    result[mask == 255] = masked_img[mask == 255]
    
    # 显示并保存结果
    cv2.imshow('Result', result)
    cv2.waitKey(0)
    cv2.imwrite('masked_output.jpg', result)
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()