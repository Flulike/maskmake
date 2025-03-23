import cv2
import numpy as np

class MaskDrawer:
    def __init__(self, image_path, brush_size=20):  # [+] 初始化时添加画笔尺寸参数
        self.image = cv2.imread(image_path)
        if self.image is None:
            raise ValueError("无法读取图像")
        self.mask = np.zeros(self.image.shape[:2], dtype=np.uint8)
        self.drawing = False
        self.prev_point = None
        self.brush_size = brush_size  # [+] 画笔尺寸变量
        self.window_name = "Draw Mask (Press +/-调整粗细 | ENTER确认)"
        
        cv2.namedWindow(self.window_name)
        cv2.setMouseCallback(self.window_name, self._mouse_callback)

    def _mouse_callback(self, event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN:
            self.drawing = True
            self.prev_point = (x, y)
            # [+] 在点击时也绘制圆形确保起始点可见
            cv2.circle(self.mask, (x, y), self.brush_size//2, 255, -1)
        elif event == cv2.EVENT_MOUSEMOVE:
            if self.drawing and self.prev_point:
                # [+] 使用更平滑的线段连接
                cv2.line(self.mask, self.prev_point, (x, y), 255, self.brush_size)
                self.prev_point = (x, y)
        elif event == cv2.EVENT_LBUTTONUP:
            self.drawing = False
            cv2.line(self.mask, self.prev_point, (x, y), 255, self.brush_size)
            self.prev_point = None

    def get_mask(self):
        while True:
            # 显示叠加效果时增加画笔尺寸提示
            overlay = self.image.copy()
            cv2.putText(overlay, f"Brush: {self.brush_size}", (10, 30),
                       cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
            
            # [+] 使用更醒目的红色半透明叠加
            overlay = cv2.addWeighted(
                overlay, 0.8,
                cv2.cvtColor(self.mask, cv2.COLOR_GRAY2BGR), 0.2, 0
            )
            
            cv2.imshow(self.window_name, overlay)
            
            key = cv2.waitKey(1)
            if key == 13:  # Enter键确认
                break
            elif key == 43:  # '+'键增大画笔
                self.brush_size = min(100, self.brush_size + 2)
            elif key == 45:  # '-'键减小画笔
                self.brush_size = max(2, self.brush_size - 2)
        
        # [+] 保存全白mask的二进制文件
        cv2.imwrite("mask.png", self.mask)
        cv2.destroyAllWindows()
        return self.mask

# 使用示例
def interactive_remove(image_path, output_path="result.png"):
    # [+] 初始化时设置默认画笔大小为20像素
    drawer = MaskDrawer(image_path, brush_size=20)
    mask = drawer.get_mask()
    
    # 使用改进后的修复参数
    result = cv2.inpaint(
        drawer.image, mask,
        inpaintRadius=3,  # 与画笔尺寸联动的修复半径
        flags=cv2.INPAINT_TELEA
    )
    
    cv2.imwrite(output_path, result)
    print(f"结果已保存至 {output_path}")
    return result

# 使用示例
interactive_remove("upocr.png", "output_interactive.png")