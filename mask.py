import cv2
import numpy as np

class ImageMasker:
    def __init__(self):
        self.all_points = []     # 所有标注区域
        self.current_points = [] # 当前区域点
        self.img_orig = None     # 原始图像
        self.scale = 1.0         # 缩放比例
        self.drawing = False     # 标注状态

    def _click_event(self, event, x_display, y_display, flags, param):
        """ 鼠标事件回调 """
        if event == cv2.EVENT_LBUTTONDOWN and len(self.current_points) < 4:
            x_orig = int(x_display / self.scale)
            y_orig = int(y_display / self.scale)
            self.current_points.append((x_orig, y_orig))
            self._update_display()

    def _update_display(self):
        """ 更新显示画面 """
        display_img = cv2.resize(self.img_orig.copy(), 
                               (int(self.img_orig.shape[1]*self.scale), 
                                int(self.img_orig.shape[0]*self.scale)))
        
        # 绘制历史区域
        for points in self.all_points:
            self._draw_polygon(display_img, points, (0, 255, 0))
        
        # 绘制当前区域
        if self.current_points:
            self._draw_polygon(display_img, self.current_points, (0, 0, 255))
        
        cv2.imshow('Mask Tool', display_img)

    def _draw_polygon(self, img, points, color):
        """ 绘制多边形辅助线 """
        pts_display = [(int(x*self.scale), int(y*self.scale)) for (x,y) in points]
        for i, (x, y) in enumerate(pts_display):
            cv2.circle(img, (x, y), 5, color, -1)
            if i > 0:
                cv2.line(img, pts_display[i-1], (x, y), color, 2)
        if len(points) == 4:
            cv2.polylines(img, [np.array(pts_display, np.int32)], True, color, 2)

    def process_image(self, image_path):
        """
        处理主函数
        :param image_path: 输入图像路径
        :return: (处理后的图像, 归一化boxes列表)
        """
        # 初始化状态
        self.__init__() 
        
        # 读取图像
        self.img_orig = cv2.imread(image_path)
        if self.img_orig is None:
            raise FileNotFoundError(f"无法加载图像: {image_path}")
        
        # 设置窗口
        h, w = self.img_orig.shape[:2]
        self.scale = min(800/w, 800/h)
        cv2.namedWindow('Mask Tool')
        cv2.setMouseCallback('Mask Tool', self._click_event)
        self._update_display()

        # 主交互循环
        while True:
            key = cv2.waitKey(20)
            
            # ESC键结束标注
            if key == 27: 
                break
            
            # 回车确认当前区域
            if key == 13 and len(self.current_points) == 4:
                self.all_points.append(self.current_points)
                self.current_points = []
                self._update_display()
                print(f"已保存区域 {len(self.all_points)}")
            
            # 退格键删除上一个点
            if key == 8 and self.current_points:
                self.current_points.pop()
                self._update_display()

        # 生成最终结果
        result_image = self._apply_masks()
        boxes = self._calculate_boxes()
        
        # 清理资源
        cv2.destroyAllWindows()
        return result_image, boxes

    def _apply_masks(self):
        """ 应用所有遮罩 """
        result = self.img_orig.copy()
        for points in self.all_points:
            mask = np.zeros_like(result[:, :, 0])
            cv2.fillPoly(mask, [np.array(points, np.int32)], 255)
            overlay = cv2.addWeighted(result, 1, 
                                     cv2.bitwise_and(result, result, mask=~mask), 
                                     1, 0)
            result = cv2.bitwise_or(overlay, 
                                  cv2.bitwise_and(result, result, mask=mask))
        return result

    def _calculate_boxes(self):
        """ 计算归一化坐标 """
        h, w = self.img_orig.shape[:2]
        boxes = []
        for points in self.all_points:
            x = [p[0] for p in points]
            y = [p[1] for p in points]
            box = [
                round(min(x)/w, 4),
                round(min(y)/h, 4),
                round(max(x)/w, 4),
                round(max(y)/h, 4)
            ]
            boxes.append(box)
        return boxes

# 使用示例 ---------------------------------------------------------------------
if __name__ == "__main__":
    # 初始化处理器
    masker = ImageMasker()
    
    # 示例图像路径
    image_path = "image.png"  # 修改为实际路径
    
    # 处理图像并获取结果
    try:
        output_image, boxes = masker.process_image(image_path)
        
        # 保存结果
        cv2.imwrite("output.jpg", output_image)
        print("\n生成的归一化坐标：")
        for i, box in enumerate(boxes):
            print(f"Box {i+1}: {box}")
        
        cv2.imshow("Final Result", output_image)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
        
    except Exception as e:
        print(f"处理出错: {str(e)}")