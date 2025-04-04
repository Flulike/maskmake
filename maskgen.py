from PIL import Image
import numpy as np

# 加载PNG图片
image_path = "cola-removebg-preview.png"  # 替换为您的图片路径
img = Image.open(image_path).convert("RGBA")

# 将图像转换为NumPy数组以便处理
data = np.array(img)

# 创建掩码：非透明区域(物品)为白色(255)，透明区域为黑色(0)
# 获取alpha通道(透明度)
alpha = data[:, :, 3]

# 创建掩码图像：alpha > 0的地方是物品，设为白色(255)；其他地方设为黑色(0)
mask = np.zeros_like(alpha)
mask[alpha > 0] = 255

# 创建新的掩码图像
mask_img = Image.fromarray(mask, mode="L")

# 保存掩码图像
mask_img.save("mask_output.png")

print("掩码已生成并保存为mask_output.png")