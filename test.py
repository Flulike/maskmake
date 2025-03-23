import cv2

image_path = 'upocr.png'
mask_path = 'mask.png'
image = cv2.imread(image_path)
mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)

result = cv2.inpaint(
        image, mask,
        inpaintRadius=1,  # 与画笔尺寸联动的修复半径
        flags=cv2.INPAINT_TELEA
    )

cv2.imwrite("test_output.png", result)