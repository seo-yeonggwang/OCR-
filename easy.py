import numpy as np
import pytesseract
import easyocr
import random
import math
import cv2
import matplotlib.pyplot as plt
from PIL import Image, ImageDraw, ImageFont

from collections import Counter

def preprocess_image(image):
    image = cv2.imread(image)

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) # RGB -> Gray
    blurred = cv2.GaussianBlur(gray, (5, 5), 0) # 흐림처리(노이즈 제거) : 높을수록 더 넓은 영역을 평균화하여 블러 효과가 강해질 수 있음
    #edged = cv2.Canny(blurred, 100, 200) # 엣지 검출 (이미지, 최소강도, 최대강도) : 최소강도보다 작거나 최대강도보다 큰 경우 엣지로 간주하지 않음
    _, binary = cv2.threshold(blurred, 127, 255, cv2.THRESH_BINARY_INV) # 이진화

    return binary

def PrintText(preImage, is_horizontal):
    reader = easyocr.Reader(['ko', 'en'])
    result = reader.readtext(preImage)

    if is_horizontal:
        # 수평이라면
        rotated_image_0 = preImage  # 변함 없음
        rotated_image_180, _ = RotateImage(preImage, 180)

        result_0 = result
        result_180 = reader.readtext(rotated_image_180)

        text_0, box_0 = count_text(result_0, 0)
        text_180, box_180 = count_text(result_180, 180)

        if text_0 == text_180:
            if box_0 > box_180:
                DrawBox(rotated_image_0, result_0)
            elif box_0 < box_180:
                DrawBox(rotated_image_180, result_180)
            else:
                print("방향을 인식하지 못했습니다.")
        elif text_0 > text_180:
            DrawBox(rotated_image_0, result_0)
        else:
            DrawBox(rotated_image_180, result_180)

    else:
        # Text is vertical
        rotated_image_90, _ = RotateImage(preImage, 90)
        rotated_image_270, _ = RotateImage(preImage, 270)

        result_90 = reader.readtext(rotated_image_90)
        result_270 = reader.readtext(rotated_image_270)

        text_90, box_90 = count_text(result_90, 90)
        text_270, box_270 = count_text(result_270, 270)
        
        if text_90 == text_270:
            if box_90 > box_270:
                DrawBox(rotated_image_90, result_90)
            elif box_90 < box_270:
                DrawBox(rotated_image_270, result_270)
            else:
                print("방향을 인식하지 못했습니다.")
        elif text_90 > text_270:
            DrawBox(rotated_image_90, result_90)
        else:
            DrawBox(rotated_image_270, result_270)

def count_text(result, angle):
    total_count = 0
    box_count = len(result)  # 바운딩 박스의 개수
    for (bbox, text, prob) in result:
        # 한글, 영어, 숫자만 계산
        total_count += sum(c.isalnum() for c in text)
    print(f"Angle: {angle}, Total Count: {total_count}, Box Count: {box_count}")
    return total_count, box_count

# 이미지 회전 함수
def RotateImage(preImage, angle):
    h, w = preImage.shape[:2] # 높이, 너비
    center = (w // 2, h // 2) # 중점
    M = cv2.getRotationMatrix2D(center, -angle, 1.0) # angle이 음수 : 시계 방향, angle이 양수 : 반시계 방향

    # 회전 후 이미지의 크기 계산
    cos = np.abs(M[0, 0])
    sin = np.abs(M[0, 1])

    new_w = int((h * sin) + (w * cos))
    new_h = int((h * cos) + (w * sin))

    # 회전 후에도 모든 픽셀을 포함할 수 있도록 이미지 크기 조정
    M[0, 2] += (new_w / 2) - center[0]
    M[1, 2] += (new_h / 2) - center[1]

    rotated_img = cv2.warpAffine(preImage, M, (w, h))

    return rotated_img, angle

# 텍스트 상자 그리기 함수
def DrawBox(preImage, result):
    img_pil = Image.fromarray(cv2.cvtColor(preImage, cv2.COLOR_BGR2RGB))
    font_path = 'Roboto-Black.ttf'
    font_size = 50
    font = ImageFont.truetype(font_path, font_size)
    draw = ImageDraw.Draw(img_pil)
    # draw = ImageDraw.Draw(img)
    np.random.seed(42)
    COLORS = np.random.randint(0, 255, size=(255, 3), dtype="uint8")

    for i in result:
        box = i[0]
        x = int(box[0][0])
        y = int(box[0][1])
        w = int(box[1][0] - box[0][0])
        h = int(box[2][1] - box[1][1])

        color_idx = random.randint(0, 200)
        color = [int(c) for c in COLORS[color_idx]]

        draw.rectangle(((x, y), (x + w, y + h)), outline=tuple(color), width=2)
        draw.text((x, y - 50), str(i[1]), font=font, fill=tuple(color))

    plt.figure(figsize=(12, 12))
    plt.imshow(img_pil)
    plt.axis('off')
    plt.show()

def analyze_projection(image):
    vertical_projection = np.sum(image, axis=0) # 각 열을 따라 더하기
    horizontal_projection = np.sum(image, axis=1) # 각 행을 따라 더하기

    v_max = np.max(vertical_projection)
    h_max = np.max(horizontal_projection)

    if v_max > h_max:
        check = False
    else:
        check = True
    return check

# 메인 함수
if __name__ == "__main__":
    image_path = 'exam5.jpg'

    preImage = preprocess_image(image_path)
    # rotation_angle = get_osd_orientation(preImage)

    is_horizontal = analyze_projection(preImage)
    print("가로입니까? : ",is_horizontal)

    # PrintText(preImage, rotation_angle)
    PrintText(preImage, is_horizontal)