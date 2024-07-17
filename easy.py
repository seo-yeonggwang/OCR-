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

def PrintText(preImage, rotation_angle):
    reader = easyocr.Reader(['ko', 'en'])
    result = reader.readtext(preImage)

    # # 텍스트만 추출하고 공백으로 결합하여 출력
    # texts = [text for _, text, _ in result]
    # print(" ".join(texts))

    slope, rotate = Slope(result, rotation_angle)
    print("평균 기울기:", slope)

    # 이미지를 회전할 필요가 있는 경우
    if abs(slope) > 45:  # 기울기가 45도 초과이면 회전 필요
        rotated_image, angle = RotateImage(preImage, -rotate)
        rotated_result = reader.readtext(rotated_image)  # 회전된 이미지에서 텍스트 인식
        # print("result : ",rotated_result)
        DrawBox(rotated_image, rotated_result)  # 회전된 이미지에 대해 박스 그리기
    else:
        DrawBox(preImage, result)  # 회전할 필요가 없는 경우 그대로 박스 그리기

# def PrintText(preImage, is_horizontal):
#     reader = easyocr.Reader(['ko', 'en'])
#     result = reader.readtext(preImage)

#     if is_horizontal:
#         # Text is horizontal
#         rotated_image_0 = preImage  # No rotation needed
#         rotated_image_180, _ = RotateImage(preImage, 180)

#         result_0 = result
#         result_180 = reader.readtext(rotated_image_180)

#         # Compare recognition results
#         if count_text(result_0) > count_text(result_180):
#             DrawBox(rotated_image_0, result_0)
#         else:
#             DrawBox(rotated_image_180, result_180)

#     else:
#         # Text is vertical
#         rotated_image_90, _ = RotateImage(preImage, 90)
#         rotated_image_270, _ = RotateImage(preImage, 270)

#         result_90 = reader.readtext(rotated_image_90)
#         result_270 = reader.readtext(rotated_image_270)

#         # Compare recognition results
#         if count_text(result_90) > count_text(result_270):
#             DrawBox(rotated_image_90, result_90)
#         else:
#             DrawBox(rotated_image_270, result_270)

# def count_text(result):
#     return len(result)

# 기울기 계산
def Slope(result, rotation_angle):
    print("회전되어야 할 기울기 : ",rotation_angle)

    slopes = []

    for box, _, _ in result:
        x1, y1 = box[0]
        x2, y2 = box[1]

        rotate = rotation_angle
        print("rotate",rotate)

        slopes.append(calculate_slope2(x1, y1, x2, y2, rotate))

    # 평균 기울기 계산
    average_slope = sum(slopes) / len(slopes)

    return average_slope, rotate

# 두 점 사이의 거리 계산
def calculate_distance(x1, y1, x2, y2):
    distance = math.sqrt((x2 - x1)**2 + (y2 - y1)**2)

    return distance

# 두 점의 기울기 계산
def calculate_slope2(x1, y1, x2, y2, rotate):
    if x2 - x1 != 0:  # 분모가 0인 경우 방지
        slope = (y2 - y1) / (x2 - x1)
        print("slope", slope)

        if slope == 0: # 기울기가 무한 또는 0에 수렴할 경우 판단
            slope = rotate

        return slope

# 이미지 회전 함수
def RotateImage(preImage, angle):
    h, w = preImage.shape[:2]
    center = (w // 2, h // 2)
    M = cv2.getRotationMatrix2D(center, angle, 1.0)
    rotated_img = cv2.warpAffine(preImage, M, (w, h))

    return rotated_img, angle

def get_osd_orientation(preImage):
    # pytesseract로부터 반환된 텍스트 방향(OSD) 가져오기
    orientation = pytesseract.image_to_osd(preImage)
    print(orientation)

    # 문자열에서 각도 추출

    # 'Rotate: ' 문자열의 시작 위치를 찾아서 그 다음 문자열을 추출하기 위한 시작 인덱스 계산
    angle_start_idx = orientation.find('Rotate: ') + len('Rotate: ')

    # 'Rotate: ' 다음에 오는 첫 번째 줄 바꿈 문자('\n')의 위치를 찾아서 그 위치를 끝 인덱스로 설정
    angle_end_idx = orientation.find('\n', angle_start_idx)

    # 추출한 문자열을 정수로 변환하여 회전 각도로 설정
    rotation_angle = int(orientation[angle_start_idx:angle_end_idx])
    print(rotation_angle)

    return rotation_angle

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
    image_path = 'exam2.jpg'

    preImage = preprocess_image(image_path)
    rotation_angle = get_osd_orientation(preImage)

    is_horizontal = analyze_projection(preImage)
    print("가로입니까? : ",is_horizontal)

    PrintText(preImage, rotation_angle)
    # PrintText(preImage, is_horizontal)