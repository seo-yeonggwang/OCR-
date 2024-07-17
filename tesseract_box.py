import cv2
import imutils
import matplotlib.pyplot as plt
from collections import Counter
import pytesseract
import numpy as np
import os

# 이미지 전처리
def preprocess_image(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (3, 3), 0)
    edged = cv2.Canny(blurred, 100, 200)
    
    return edged

def draw_rbox(image, contours):
    angles = []
    for cnt in contours:
        rect = cv2.minAreaRect(cnt)
        box = cv2.boxPoints(rect)
        box = np.int32(box)
        cv2.drawContours(image, [box], 0, (0, 255, 0), 2)

        angle = rect[-1]
        print(angle)
        if angle < 45:
            angle = 90 + angle
        angles.append(angle)

    if angles:
        selectAngle = compute_mean_of_most_common_tens(angles) * 10
        print(selectAngle)
        return selectAngle
    
    return 0  # 각도가 없는 경우 0을 반환

def compute_mean_of_most_common_tens(angles):
    # 각도 리스트에서 십의 자리를 Counter로 추출
    ten_counts = Counter([angle // 10 for angle in angles])
    
    # 가장 많이 나온 십의 자리 숫자들 추출
    most_common_tens = ten_counts.most_common()
    max_count = most_common_tens[0][1]
    
    # 가장 많이 나온 십의 자리 숫자들의 평균 계산
    total = 0
    count = 0
    for ten, cnt in most_common_tens:
        if cnt == max_count:
            total += ten
            count += 1
    
    if count > 0:
        return total / count
    else:
        return 0

def rotate_image(image, angle):
    (h, w) = image.shape[:2]
    center = (w // 2, h // 2)
    
    # 회전할 각도를 90에서 현재 각도를 뺀 값을 사용하여 회전시키도록 수정
    M = cv2.getRotationMatrix2D(center, 90 - angle, 1.0)
    rotated = cv2.warpAffine(image, M, (w, h), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE)
    
    return rotated


def find_text_contours(image):
    # Tesseract OCR를 사용하여 이미지에서 텍스트 인식
    custom_config = r'--oem 3 --psm 6'
    results = pytesseract.image_to_data(image, output_type=pytesseract.Output.DICT, config=custom_config, lang='eng+kor+math')
    
    text_contours = []
    for i in range(len(results['text'])):
        x = results['left'][i]
        y = results['top'][i]
        w = results['width'][i]
        h = results['height'][i]
        
        if int(results['conf'][i]) > 0:
            roi = image[y:y+h, x:x+w]
            contours, _ = cv2.findContours(roi, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            for cnt in contours:
                cnt[:, 0, 0] += x
                cnt[:, 0, 1] += y
                text_contours.append(cnt)
    
    return text_contours

def draw_text_contours(image, contours):
    cv2.drawContours(image, contours, -1, (0, 255, 0), 2)

    plt.figure(figsize=(10, 10))
    plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    plt.axis('off')
    plt.show()

def process_image(image_path):
    if not os.path.isfile(image_path):
        print(f"Error: {image_path} does not exist.")
        return
    
    # 이미지 Load
    image = cv2.imread(image_path, cv2.IMREAD_COLOR)

    # 이미지 전처리
    edged = preprocess_image(image)

    # 외곽선 찾기
    contours = find_text_contours(edged)

    skew_angle = draw_rbox(image, contours)

    rotated_image = rotate_image(image, skew_angle)

    plt.figure(figsize=(10, 10))
    plt.imshow(cv2.cvtColor(rotated_image, cv2.COLOR_BGR2RGB))
    plt.axis('off')
    plt.show()

    # edged_rotated = preprocess_image(rotated_image)
    print("end")

# 이미지 파일 경로 설정
image_path = 'test5.jpg'
process_image(image_path)