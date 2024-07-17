import cv2
import matplotlib.pyplot as plt
from PIL import Image
from collections import Counter
import pytesseract
import imutils
import numpy as np
import os

# 이미지 전처리
def preprocess_image(image):
    # image = imutils.resize(image, width=1960) # 해상도 높이기
    # image = cv2.resize(image, None, fx=2, fy=2, interpolation=cv2.INTER_CUBIC)

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) # RGB -> Gray
    blurred = cv2.GaussianBlur(gray, (5, 5), 0) # 흐림처리(노이즈 제거) : 높을수록 더 넓은 영역을 평균화하여 블러 효과가 강해질 수 있음
    edged = cv2.Canny(blurred, 100, 200) # 엣지 검출 (이미지, 최소강도, 최대강도) : 최소강도보다 작거나 최대강도보다 큰 경우 엣지로 간주하지 않음

    return edged

def rotate_image(image, angle):
    (h, w) = image.shape[:2]
    center = (w // 2, h // 2)
    M = cv2.getRotationMatrix2D(center, angle, 1.0)
    rotated = cv2.warpAffine(image, M, (w, h), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE)
    return rotated

def get_osd_orientation(edged, image):

    # pytesseract로부터 반환된 텍스트 방향(OSD) 가져오기
    orientation = pytesseract.image_to_osd(edged)
    print(orientation)

    # 문자열에서 각도 추출

    # 'Rotate: ' 문자열의 시작 위치를 찾아서 그 다음 문자열을 추출하기 위한 시작 인덱스 계산
    angle_start_idx = orientation.find('Rotate: ') + len('Rotate: ')

    # 'Rotate: ' 다음에 오는 첫 번째 줄 바꿈 문자('\n')의 위치를 찾아서 그 위치를 끝 인덱스로 설정
    angle_end_idx = orientation.find('\n', angle_start_idx)

    # 추출한 문자열을 정수로 변환하여 회전 각도로 설정
    rotation_angle = int(orientation[angle_start_idx:angle_end_idx])
    print(rotation_angle)

    # 이미지 회전
    rotated_image = rotate_image(image, -rotation_angle) # 시계방향 회전

    # 회전된 이미지를 다시 텍스트 인식
    rotated_text = pytesseract.image_to_string(rotated_image, lang='eng+kor+math')

    # 결과 출력
    print("Text:")
    print(rotated_text)

    # 회전된 이미지 저장
    # cv2.imwrite('rotated_image.jpg', rotated_image)

    # 최종 이미지 확인
    plt.figure(figsize=(10, 10))
    plt.imshow(cv2.cvtColor(rotated_image, cv2.COLOR_BGR2RGB))
    plt.axis('off')
    plt.show()

def find_text_contours(image):
    # Tesseract OCR를 사용하여 이미지에서 텍스트 인식
    custom_config = r'--oem 3 --psm 6'
    results = pytesseract.image_to_data(image, output_type=pytesseract.Output.DICT, config=custom_config, lang='eng+kor+math')
    
    text_contours = []
    # 각 텍스트 블록에 대해 반복
    for i in range(len(results['text'])):
        x = results['left'][i]    # 텍스트 블록의 왼쪽 상단 x 좌표
        y = results['top'][i]     # 텍스트 블록의 왼쪽 상단 y 좌표
        w = results['width'][i]   # 텍스트 블록의 너비
        h = results['height'][i]  # 텍스트 블록의 높이
        
        # 인식 신뢰도(confidence)가 0보다 큰 경우에만 처리
        if int(results['conf'][i]) > 0:
            roi = image[y:y+h, x:x+w]  # 이미지에서 텍스트 블록의 ROI 추출
            contours, _ = cv2.findContours(roi, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            # 각 외곽선의 좌표를 전체 이미지 좌표로 변환하여 리스트에 추가
            for cnt in contours:
                cnt[:, 0, 0] += x  # ROI에서 전체 이미지 좌표로 x 좌표 변환
                cnt[:, 0, 1] += y  # ROI에서 전체 이미지 좌표로 y 좌표 변환
                text_contours.append(cnt)
    
    return text_contours

def draw_rbox(image, contours):
    angles = []
    for cnt in contours:
        rect = cv2.minAreaRect(cnt)
        box = cv2.boxPoints(rect)
        box = np.int32(box)
        cv2.drawContours(image, [box], 0, (0, 255, 0), 2)

        angle = rect[-1]
        # print("개별 각도",angle)
        # if angle < 45:
        #     angle = 90 + angle
        angles.append(angle)

    if angles:
        selectAngle = compute_mean_of_most_common_tens(angles) * 10
        print("글자의 각도 : ", selectAngle)
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

def process_image(image_path):
    # 파일이 없을 시
    if not os.path.isfile(image_path):
        print(f"Error: {image_path} does not exist.")
        return
    
    # 이미지 Load
    image = cv2.imread(image_path, cv2.IMREAD_COLOR)

    # 이미지 전처리
    edged = preprocess_image(image)

    # 외곽선 찾기
    contours = find_text_contours(edged)

    # 글자들의 기울기들 중 분포도가 가장 높은 각도
    skew_angle = draw_rbox(image, contours)

    # osd 기능 : 스크립트 방향 인식
    get_osd_orientation(edged, image)

    print("end")

# 이미지 파일 경로 설정
image_path = 'exam3_2.jpg'
process_image(image_path)
