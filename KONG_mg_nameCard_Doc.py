import cv2
import numpy as np

path = "C:/KONG_2027Prg/data/3dvision/"
img_path = path + "card.jpg"
# C:\KONG_2027Prg\data\3dvision\card.jpg
# --------------------------
# 4개의 점을 (tl, tr, br, bl) 순서로 정렬 (y기준 + x기준)
# --------------------------
def order_points(pts):
    # pts: shape (4,2)
    pts = np.array(pts, dtype="float32")

    # 1) y(세로) 기준으로 정렬 → 위쪽 2점, 아래쪽 2점
    y_sorted = pts[np.argsort(pts[:, 1]), :]   # 작은 y가 위쪽
    top = y_sorted[:2, :]
    bottom = y_sorted[2:, :]

    # 2) 위쪽 2점은 x 기준 (왼쪽/오른쪽)
    top = top[np.argsort(top[:, 0]), :]
    tl, tr = top

    # 3) 아래쪽 2점도 x 기준 (왼쪽/오른쪽)
    bottom = bottom[np.argsort(bottom[:, 0]), :]
    bl, br = bottom

    # 4) (tl, tr, br, bl) 순서로 반환
    rect = np.array([tl, tr, br, bl], dtype="float32")
    return rect


def main():
    image = cv2.imread(img_path)
    if image is None:
        print("이미지 로드 실패:", img_path)
        return

    orig = image.copy()
    h, w = image.shape[:2]
    img_area = h * w

    # 1) Gray + 블러
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (5, 5), 0)

    # 2) Otsu 이진화 (명함은 어둡고 배경은 밝음 → 반전해서 명함을 흰색으로)
    _, th_inv = cv2.threshold(gray, 0, 255,
                              cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

    # 3) 모폴로지 Close 로 구멍 메꾸기
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (15, 15))
    closed = cv2.morphologyEx(th_inv, cv2.MORPH_CLOSE, kernel, iterations=2)

    # 4) 외곽(바깥 컨투어) 찾기
    contours, _ = cv2.findContours(
        closed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )

    if not contours:
        print("컨투어를 찾지 못했습니다.")
        return

    # 5) 면적 큰 순으로 정렬
    contours = sorted(contours, key=cv2.contourArea, reverse=True)

    card_cnt = None
    for c in contours:
        area = cv2.contourArea(c)
        if area < img_area * 0.05:  # 너무 작은 것은 무시
            continue

        peri = cv2.arcLength(c, True)
        approx = cv2.approxPolyDP(c, 0.02 * peri, True)

        if len(approx) == 4:
            card_cnt = approx
            break

    # 4점 근사 실패 시: 가장 큰 컨투어에 대해 minAreaRect 사용
    if card_cnt is None:
        c = contours[0]
        rect_min = cv2.minAreaRect(c)
        box = cv2.boxPoints(rect_min)
        card_cnt = box.reshape(-1, 1, 2).astype("float32")

    # 6) 4점 정리 & 순서 맞추기
    pts = card_cnt.reshape(4, 2)
    rect = order_points(pts)
    (tl, tr, br, bl) = rect

    # 7) 가로, 세로 길이 계산
    widthA = np.linalg.norm(br - bl)
    widthB = np.linalg.norm(tr - tl)
    maxWidth = int(max(widthA, widthB))

    heightA = np.linalg.norm(tr - br)
    heightB = np.linalg.norm(tl - bl)
    maxHeight = int(max(heightA, heightB))

    dst = np.array([
        [0, 0],
        [maxWidth - 1, 0],
        [maxWidth - 1, maxHeight - 1],
        [0, maxHeight - 1]
    ], dtype="float32")

    # 8) 투시 변환
    M = cv2.getPerspectiveTransform(rect, dst)
    warped = cv2.warpPerspective(orig, M, (maxWidth, maxHeight))

    # 9) 디버그용 이미지
    debug = image.copy()
    cv2.drawContours(debug, [card_cnt.astype(int)], -1, (0, 255, 0), 3)

    cv2.imshow("Threshold INV", th_inv)
    cv2.imshow("Closed", closed)
    cv2.imshow("Detected Card Contour", debug)
    cv2.imshow("Warped (Card)", warped)

    cv2.imwrite(path + "card_scanned_full_yorder.jpg", warped)

    cv2.waitKey(0)
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
