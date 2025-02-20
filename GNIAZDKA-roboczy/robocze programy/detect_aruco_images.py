import cv2
import numpy as np
import sys
from utils import ARUCO_DICT, aruco_display

# Wczytaj obraz wejściowy
print("Loading image...")
image_path = "TablicaERC/WIN_20240628_15_31_42_Pro.jpg"
image = cv2.imread(image_path)
h, w, _ = image.shape

# Skaluje obraz dla lepszej widoczności
width = 600
height = int(width * (h / w))
image = cv2.resize(image, (width, height), interpolation=cv2.INTER_CUBIC)

# Weryfikacja typu znacznika
aruco_type = "DICT_6X6_100"
if ARUCO_DICT.get(aruco_type, None) is None:
    print(f"ArUCo tag type '{aruco_type}' is not supported")
    sys.exit(0)

# Ładowanie słownika i parametrów detekcji
print(f"Detecting '{aruco_type}' tags...")
aruco_dict = cv2.aruco.getPredefinedDictionary(ARUCO_DICT[aruco_type])
aruco_params = cv2.aruco.DetectorParameters()

# Detekcja znaczników
corners, ids, rejected = cv2.aruco.detectMarkers(image, aruco_dict, parameters=aruco_params)

# Wyświetlanie i zapisywanie obrazu z wykrytymi znacznikami
if ids is not None:
    print(f"Detected markers: {ids.flatten()}")
else:
    print("No markers detected.")

# Wyświetlanie wykrytych znaczników na obrazie
detected_markers = aruco_display(corners, ids, rejected, image)

# Konwersja do skali szarości
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
cv2.imwrite("output_gray.png", gray)

# Binaryzacja obrazu (thresholding)
_, binary = cv2.threshold(gray, 175, 255, cv2.THRESH_BINARY)
cv2.imwrite("output_binary.png", binary)

# Perspektywa usunięta
def remove_perspective(image, corners):
    if corners is not None and len(corners) > 0:
        for corner in corners:
            pts1 = np.float32(corner[0])
            size = 100
            pts2 = np.float32([[0, 0], [size, 0], [size, size], [0, size]])
            matrix = cv2.getPerspectiveTransform(pts1, pts2)
            dst = cv2.warpPerspective(image, matrix, (size, size))
            cv2.imwrite("output_perspective_removed.png", dst)
remove_perspective(image, corners)

# Rysowanie marginesów kodu Aruco
if ids is not None:
    margin_image = image.copy()
    for corner in corners:
        for point in corner[0]:
            cv2.circle(margin_image, tuple(point.astype(int)), 5, (0, 255, 0), -1)
    cv2.imwrite("output_marker_margins.png", margin_image)

# Zapis obrazów
output_detected = "output_detected.png"
output_rejected = "output_rejected.png"
cv2.imwrite(output_detected, detected_markers)

# Opcjonalne zapisywanie odrzuconych kandydatów
if rejected:
    rejected_image = image.copy()
    cv2.aruco.drawDetectedMarkers(rejected_image, rejected, None, (100, 0, 255))
    cv2.imwrite(output_rejected, rejected_image)

cv2.imshow("Detected Markers", detected_markers)
cv2.waitKey(0)
cv2.destroyAllWindows()

print(f"Images saved: {output_detected}, {output_rejected if rejected else 'No rejected markers.'}, output_binary.png, output_perspective_removed.png, output_marker_margins.png")
