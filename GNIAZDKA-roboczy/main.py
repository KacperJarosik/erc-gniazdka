import cv2
import numpy as np
import sys

def extract_roi(image, ids, corners, roi_ids, scale_factor=4):
    """
    Funkcja wycina obszar wokół znaczników ArUco, wypełnia czarne tło
    i zapisuje jako plik o oryginalnym rozmiarze obrazu.
    """
    try:
        all_corners = []
        left_edges = []
        top_edges = []

        for roi_id in roi_ids:
            # Znalezienie indeksu dla ID
            if roi_id in ids:
                index = np.where(ids == roi_id)[0][0]
                c = corners[index][0]
                all_corners.append(c)

                left_edges.append(np.min(c[:, 0]))  # Najbardziej lewy wierzchołek
                top_edges.append(np.min(c[:, 1]))  # Najwyższy wierzchołek

        if not all_corners:
            print(f"Nie znaleziono znaczników dla ID: {roi_ids}")
            return

        all_corners = np.concatenate(all_corners, axis=0)
        x_min = int(np.min(all_corners[:, 0]))
        x_max = int(np.max(all_corners[:, 0]))
        y_min = int(np.min(all_corners[:, 1]))
        y_max = int(np.max(all_corners[:, 1]))

        # Obliczenie marginesów
        width_margin = int((x_max - x_min) * scale_factor)
        height_margin = int((y_max - y_min) * scale_factor)
        x_min = max(x_min - width_margin, 0)
        y_min = max(y_min - height_margin, 0)
        x_max = min(x_max + width_margin, image.shape[1])
        y_max = min(y_max + height_margin, image.shape[0])

        # Debug współrzędnych
        print(f"Przed ograniczeniami dla {roi_ids}: x_min={x_min}, x_max={x_max}, y_min={y_min}, y_max={y_max}")

        # Wierzchołki narożników
        top_left = c[0]
        top_right = c[1]
        bottom_right = c[2]
        bottom_left = c[3]

        # Obliczanie szerokości i wysokości znaczników
        marker_width = int(np.linalg.norm(top_right - top_left))
        marker_height = int(np.linalg.norm(top_left - bottom_left))

        # Ograniczenie x_max dla znaczników 63 i 64
        if roi_id == 63:
            # Ustawienia dla znacznika o ID 63
            x_min = int(top_left[0] - 4 * marker_width)  # Minimalna wartość x z uwzględnieniem przesunięcia
            x_max = int(top_left[0])  # Maksymalna wartość x dla obszaru
            y_min = int(top_left[1] - 2 * marker_height)  # Minimalna wartość y z uwzględnieniem przesunięcia
            y_max = int(bottom_left[1] + 5 * marker_height)  # Maksymalna wartość y dla obszaru
        elif roi_id == 64:
            # Ustawienia dla znacznika o ID 64
            x_min = int(top_left[0] - 4 * marker_width)  # Minimalna wartość x z uwzględnieniem przesunięcia
            x_max = int(top_left[0])  # Maksymalna wartość x dla obszaru
            y_min = int(top_left[1] - 5 * marker_height)  # Minimalna wartość y z uwzględnieniem przesunięcia
            y_max = int(bottom_left[1] + 2 * marker_height)  # Maksymalna wartość y dla obszaru
        if roi_id == 51:
            x_min = int(bottom_left[0] - 1 * marker_width)  # Lewy przesunięty o 1 szerokość
            x_max = int(bottom_right[0] + 3 * marker_width)  # Prawy przesunięty o 3 szerokości
            y_min = int(top_left[1] - 1 * marker_height)  # Górny przesunięty o 1 wysokość
            y_max = int(bottom_left[1] + 4 * marker_height)  # Dolny przesunięty o 4 wysokości
            
            #Dodaj wycięcie lewego górnego kwadratu powyżej y_max i mniejsze niż x_max dla roi_id == 51:
            
        # Ograniczenie y_max dla znacznika 52
        if 52 in roi_ids:
            # Obliczenie przesunięć
            x_min = int(bottom_left[0] - marker_width)  # Lewy przesunięty o 1 szerokość znacznika
            x_max = int(bottom_right[0] + marker_width)  # Prawy przesunięty o 1 szerokość znacznika
            y_min = int(top_left[1] - 2.5 * marker_height)  # Górny przesunięty o 2.5 wysokości znacznika
            min_top_edge = int(min(top_edges))  # Najniższy górny wierzchołek
            y_max = min(y_max, min_top_edge)

        # Debug współrzędnych po ograniczeniach
        print(f"Po ograniczeniach dla {roi_ids}: x_min={x_min}, x_max={x_max}, y_min={y_min}, y_max={y_max}")

        # Tworzenie maski
        mask = np.zeros_like(image)
        mask[y_min:y_max, x_min:x_max] = image[y_min:y_max, x_min:x_max]
        if roi_id == 51:
            # Wyznaczenie granic kodu ArUco
            aruco_right_edge = int(top_right[0])     # Prawa krawędź znacznika
            aruco_bottom_edge = int(bottom_left[1])  # Dolna krawędź znacznika

            # Wyzerowanie lewego górnego rogu (powyżej dolnej krawędzi znacznika i przed jego prawą krawędzią)
            mask[0:aruco_bottom_edge, 0:aruco_right_edge] = 0
        return mask
    except IndexError:
        print(f"ID {roi_ids} nie znaleziono.")
        return None



def aruco_display(corners, ids, rejected, image):
    """
    Funkcja rysuje wykryte znaczniki ArUco na obrazie.
    """
    if len(corners) > 0:
        for markerCorner, markerID in zip(corners, ids.flatten()):
            corners = markerCorner.reshape((4, 2))
            (topLeft, topRight, bottomRight, bottomLeft) = corners
            # Zamiana na integer
            topLeft = (int(topLeft[0]), int(topLeft[1]))
            topRight = (int(topRight[0]), int(topRight[1]))
            bottomRight = (int(bottomRight[0]), int(bottomRight[1]))
            bottomLeft = (int(bottomLeft[0]), int(bottomLeft[1]))

            # Rysowanie linii wokół znacznika
            cv2.line(image, topLeft, topRight, (0, 255, 0), 2)
            cv2.line(image, topRight, bottomRight, (0, 255, 0), 2)
            cv2.line(image, bottomRight, bottomLeft, (0, 255, 0), 2)
            cv2.line(image, bottomLeft, topLeft, (0, 255, 0), 2)

            # Dodanie ID znacznika
            cv2.putText(image, str(markerID), (topLeft[0], topLeft[1] - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

    return image


def filter_thin_objects(binary_image, min_contour_area=50):
    """
    Usuwa cienkie obiekty z obrazu binarnego.
    - binary_image: Obraz binarny (np. wynik progowania)
    - min_contour_area: Minimalna powierzchnia konturu do zachowania
    """
    # Znajdź kontury
    contours, _ = cv2.findContours(binary_image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    # Utwórz nowy obraz wynikowy
    filtered_image = np.zeros_like(binary_image)

    for contour in contours:
        # Sprawdź powierzchnię konturu
        if cv2.contourArea(contour) >= min_contour_area:
            # Zachowaj tylko wystarczająco duże obiekty
            cv2.drawContours(filtered_image, [contour], -1, (255), thickness=cv2.FILLED)

    return filtered_image
# Funkcja do nałożenia maski na obraz w określonym kolorze
def apply_mask_with_color(base_image, mask, color):
    """
    Nakłada maskę na obraz bazowy w określonym kolorze.
    - base_image: Obraz bazowy (w kolorze BGR)
    - mask: Maska (w skali szarości)
    - color: Kolor (BGR) w formacie tuple, np. (255, 0, 0) dla niebieskiego
    """
    # Upewnij się, że obraz wejściowy jest trójwymiarowy (BGR)
    if len(base_image.shape) == 2:  # Jeśli obraz jest w odcieniach szarości (2D)
        base_image = cv2.cvtColor(base_image, cv2.COLOR_GRAY2BGR)  # Konwertuj na BGR

    # Tworzenie kolorowej maski na podstawie koloru
    overlay = np.zeros_like(base_image, dtype=np.uint8)
    for i in range(3):  # Dla każdego kanału BGR
        overlay[:, :, i] = mask * (color[i] / 255.0)

    # Połączenie obrazu bazowego z maską (przezroczystość = 1)
    return cv2.addWeighted(base_image, 1, overlay, 1, 0)


if __name__ == "__main__":
        # Stała ścieżka do obrazu i słownik ArUco
    image_path = "robocze zdjecia/70.jpg"
    output_directory = "./"  # Katalog docelowy dla wynikowych plików
    aruco_type = "DICT_ARUCO_ORIGINAL"

    print("Wczytywanie obrazu...")
    image = cv2.imread(image_path)
    if image is None:
        print("Błąd wczytywania obrazu.")
        sys.exit(0)

    # Detekcja znaczników ArUco
    print(f"Wykrywanie znaczników typu '{aruco_type}'...")
    arucoDict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_ARUCO_ORIGINAL)
    arucoParams = cv2.aruco.DetectorParameters()
    corners, ids, rejected = cv2.aruco.detectMarkers(image, arucoDict, parameters=arucoParams)

    if corners:
        widths = []
        for marker_corners in corners:
            # marker_corners ma kształt (1, 4, 2) -> [Top-Left, Top-Right, Bottom-Right, Bottom-Left]
            pts = marker_corners[0]
            
            # Oblicz szerokość jako średnią odległość między górnymi i dolnymi bokami
            top_width = np.linalg.norm(pts[0] - pts[1])   # Top-Left ↔ Top-Right
            bottom_width = np.linalg.norm(pts[3] - pts[2]) # Bottom-Left ↔ Bottom-Right
            avg_width = (top_width + bottom_width) / 2
            widths.append(avg_width)
        
        # Średnia szerokość wszystkich znaczników
        mean_width = np.mean(widths)
        print(f"Średnia szerokość znaczników ArUco: {mean_width:.2f} pikseli")
    else:
        print("Nie wykryto znaczników ArUco.")

    if ids is not None:
        ids = ids.flatten()

        # Wyświetlanie wykrytych znaczników
        detected_image = aruco_display(corners, ids, rejected, image.copy())
        cv2.imwrite(f"{output_directory}output_sample.png", detected_image)

        # Tworzenie masek dla ID 51, 52, 63 i 64
        mask_51 = extract_roi(image, ids, corners, [51])
        mask_52 = extract_roi(image, ids, corners, [52])
        mask_63_64 = extract_roi(image, ids, corners, [63, 64])

        mask_51 = cv2.cvtColor(mask_51, cv2.COLOR_BGR2GRAY)
        mask_52 = cv2.cvtColor(mask_52, cv2.COLOR_BGR2GRAY)
        mask_63_64 = cv2.cvtColor(mask_63_64, cv2.COLOR_BGR2GRAY)

        # Sprawdzenie rozmiarów
        if mask_63_64 is not None and mask_51 is not None:
            if mask_51.shape == mask_63_64.shape:
                # Część wspólna: gdzie maska 63_64 nie jest czarna, ustaw na czarno w masce 51
                mask_51 = cv2.bitwise_and(mask_51, cv2.bitwise_not(mask_63_64))

            if mask_51.shape == mask_52.shape:
                # Część wspólna: gdzie maska 51 nie jest czarna, ustaw na czarno w masce 52
                mask_52 = cv2.bitwise_and(mask_52, cv2.bitwise_not(mask_51))

        # Progowanie dla białych gniazdek

        thresh = 183  # Dolny próg dla wartości pikseli
        maxval = 255  # Maksymalna wartość do ustawienia dla pikseli spełniających próg
        _, mask_51_thresh = cv2.threshold(mask_51, thresh, maxval, cv2.THRESH_BINARY)
        _, mask_52_thresh = cv2.threshold(mask_52, thresh, maxval, cv2.THRESH_BINARY)
        _, mask_63_64_thresh = cv2.threshold(mask_63_64, thresh, maxval, cv2.THRESH_BINARY)

        filtered_51 = None
        filtered_52 = None
        filtered_63_64 = None
        # Filtrowanie cienkich obiektów na progowanych maskach
        if mask_51_thresh is not None:
            filtered_51 = filter_thin_objects(mask_51_thresh, min_contour_area=1000)

        if mask_52_thresh is not None:
            filtered_52 = filter_thin_objects(mask_52_thresh, min_contour_area=1000)

        if mask_52_thresh is not None:
            filtered_63_64 = filter_thin_objects(mask_63_64_thresh, min_contour_area=1000)

        # Algorytm Canny'ego na progowanych maskach
        canny_51 = cv2.Canny(filtered_51, 100, 200)
        canny_52 = cv2.Canny(filtered_52, 100, 200)
        canny_63_64 = cv2.Canny(filtered_63_64, 100, 200)

        # Dylatacja, aby pogrubić linie krawędzi
        kernel = np.ones((9, 9), np.uint8)  # Tworzymy 9x9 kernel do dylatacji i erozji
        
        iteration_number = int(mean_width)//4
        canny_51_dilated = cv2.dilate(canny_51, kernel, iterations=iteration_number)  # Pogrubienie krawędzi
        canny_52_dilated = cv2.dilate(canny_52, kernel, iterations=iteration_number)  # Pogrubienie krawędzi
        canny_63_64_dilated = cv2.dilate(canny_63_64, kernel, iterations=iteration_number)  # Pogrubienie krawędzi

        # # Przywrócenie do oryginalnego rozmiaru (erozja)
        canny_51_dilated = cv2.erode(canny_51_dilated, kernel, iterations=iteration_number)
        canny_52_dilated = cv2.erode(canny_52_dilated, kernel, iterations=iteration_number)
        canny_63_64_dilated = cv2.erode(canny_63_64_dilated, kernel, iterations=iteration_number)
  
    output_directory = "./"  # Katalog docelowy dla wynikowych plików
    # Nakładanie masek z różnymi kolorami
    output_image = cv2.imread('output_sample.png')
    if canny_51_dilated is not None:
        output_image = apply_mask_with_color(output_image, canny_51_dilated, (255, 0, 0))  # Niebieski dla maski 51
    if canny_52_dilated is not None:
        output_image = apply_mask_with_color(output_image, canny_52_dilated, (0, 255, 0))  # Zielony dla maski 52
    if canny_63_64_dilated is not None:
        output_image = apply_mask_with_color(output_image, canny_63_64_dilated, (0, 0, 255))  # Czerwony dla maski 63 i 64
    cv2.imwrite(f"{output_directory}output_sample.png", output_image)
    cv2.imshow("Wykryte znaczniki ArUco", output_image)
    cv2.waitKey(0)


