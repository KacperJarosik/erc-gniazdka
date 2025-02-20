import cv2
import numpy as np
import sys

def extract_roi(image, ids, corners, roi_ids, output_filename, scale_factor=4):
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

        # Ograniczenie x_max dla znaczników 63 i 64
        if 63 in roi_ids or 64 in roi_ids:
            max_left_edge = int(max(left_edges))  # Najbardziej oddalony w prawo lewy wierzchołek
            x_max = min(x_max, max_left_edge)

        # Ograniczenie y_max dla znacznika 52
        if 52 in roi_ids:
            min_top_edge = int(min(top_edges))  # Najniższy górny wierzchołek
            y_max = min(y_max, min_top_edge)

        # Debug współrzędnych po ograniczeniach
        print(f"Po ograniczeniach dla {roi_ids}: x_min={x_min}, x_max={x_max}, y_min={y_min}, y_max={y_max}")

        # Tworzenie maski
        mask = np.zeros_like(image)
        mask[y_min:y_max, x_min:x_max] = image[y_min:y_max, x_min:x_max]

        # Zapis maski
        cv2.imwrite(output_filename, mask)
        print(f"Maska zapisana: {output_filename}")
    except IndexError:
        print(f"ID {roi_ids} nie znaleziono.")



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

def main():
    # Stała ścieżka do obrazu i słownik ArUco
    image_path = "70v4.png"
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

    if ids is not None:
        ids = ids.flatten()

        # Wyświetlanie wykrytych znaczników
        detected_image = aruco_display(corners, ids, rejected, image.copy())
        cv2.imshow("Wykryte znaczniki ArUco", detected_image)
        cv2.imwrite(f"{output_directory}output_sample.png", detected_image)

        # Tworzenie masek dla ID 51, 52, 63 i 64
        mask_51_path = f"{output_directory}mask_51.png"
        mask_52_path = f"{output_directory}mask_52.png"
        mask_63_64_path = f"{output_directory}mask_63_64.png"

        extract_roi(image, ids, corners, [51], mask_51_path)
        extract_roi(image, ids, corners, [52], mask_52_path)
        extract_roi(image, ids, corners, [63, 64], mask_63_64_path)

        # Wczytanie masek dla ID 51 i 63_64
        mask_51 = cv2.imread(mask_51_path, cv2.IMREAD_GRAYSCALE)
        mask_52 = cv2.imread(mask_52_path, cv2.IMREAD_GRAYSCALE)
        mask_63_64 = cv2.imread(mask_63_64_path, cv2.IMREAD_GRAYSCALE)

        # Sprawdzenie rozmiarów
        if mask_63_64 is not None and mask_51 is not None:
            if mask_51.shape == mask_63_64.shape:
                # Część wspólna: gdzie maska 63_64 nie jest czarna, ustaw na czarno w masce 51
                overlap_mask = cv2.bitwise_and(mask_51, cv2.bitwise_not(mask_63_64))
                # Zapis nowej maski dla ID 51
                cv2.imwrite(mask_51_path, overlap_mask)
                print(f"Zaktualizowana maska 51 zapisana: {mask_51_path}")
                mask_51 = cv2.imread(mask_51_path, cv2.IMREAD_GRAYSCALE)
        # Algorytm Canny'ego na maskach
        canny_51 = cv2.Canny(mask_51, 100, 200)
        canny_52 = cv2.Canny(mask_52, 100, 200)
        canny_63_64 = cv2.Canny(mask_63_64, 100, 200)

        # Zapis wyników algorytmu Canny'ego
        if mask_51 is not None:
            cv2.imwrite(f"{output_directory}canny_51.png", canny_51)
            print(f"Maska Canny'ego dla 51 zapisana: {output_directory}canny_51.png")
        if mask_52 is not None:
            cv2.imwrite(f"{output_directory}canny_52.png", canny_52)
            print(f"Maska Canny'ego dla 52 zapisana: {output_directory}canny_52.png")
        if mask_63_64 is not None:
            cv2.imwrite(f"{output_directory}canny_63_64.png", canny_63_64)
            print(f"Maska Canny'ego dla 63_64 zapisana: {output_directory}canny_63_64.png")

        cv2.waitKey(0)
    else:
        print("Nie wykryto żadnych znaczników ArUco.")
        sys.exit(0)


if __name__ == "__main__":
    main()



