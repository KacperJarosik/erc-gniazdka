import cv2
import numpy as np

def find_and_draw_quadrilaterals(search_image_path, output_path):
    # Wczytanie obrazu do przeszukania
    search_image = cv2.imread(search_image_path)
    if search_image is None:
        print("Nie można wczytać obrazu. Sprawdź ścieżkę do pliku.")
        return

    # Konwersja obrazu na skalę szarości
    gray = cv2.cvtColor(search_image, cv2.COLOR_BGR2GRAY)

    # Wykrywanie krawędzi za pomocą algorytmu Canny'ego
    edges = cv2.Canny(gray, 50, 150)

    # Znajdowanie konturów
    contours, _ = cv2.findContours(edges, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    # Lista konturów, które są czworokątami
    quadrilaterals = []

    for cnt in contours:
        # Aproksymacja konturu
        epsilon = 0.02 * cv2.arcLength(cnt, True)
        approx = cv2.approxPolyDP(cnt, epsilon, True)

        # Sprawdzamy, czy kontur ma 4 wierzchołki i jest zamknięty
        if len(approx) == 4 and cv2.isContourConvex(approx):
            quadrilaterals.append(approx)

    # Rysowanie wykrytych czworokątów
    result_image = search_image.copy()
    for quad in quadrilaterals:
        cv2.polylines(result_image, [quad], True, (0, 255, 0), 3)

    # Zapisanie wyniku
    cv2.imwrite(output_path, result_image)
    print(f"Znaleziono czworokąty i zapisano obraz: {output_path}")

# Ścieżki do plików
search_image_path = "canny_63_64.png"  # Ścieżka do obrazu
output_path = "result_quadrilaterals.png"  # Wynikowy obraz z czworokątami

# Uruchomienie funkcji
find_and_draw_quadrilaterals(search_image_path, output_path)
