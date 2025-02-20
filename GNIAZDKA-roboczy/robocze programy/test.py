import cv2
import numpy as np
import sys
import psutil
import GPUtil
import time

def log_performance(start_time):
    """
    Loguje zajętość pamięci, użycie GPU i czas działania.
    """
    # Czas działania
    elapsed_time = time.time() - start_time

    # Zajętość pamięci
    memory_info = psutil.virtual_memory()
    total_memory = memory_info.total / (1024 ** 3)  # GB
    used_memory = memory_info.used / (1024 ** 3)  # GB
    memory_percent = memory_info.percent

    # GPU usage
    gpus = GPUtil.getGPUs()
    if gpus:
        gpu_info = [(gpu.name, gpu.load * 100, gpu.memoryUsed, gpu.memoryTotal) for gpu in gpus]
    else:
        gpu_info = [("No GPU detected", 0, 0, 0)]

    # Logowanie
    print(f"\n--- Performance Metrics ---")
    print(f"Elapsed Time: {elapsed_time:.2f} seconds")
    print(f"Memory Usage: {used_memory:.2f} GB / {total_memory:.2f} GB ({memory_percent}%)")
    for idx, (name, load, mem_used, mem_total) in enumerate(gpu_info):
        print(f"GPU {idx}: {name}, Load: {load:.2f}%, Memory: {mem_used:.2f} MB / {mem_total:.2f} MB")
    print(f"---------------------------\n")

def main():
    # Start timer
    start_time = time.time()

    # Stała ścieżka do obrazu i słownik ArUco
    image_path = "70.jpg"
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
        cv2.imwrite(f"{output_directory}output_sample.png", detected_image)

        # Tworzenie masek dla ID 51, 52, 63 i 64
        mask_51_path = f"{output_directory}mask_51.png"
        mask_52_path = f"{output_directory}mask_52.png"
        mask_63_64_path = f"{output_directory}mask_63_64.png"

        extract_roi(image, ids, corners, [51], mask_51_path)
        extract_roi(image, ids, corners, [52], mask_52_path)
        extract_roi(image, ids, corners, [63, 64], mask_63_64_path)

    else:
        print("Nie wykryto żadnych znaczników ArUco.")
        sys.exit(0)

    # Logowanie wydajności
    log_performance(start_time)

if __name__ == "__main__":
    main()