#!/usr/bin/env python3
import sys
from typing import List

import cv2
import numpy as np

from image_processor import process_image_to_documents
from pdf_maker import create_single_page_pdf_from_images


def main():
    """
    Uso:
        python3 make_pdf.py salida.pdf foto1.jpg [foto2.jpg ...fotoN.jpg]

    Ejemplo:
        python3 make_pdf.py ./pdf/pdf_final19.pdf ./img/dos_tarjetas.jpeg
    """
    if len(sys.argv) < 3:
        print("Uso: python3 make_pdf.py salida.pdf img1 [img2 ...]")
        sys.exit(1)

    output_path = sys.argv[1]
    image_paths = sys.argv[2:]
    
    MAX_DOCS = 6

    processed_images: List[np.ndarray] = []

    for idx, path in enumerate(image_paths):
        # si ya tenemos 6 documentos, no procesamos más
        if len(processed_images) >= MAX_DOCS:
            break

        img = cv2.imread(path)
        if img is None:
            print(f"[WARN] No se pudo leer la imagen: {path}")
            continue

        docs_left = MAX_DOCS - len(processed_images)
        try:
            docs = process_image_to_documents(
                img,
                debug=False,          # ponelo en False si no querés imágenes de debug
                debug_prefix=f"img{idx+1}_",     # ← prefijo para no pisar archivos
                margin_ratio=0.06,
                rotate_portrait=True,
                max_docs=docs_left,
            )
        except ValueError as e:
            print(f"[WARN] {e} en la imagen: {path}")
            continue

        for d in docs:
            processed_images.append(d)
            if len(processed_images) >= MAX_DOCS:
                break

    if not processed_images:
        print("Error: no se pudo extraer ningún documento de las imágenes.")
        sys.exit(1)

    try:
        create_single_page_pdf_from_images(
            images_bgr=processed_images,
            output_path=output_path,
            dpi=300,
            outer_margin_mm=8.0,
            inner_margin_mm_x=8.0,
            inner_margin_mm_y=3.0,
            grid_rows=3,
            grid_cols=2,
            grid_height_fraction=0.7,
        )
    except Exception as e:
        print(f"Error al generar el PDF: {e}")
        sys.exit(1)

    print(f"PDF generado correctamente en: {output_path}")


if __name__ == "__main__":
    main()