import sys
import cv2

from image_processor import process_image_to_document
from pdf_maker import create_single_page_pdf_from_images


def main():
    if len(sys.argv) < 3:
        print("Uso: python make_pdf.py salida.pdf img1.jpg img2.jpg ...")
        sys.exit(1)

    output_pdf = sys.argv[1]
    input_images = sys.argv[2:]

    processed_images = []

    for path in input_images:
        img = cv2.imread(path)
        if img is None:
            print(f"No se pudo abrir la imagen: {path}")
            sys.exit(1)

        try:
            doc_img = process_image_to_document(img, debug=False)
        except ValueError as e:
            print(f"Error procesando '{path}': {e}")
            sys.exit(1)

        processed_images.append(doc_img)

    try:
        create_single_page_pdf_from_images(processed_images, output_pdf)
    except Exception as e:
        print(f"Error creando el PDF: {e}")
        sys.exit(1)

    print(f"PDF generado correctamente en: {output_pdf}")


if __name__ == "__main__":
    main()