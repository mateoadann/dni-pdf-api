import sys
import cv2
from image_processor import process_image_to_document


def main():
    if len(sys.argv) < 3:
        print("Uso: python process_dni.py input.jpg output.jpg [--debug]")
        sys.exit(1)

    input_path = sys.argv[1]
    output_path = sys.argv[2]
    debug = "--debug" in sys.argv

    image = cv2.imread(input_path)
    if image is None:
        print(f"No se pudo abrir la imagen de entrada: {input_path}")
        sys.exit(1)

    try:
        processed = process_image_to_document(image, debug=debug)
    except ValueError as e:
        print(f"Error procesando la imagen: {e}")
        sys.exit(1)

    ok = cv2.imwrite(output_path, processed)
    if not ok:
        print(f"No se pudo guardar la imagen de salida: {output_path}")
        sys.exit(1)

    print(f"Imagen procesada guardada en: {output_path}")


if __name__ == "__main__":
    main()