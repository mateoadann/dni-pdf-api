# DNI Processor API

Sistema para procesar fotos de DNI / Documento tipo tarjeta y generar un PDF A4 listo para imprimir.

Este proyecto nació para resolver un problema concreto en un comercio:

- El cliente envía fotos del DNI o tarjeta (frente/dorso).
- Las imágenes vienen con fondo, sombras, mala perspectiva, etc.
- Persona necesita imprimirlas en una hoja A4 de forma ordenada.

Con esta API se puede:

1. Recortar automáticamente el documento (DNI / tarjeta) de la foto.
2. Corregir la perspectiva (como si fuera escaneado).
3. Mejorar la legibilidad (contraste / reducción de sombras).
4. Rotar para que quede en orientación vertical (portrait).
5. Armar una hoja A4 con hasta 4 imágenes (2 columnas x 2 filas).
6. Devolver un único PDF de 1 página.

---

## Estructura del proyecto

```text
dni_processor/
├─ app.py                # API FastAPI
├─ image_processor.py    # Lógica de procesamiento de imágenes (OpenCV)
├─ pdf_maker.py          # Composición de la hoja A4 y PDF (Pillow)
├─ make_pdf.py           # Script CLI para pruebas manuales
├─ requirements.txt      # Dependencias de Python
├─ Dockerfile            # Imagen Docker para deploy (ej. EasyPanel)
└─ README.md
