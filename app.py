# app.py

from typing import List

import numpy as np
import cv2
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.responses import Response

from image_processor import process_image_to_document
from pdf_maker import create_single_page_pdf_bytes

app = FastAPI(title="Image Processor API")


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/dni-pdf")
async def dni_pdf(files: List[UploadFile] = File(...)):
    """
    Recibe entre 1 y 4 imágenes (JPG/PNG) y devuelve
    un PDF A4 de 1 hoja con todas las imágenes procesadas.
    """
    if not files:
        raise HTTPException(status_code=400, detail="Debes enviar al menos una imagen")

    if len(files) > 4:
        raise HTTPException(status_code=400, detail="Máximo 4 imágenes por PDF")

    processed_images = []

    for f in files:
        if f.content_type not in ("image/jpeg", "image/png", "image/jpg"):
            raise HTTPException(
                status_code=400,
                detail=f"Tipo de archivo no soportado: {f.content_type}",
            )

        data = await f.read()
        nparr = np.frombuffer(data, np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

        if img is None:
            raise HTTPException(status_code=400, detail=f"No se pudo leer la imagen {f.filename}")

        try:
            doc_img = process_image_to_document(
                img,
                debug=False,
                margin_ratio=0.06,
                rotate_portrait=True,
                enhance_mode="soft",  # o "hard" si querés blanco/negro tipo fotocopia
            )
        except ValueError as e:
            raise HTTPException(status_code=422, detail=str(e))

        processed_images.append(doc_img)

    try:
        pdf_bytes = create_single_page_pdf_bytes(
            images_bgr=processed_images,
            dpi=300,
            outer_margin_mm=8.0,
            inner_margin_mm_x=8.0,
            inner_margin_mm_y=3.0,
            grid_rows=2,
            grid_cols=2,
            grid_height_fraction=0.7,
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error al generar el PDF: {e}")

    return Response(
        content=pdf_bytes,
        media_type="application/pdf",
        headers={
            "Content-Disposition": 'inline; filename="dni_documentos.pdf"'
        },
    )