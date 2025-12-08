# app.py

from typing import List

import numpy as np
import cv2
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.responses import Response

from image_processor import process_image_to_documents
from pdf_maker import create_single_page_pdf_bytes    

app = FastAPI(title="Image Processor API")


@app.get("/health")
def health():
    return {"status": "ok"}

MAX_DOCS = 6          # ← máximo de documentos por PDF
MAX_FILES = 4         # ← máximo de imágenes subidas (si querés lo podés cambiar)

@app.post("/dni-pdf")
async def dni_pdf(files: List[UploadFile] = File(...)):
    """
    Recibe entre 1 y 4 imágenes (JPG/PNG) y devuelve
    un PDF A4 de 1 hoja con hasta 6 documentos procesados.
    Puede detectar varios documentos en una misma foto.
    """
    if not files:
        raise HTTPException(status_code=400, detail="Debes enviar al menos una imagen")

    if len(files) > MAX_FILES:
        raise HTTPException(status_code=400, detail=f"Máximo {MAX_FILES} imágenes por PDF")

    processed_images: list[np.ndarray] = []

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
            raise HTTPException(
                status_code=400, detail=f"No se pudo leer la imagen {f.filename}"
            )

        # ¿cuántos documentos más podemos agregar?
        docs_left = MAX_DOCS - len(processed_images)
        if docs_left <= 0:
            break   # ya estamos en el límite: 5

        try:
            docs = process_image_to_documents(
                img,
                debug=False,           # en producción
                debug_prefix="",
                margin_ratio=0.06,
                rotate_portrait=True,
                max_docs=docs_left,
            )
        except ValueError as e:
            # no encontró documentos válidos en esta imagen
            continue

        for d in docs:
            processed_images.append(d)
            if len(processed_images) >= MAX_DOCS:
                break

        if len(processed_images) >= MAX_DOCS:
            break

    if not processed_images:
        raise HTTPException(status_code=422, detail="No se pudo extraer ningún documento")

    # DEBUG opcional: verificar cuántos docs realmente se van al PDF
    print(f"[API] Documentos que irán al PDF: {len(processed_images)}")

    try:
        pdf_bytes = create_single_page_pdf_bytes(
            images_bgr=processed_images,
            dpi=300,
            outer_margin_mm=8.0,
            inner_margin_mm_x=8.0,
            inner_margin_mm_y=1.0,
            grid_rows=3,   # 3 filas
            grid_cols=2,   # 2 columnas → hasta 6 slots, usamos máx. 6
            grid_height_fraction=0.8,
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