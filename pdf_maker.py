import io
from typing import List

import numpy as np
from PIL import Image


def _build_page_image(
    images_bgr: List[np.ndarray],
    dpi: int = 300,
    outer_margin_mm: float = 8.0,
    inner_margin_mm_x: float = 8.0,
    inner_margin_mm_y: float = 1.0,
    grid_rows: int = 3,
    grid_cols: int = 2,
    grid_height_fraction: float = 0.8,  # ya no lo usamos directamente, pero lo dejamos en la firma
) -> Image.Image:
    """
    Devuelve una imagen PIL con la hoja A4 armada.

    IMPORTANTE:
    - La altura de cada tarjeta se calcula en base al ANCHO disponible y a una
    relación de aspecto típica de tarjeta (width/height ~ 1.6).
    - Así se reduce muchísimo el espacio vertical entre filas.
    """

    if not images_bgr:
        raise ValueError("No se recibieron imágenes para generar el PDF")

    max_images = grid_rows * grid_cols  # ahora = 6
    if len(images_bgr) > max_images:
        raise ValueError(
            f"Se recibieron {len(images_bgr)} imágenes y el máximo permitido es "
            f"{max_images} para una grilla {grid_rows}x{grid_cols}"
        )

    # ---- Tamaño A4 en píxeles ----
    a4_width_mm, a4_height_mm = 210, 297
    a4_width_in = a4_width_mm / 25.4
    a4_height_in = a4_height_mm / 25.4

    page_width_px = int(a4_width_in * dpi)
    page_height_px = int(a4_height_in * dpi)

    # ---- Márgenes en píxeles ----
    outer_margin_px = int((outer_margin_mm / 25.4) * dpi)
    inner_margin_px_x = int((inner_margin_mm_x / 25.4) * dpi)
    inner_margin_px_y = int((inner_margin_mm_y / 25.4) * dpi)

    # ---- Área útil para la grilla ----
    content_width_px = page_width_px - 2 * outer_margin_px

    # ancho de cada celda (2 columnas)
    cell_width = (content_width_px - inner_margin_px_x * (grid_cols - 1)) // grid_cols
    if cell_width <= 0:
        raise ValueError("No hay espacio horizontal para las imágenes")

    # Relación de aspecto típica de tarjeta: ancho/alto ~ 1.6
    card_aspect_ratio = 1.6
    card_height_px = int(cell_width / card_aspect_ratio)

    # altura total de la grilla (solo celdas + espacio entre filas)
    grid_height_px = grid_rows * card_height_px + inner_margin_px_y * (grid_rows - 1)

    # Centramos la grilla en vertical dentro de la hoja
    # (si querés más arriba, podés sumar un offset pequeño)
    grid_start_y = outer_margin_px
    
    page = Image.new("RGB", (page_width_px, page_height_px), color=(255, 255, 255))

    # ---- Pegamos cada imagen ----
    for idx, img_bgr in enumerate(images_bgr):
        # conversión BGR -> RGB
        img_rgb = img_bgr[:, :, ::-1]
        pil_img = Image.fromarray(img_rgb)

        w, h = pil_img.size

        # Queremos que la imagen "llene" la celda en ancho, y en altura
        # la limitamos por card_height_px.
        scale = min(cell_width / w, card_height_px / h)
        new_w = int(w * scale)
        new_h = int(h * scale)
        pil_img = pil_img.resize((new_w, new_h), Image.LANCZOS)

        row = idx // grid_cols
        col = idx % grid_cols

        # Coordenadas de la esquina sup-izq. de la celda
        cell_x0 = outer_margin_px + col * (cell_width + inner_margin_px_x)
        cell_y0 = grid_start_y + row * (card_height_px + inner_margin_px_y)

        # Centramos la tarjeta dentro de la celda
        x = cell_x0 + (cell_width - new_w) // 2
        y = cell_y0 + (card_height_px - new_h) // 2

        page.paste(pil_img, (x, y))

    return page


# ---------- Versión que guarda a archivo (CLI / make_pdf.py) ----------

def create_single_page_pdf_from_images(
    images_bgr: List[np.ndarray],
    output_path: str,
    dpi: int = 300,
    outer_margin_mm: float = 8.0,
    inner_margin_mm_x: float = 8.0,
    inner_margin_mm_y: float = 1.0,
    grid_rows: int = 3,
    grid_cols: int = 2,
    grid_height_fraction: float = 0.8,
) -> None:
    page = _build_page_image(
        images_bgr=images_bgr,
        dpi=dpi,
        outer_margin_mm=outer_margin_mm,
        inner_margin_mm_x=inner_margin_mm_x,
        inner_margin_mm_y=inner_margin_mm_y,
        grid_rows=grid_rows,
        grid_cols=grid_cols,
        grid_height_fraction=grid_height_fraction,
    )
    page.save(output_path, "PDF", resolution=dpi)


# ---------- Versión que devuelve bytes (API FastAPI) ----------

def create_single_page_pdf_bytes(
    images_bgr: List[np.ndarray],
    dpi: int = 300,
    outer_margin_mm: float = 8.0,
    inner_margin_mm_x: float = 8.0,
    inner_margin_mm_y: float = 1.0,
    grid_rows: int = 3,
    grid_cols: int = 2,
    grid_height_fraction: float = 0.8,
) -> bytes:
    page = _build_page_image(
        images_bgr=images_bgr,
        dpi=dpi,
        outer_margin_mm=outer_margin_mm,
        inner_margin_mm_x=inner_margin_mm_x,
        inner_margin_mm_y=inner_margin_mm_y,
        grid_rows=grid_rows,
        grid_cols=grid_cols,
        grid_height_fraction=grid_height_fraction,
    )

    buf = io.BytesIO()
    page.save(buf, "PDF", resolution=dpi)
    buf.seek(0)
    return buf.read()