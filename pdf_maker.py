import math
from typing import List

import numpy as np
from PIL import Image

import io
import math
from typing import List

import numpy as np
from PIL import Image


def create_single_page_pdf_from_images(
    images_bgr: List[np.ndarray],
    output_path: str,
    dpi: int = 300,
    outer_margin_mm: float = 8.0,     # margen de la hoja
    inner_margin_mm_x: float = 6.0,   # espacio entre columnas
    inner_margin_mm_y: float = 3.0,   # espacio entre filas (más chico)
    grid_rows: int = 2,
    grid_cols: int = 2,
    grid_height_fraction: float = 0.7,
) -> None:
    if not images_bgr:
        raise ValueError("No se recibieron imágenes para generar el PDF")

    max_images = grid_rows * grid_cols
    if len(images_bgr) > max_images:
        raise ValueError(
            f"Se recibieron {len(images_bgr)} imágenes y el máximo permitido es "
            f"{max_images} para una grilla {grid_rows}x{grid_cols}"
        )

    # A4 en píxeles
    a4_width_mm, a4_height_mm = 210, 297
    a4_width_in = a4_width_mm / 25.4
    a4_height_in = a4_height_mm / 25.4

    page_width_px = int(a4_width_in * dpi)
    page_height_px = int(a4_height_in * dpi)

    # márgenes en píxeles
    outer_margin_px = int((outer_margin_mm / 25.4) * dpi)
    inner_margin_px_x = int((inner_margin_mm_x / 25.4) * dpi)
    inner_margin_px_y = int((inner_margin_mm_y / 25.4) * dpi)

    # altura que ocupa la grilla
    grid_height_px = int(page_height_px * grid_height_fraction)

    # total de píxeles ocupados por márgenes
    total_margin_x = outer_margin_px * 2 + inner_margin_px_x * (grid_cols - 1)
    total_margin_y = outer_margin_px * 2 + inner_margin_px_y * (grid_rows - 1)

    cell_width = (page_width_px - total_margin_x) // grid_cols
    cell_height = (grid_height_px - total_margin_y) // grid_rows

    if cell_width <= 0 or cell_height <= 0:
        raise ValueError("No hay espacio para las imágenes con estos parámetros")

    page = Image.new("RGB", (page_width_px, page_height_px), color=(255, 255, 255))

    # y inicial de la grilla
    grid_start_y = outer_margin_px

    for idx, img_bgr in enumerate(images_bgr):
        img_rgb = img_bgr[:, :, ::-1]
        pil_img = Image.fromarray(img_rgb)

        w, h = pil_img.size
        scale = min(cell_width / w, cell_height / h)
        new_w = int(w * scale)
        new_h = int(h * scale)
        pil_img = pil_img.resize((new_w, new_h), Image.LANCZOS)

        row = idx // grid_cols
        col = idx % grid_cols

        x0 = outer_margin_px + col * (cell_width + inner_margin_px_x)
        y0 = grid_start_y + row * (cell_height + inner_margin_px_y)

        x = x0 + (cell_width - new_w) // 2
        y = y0 + (cell_height - new_h) // 2

        page.paste(pil_img, (x, y))

    page.save(output_path, "PDF", resolution=dpi)
    
    
def _build_page_image(
    images_bgr: List[np.ndarray],
    dpi: int = 300,
    outer_margin_mm: float = 8.0,
    inner_margin_mm_x: float = 8.0,
    inner_margin_mm_y: float = 3.0,
    grid_rows: int = 2,
    grid_cols: int = 2,
    grid_height_fraction: float = 0.7,
) -> Image.Image:
    """
    Devuelve una imagen PIL con la hoja A4 armada (no guarda ni devuelve bytes).
    La usan tanto la versión que guarda a archivo como la versión que devuelve bytes.
    """

    if not images_bgr:
        raise ValueError("No se recibieron imágenes para generar el PDF")

    max_images = grid_rows * grid_cols
    if len(images_bgr) > max_images:
        raise ValueError(
            f"Se recibieron {len(images_bgr)} imágenes y el máximo permitido es "
            f"{max_images} para una grilla {grid_rows}x{grid_cols}"
        )

    # A4 en píxeles
    a4_width_mm, a4_height_mm = 210, 297
    a4_width_in = a4_width_mm / 25.4
    a4_height_in = a4_height_mm / 25.4

    page_width_px = int(a4_width_in * dpi)
    page_height_px = int(a4_height_in * dpi)

    # márgenes en píxeles
    outer_margin_px = int((outer_margin_mm / 25.4) * dpi)
    inner_margin_px_x = int((inner_margin_mm_x / 25.4) * dpi)
    inner_margin_px_y = int((inner_margin_mm_y / 25.4) * dpi)

    # altura de la grilla
    grid_height_px = int(page_height_px * grid_height_fraction)

    total_margin_x = outer_margin_px * 2 + inner_margin_px_x * (grid_cols - 1)
    total_margin_y = outer_margin_px * 2 + inner_margin_px_y * (grid_rows - 1)

    cell_width = (page_width_px - total_margin_x) // grid_cols
    cell_height = (grid_height_px - total_margin_y) // grid_rows

    if cell_width <= 0 or cell_height <= 0:
        raise ValueError("No hay espacio para las imágenes con estos parámetros")

    page = Image.new("RGB", (page_width_px, page_height_px), color=(255, 255, 255))
    grid_start_y = outer_margin_px

    for idx, img_bgr in enumerate(images_bgr):
        img_rgb = img_bgr[:, :, ::-1]
        pil_img = Image.fromarray(img_rgb)

        w, h = pil_img.size
        scale = min(cell_width / w, cell_height / h)
        new_w = int(w * scale)
        new_h = int(h * scale)
        pil_img = pil_img.resize((new_w, new_h), Image.LANCZOS)

        row = idx // grid_cols
        col = idx % grid_cols

        x0 = outer_margin_px + col * (cell_width + inner_margin_px_x)
        y0 = grid_start_y + row * (cell_height + inner_margin_px_y)

        x = x0 + (cell_width - new_w) // 2
        y = y0 + (cell_height - new_h) // 2

        page.paste(pil_img, (x, y))

    return page


# --- Versión que ya usás desde make_pdf.py (la dejamos igual pero usando el helper) ---

def create_single_page_pdf_from_images(
    images_bgr: List[np.ndarray],
    output_path: str,
    dpi: int = 300,
    outer_margin_mm: float = 8.0,
    inner_margin_mm_x: float = 8.0,
    inner_margin_mm_y: float = 3.0,
    grid_rows: int = 2,
    grid_cols: int = 2,
    grid_height_fraction: float = 0.7,
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


# --- NUEVA: para FastAPI, devuelve bytes del PDF ---

def create_single_page_pdf_bytes(
    images_bgr: List[np.ndarray],
    dpi: int = 300,
    outer_margin_mm: float = 8.0,
    inner_margin_mm_x: float = 8.0,
    inner_margin_mm_y: float = 3.0,
    grid_rows: int = 2,
    grid_cols: int = 2,
    grid_height_fraction: float = 0.7,
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