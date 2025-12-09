import cv2
import numpy as np
from typing import List

def order_points(pts: np.ndarray) -> np.ndarray:
    """
    Ordena los 4 puntos de un contorno en: topleft, topright, bottomright, bottomleft.
    """
    rect = np.zeros((4, 2), dtype="float32")

    # suma y resta de coordenadas para identificar esquinas
    s = pts.sum(axis=1)
    rect[0] = pts[np.argmin(s)]  # top-left
    rect[2] = pts[np.argmax(s)]  # bottom-right

    diff = np.diff(pts, axis=1)
    rect[1] = pts[np.argmin(diff)]  # top-right
    rect[3] = pts[np.argmax(diff)]  # bottom-left

    return rect

def four_point_transform(image: np.ndarray, pts: np.ndarray) -> np.ndarray:
    """
    Hace la corrección de perspectiva a partir de 4 puntos.
    """
    rect = order_points(pts)
    (tl, tr, br, bl) = rect

    # calcular ancho y alto máximo del nuevo rectángulo
    widthA = np.linalg.norm(br - bl)
    widthB = np.linalg.norm(tr - tl)
    maxWidth = int(max(widthA, widthB))

    heightA = np.linalg.norm(tr - br)
    heightB = np.linalg.norm(tl - bl)
    maxHeight = int(max(heightA, heightB))

    dst = np.array(
        [[0, 0], [maxWidth - 1, 0], [maxWidth - 1, maxHeight - 1], [0, maxHeight - 1]],
        dtype="float32",
    )

    # matriz de transformación y warp
    M = cv2.getPerspectiveTransform(rect, dst)
    warped = cv2.warpPerspective(image, M, (maxWidth, maxHeight))

    return warped

def find_document_contour(
    edged: np.ndarray, min_area: float = 5000
) -> np.ndarray | None:
    """
    Busca el contorno que parezca ser una tarjeta (4 lados, área grande).
    Devuelve 4 puntos (x, y) o None si no encuentra un contorno adecuado.
    """
    contours, _ = cv2.findContours(edged.copy(), cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)

    # ordenar por área (descendente)
    contours = sorted(contours, key=cv2.contourArea, reverse=True)

    for c in contours:
        area = cv2.contourArea(c)
        if area < min_area:
            continue

        # aproximar contorno a un polígono
        peri = cv2.arcLength(c, True)
        approx = cv2.approxPolyDP(c, 0.02 * peri, True)

        # si tiene 4 puntos, probablemente es el documento
        if len(approx) == 4:
            return approx.reshape(4, 2)

    return None

def process_image_to_document(
    image_bgr: np.ndarray,
    debug: bool = False,
    margin_ratio: float = 0.06,  # NUEVO: 6% del lado más pequeño como margen
    rotate_portrait: bool = True,  # NUEVO
    enhance_mode: str = "soft",  # "soft" o "hard"
) -> np.ndarray:
    """
    Recibe una imagen BGR y devuelve el documento recortado y corregido.
    margin_ratio: fracción del tamaño mínimo de la tarjeta usada como margen.
    """

    # 1. Redimensionar
    orig = image_bgr.copy()
    orig_h, orig_w = orig.shape[:2]
    max_dim = 1000

    scale = 1.0
    if max(orig_h, orig_w) > max_dim:
        scale = max_dim / float(max(orig_h, orig_w))
        image_bgr = cv2.resize(
            image_bgr, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA
        )

    # 2. Preprocesado
    gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (5, 5), 0)
    edged = cv2.Canny(gray, 50, 150)

    if debug:
        cv2.imwrite("debug_edges.jpg", edged)

    # 3. Buscar contorno
    doc_contour = find_document_contour(edged)
    if doc_contour is None:
        raise ValueError("No se encontró un contorno de documento adecuado")

    doc_contour = doc_contour / scale

    if debug:
        debug_img = orig.copy()
        cv2.drawContours(debug_img, [doc_contour.astype(int)], -1, (0, 255, 0), 3)
        cv2.imwrite("debug_contour.jpg", debug_img)

    # 4. Perspectiva
    warped = four_point_transform(orig, doc_contour.astype("float32"))

    # 5. Mejora de contraste / legibilidad
    warped_gray = cv2.cvtColor(warped, cv2.COLOR_BGR2GRAY)

    if enhance_mode == "hard":
        # Estilo "documento escaneado" B/N
        # Eliminamos gran parte de sombras
        blurred = cv2.GaussianBlur(warped_gray, (5, 5), 0)
        bin_doc = cv2.adaptiveThreshold(
            blurred,
            255,
            cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY,
            35,  # tamaño del bloque (ajustable: debe ser impar)
            10,  # constante que baja/sube el umbral
        )
        warped_final = cv2.cvtColor(bin_doc, cv2.COLOR_GRAY2BGR)
    else:
        # "soft": mejorar contraste sin llegar a B/N duro
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        cl = clahe.apply(warped_gray)
        # Suavizar un poco para quitar manchones finos
        cl = cv2.medianBlur(cl, 3)
        warped_final = cv2.cvtColor(cl, cv2.COLOR_GRAY2BGR)

    # 6. Margen blanco (igual que antes)
    h, w = warped_final.shape[:2]
    base = min(h, w)
    pad = int(base * margin_ratio)

    warped_padded = cv2.copyMakeBorder(
        warped_final,
        pad,
        pad,
        pad,
        pad,
        borderType=cv2.BORDER_CONSTANT,
        value=(255, 255, 255),
    )

    # 7. Rotar a landscape (tarjeta acostada)
    if rotate_portrait:  # o rotate_landscape
        h, w = warped_padded.shape[:2]
        if h > w:
            warped_padded = cv2.rotate(
                warped_padded,
                cv2.ROTATE_90_CLOCKWISE,
            )

    return warped_padded

def shrink_quad(quad: np.ndarray, factor: float = 0.92) -> np.ndarray:
    """
    Encoge un cuadrilátero alrededor de su centro.
    quad: array (4,2) con puntos [x,y].
    factor: < 1.0 → más chico; 1.0 → sin cambio.
    """
    quad = quad.astype(np.float32)
    center = quad.mean(axis=0, keepdims=True)  # (1,2)
    return center + (quad - center) * factor

def build_foreground_mask_by_bg_color(
    image_bgr: np.ndarray,
    border_width: int = 15,
    color_thresh: float = 10.0,
) -> np.ndarray:
    """
    Intenta estimar el color de fondo tomando los bordes de la imagen
    y construye una máscara donde:
    - 0 = fondo (similar al color de fondo)
    - 255 = foreground (distinto al fondo)

    Devuelve una máscara uint8 (0 o 255).
    """

    h, w = image_bgr.shape[:2]
    if h < 2 * border_width or w < 2 * border_width:
        # imagen muy chica, devolvemos máscara vacía
        return np.zeros((h, w), dtype=np.uint8)

    lab = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2LAB)

    # Zonas de borde: top, bottom, left, right
    top = lab[0:border_width, :, :]
    bottom = lab[h - border_width : h, :, :]
    left = lab[:, 0:border_width, :]
    right = lab[:, w - border_width : w, :]

    # Concatenar todos los bordes
    border_pixels = np.concatenate(
        [
            top.reshape(-1, 3),
            bottom.reshape(-1, 3),
            left.reshape(-1, 3),
            right.reshape(-1, 3),
        ],
        axis=0,
    )

    # Color promedio del fondo en Lab
    bg_mean = border_pixels.mean(axis=0, keepdims=True)  # (1,3)

    # Distancia a ese color para cada pixel
    lab_f = lab.astype(np.float32)
    diff = lab_f - bg_mean  # broadcasting (H,W,3) - (1,3)
    dist = np.linalg.norm(diff, axis=2)  # (H,W)

    # Umbral: pixeles cercanos al bg_mean => fondo
    # Ajustar color_thresh según tus fotos (10, 15, 20...)
    mask_fg = np.zeros((h, w), dtype=np.uint8)
    mask_fg[dist >= color_thresh] = 255

    # Morfología para limpiar ruido
    kernel = np.ones((5, 5), np.uint8)
    mask_fg = cv2.morphologyEx(mask_fg, cv2.MORPH_OPEN, kernel)
    mask_fg = cv2.morphologyEx(mask_fg, cv2.MORPH_CLOSE, kernel)

    return mask_fg

def auto_crop_background(
    img_bgr: np.ndarray,
    diff_thresh: int = 12,
    content_fraction: float = 0.08,
    max_crop_frac: float = 0.35,
) -> np.ndarray:
    """
    Recorta bordes "uniformes" (fondo) alrededor de la tarjeta.

    - diff_thresh: cuánta diferencia de gris con el fondo consideramos "contenido".
    - content_fraction: % mínimo de píxeles distintos al fondo para decir "acá empieza el documento".
    - max_crop_frac: máximo % del tamaño que se permite recortar por cada lado.

    Si no encuentra nada razonable, devuelve la imagen original.
    """
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    h, w = gray.shape

    if h < 20 or w < 20:
        return img_bgr.copy()

    # Tomamos el color de fondo promedio del perímetro
    border = np.concatenate([
        gray[0, :],          # top
        gray[-1, :],         # bottom
        gray[:, 0],          # left
        gray[:, -1],         # right
    ])
    bg_val = np.median(border)

    def find_cut_from_top(g):
        max_top = int(h * max_crop_frac)
        for y in range(max_top):
            row = g[y, :]
            diff = np.abs(row.astype(np.int16) - int(bg_val))
            # porcentaje de píxeles que difieren del fondo
            frac = (diff > diff_thresh).mean()
            if frac > content_fraction:
                return y
        return 0

    def find_cut_from_bottom(g):
        max_bottom = int(h * max_crop_frac)
        for i in range(max_bottom):
            y = h - 1 - i
            row = g[y, :]
            diff = np.abs(row.astype(np.int16) - int(bg_val))
            frac = (diff > diff_thresh).mean()
            if frac > content_fraction:
                return y + 1
        return h

    def find_cut_from_left(g):
        max_left = int(w * max_crop_frac)
        for x in range(max_left):
            col = g[:, x]
            diff = np.abs(col.astype(np.int16) - int(bg_val))
            frac = (diff > diff_thresh).mean()
            if frac > content_fraction:
                return x
        return 0

    def find_cut_from_right(g):
        max_right = int(w * max_crop_frac)
        for i in range(max_right):
            x = w - 1 - i
            col = g[:, x]
            diff = np.abs(col.astype(np.int16) - int(bg_val))
            frac = (diff > diff_thresh).mean()
            if frac > content_fraction:
                return x + 1
        return w

    top = find_cut_from_top(gray)
    bottom = find_cut_from_bottom(gray)
    left = find_cut_from_left(gray)
    right = find_cut_from_right(gray)

    # sanity check
    if bottom - top < 10 or right - left < 10:
        return img_bgr.copy()

    return img_bgr[top:bottom, left:right].copy()

def process_image_to_documents(
    image_bgr: np.ndarray,
    debug: bool = False,
    debug_prefix: str = "",
    margin_ratio: float = 0.06,
    rotate_portrait: bool = True,
    max_docs: int = 6,
    enhance_mode: str = "soft",   # "soft" o "hard"
) -> List[np.ndarray]:
    """
    Procesa una imagen que puede contener 1 o más documentos.
    Devuelve una lista de imágenes BGR, una por documento detectado.

    - max_docs: máximo de documentos a devolver (por ej. 6 para tu PDF 3x2).
    """

    processed_images: List[np.ndarray] = []

    # 1) Redimensionar
    orig = image_bgr.copy()
    orig_h, orig_w = orig.shape[:2]
    max_dim = 1000

    scale = 1.0
    if max(orig_h, orig_w) > max_dim:
        scale = max_dim / float(max(orig_h, orig_w))
        image_bgr = cv2.resize(
            image_bgr, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA
        )

    # 2) Preprocesado: gris + blur + Canny
    gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (5, 5), 0)
    edged = cv2.Canny(gray, 50, 150)

    if debug:
        cv2.imwrite(f"{debug_prefix}debug_edges_multi.jpg", edged)

    # 3) Contornos externos sobre Canny
    contours, _ = cv2.findContours(
        edged.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )

    h, w = edged.shape[:2]
    min_area_ratio = 0.05  # 5% del área total como mínimo
    min_area = min_area_ratio * h * w

    internal_candidates: list[tuple[float, np.ndarray]] = []
    border_candidates: list[tuple[float, np.ndarray]] = []

    border_eps = 5  # margen en píxeles para considerar que toca el borde

    for c in contours:
        area = cv2.contourArea(c)
        if area < min_area:
            continue

        x, y, cw, ch = cv2.boundingRect(c)
        touches_border = (
            x <= border_eps or
            y <= border_eps or
            x + cw >= w - border_eps or
            y + ch >= h - border_eps
        )

        # Rectángulo mínimo rotado que "envuelve" el contorno
        rect = cv2.minAreaRect(c)  # (cx, cy), (width, height), angle
        box = cv2.boxPoints(rect).astype("float32")
        box_area = cv2.contourArea(box)
        if box_area < min_area:
            continue

        # Proporción típica de tarjeta
        w_box, h_box = rect[1]
        if w_box == 0 or h_box == 0:
            continue

        ratio = max(w_box, h_box) / max(1.0, min(w_box, h_box))
        # típico DNI/tarjeta: ~1.4–1.7 → dejamos margen
        if not (1.2 <= ratio <= 2.2):
            continue

        if touches_border:
            border_candidates.append((box_area, box))
        else:
            internal_candidates.append((box_area, box))

    # Prioridad: primero rectángulos internos (no tocan borde)
    if internal_candidates:
        candidates = sorted(internal_candidates, key=lambda x: x[0], reverse=True)
    elif border_candidates:
        # Caso 2 / 4: solo tenemos cosas que tocan borde
        candidates = sorted(border_candidates, key=lambda x: x[0], reverse=True)
    else:
        candidates = []

    # Limitar a max_docs
    candidates = candidates[:max_docs]

    if not candidates:
        # Fallback final: usar todo el frame como "documento"
        full_rect = np.array(
            [[0, 0], [w - 1, 0], [w - 1, h - 1], [0, h - 1]],
            dtype="float32",
        )
        candidates = [(h * w, full_rect)]

    # Ordenar por área (más grande primero) y limitar a max_docs
    candidates.sort(key=lambda x: x[0], reverse=True)
    candidates = candidates[:max_docs]

    if not candidates:
        # Fallback: usar todo el frame como "documento"
        full_rect = np.array(
            [[0, 0], [w - 1, 0], [w - 1, h - 1], [0, h - 1]],
            dtype="float32",
        )
        candidates = [(h * w, full_rect)]

        # Procesar cada candidato
    for idx, (_, cnt) in enumerate(candidates):
        if idx >= max_docs:
            break

        # cnt está en coordenadas de la imagen reescalada → volvemos a la escala original
        doc_contour = cnt / scale

        # Encoger un poco el rectángulo para comer el fondo
        doc_contour = shrink_quad(doc_contour, factor=0.92)

        if debug:
            dbg = orig.copy()
            cv2.drawContours(dbg, [doc_contour.astype(int)], -1, (0, 255, 0), 3)
            cv2.imwrite(
                f"{debug_prefix}debug_contour_doc_{idx+1}.jpg",
                dbg,
            )

        # 4.a) Corrección de perspectiva
        warped = four_point_transform(orig, doc_contour.astype("float32"))

        # 4.b) Mejora de contraste
        warped_gray = cv2.cvtColor(warped, cv2.COLOR_BGR2GRAY)

        if enhance_mode == "hard":
            blurred = cv2.GaussianBlur(warped_gray, (5, 5), 0)
            bin_doc = cv2.adaptiveThreshold(
                blurred,
                255,
                cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                cv2.THRESH_BINARY,
                35,
                10,
            )
            warped_final = cv2.cvtColor(bin_doc, cv2.COLOR_GRAY2BGR)
        else:
            clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
            cl = clahe.apply(warped_gray)
            cl = cv2.medianBlur(cl, 3)
            warped_final = cv2.cvtColor(cl, cv2.COLOR_GRAY2BGR)

        # 4.c) NUEVO: recortar fondo uniforme alrededor de la tarjeta
        warped_final = auto_crop_background(
            warped_final,
            diff_thresh=12,
            content_fraction=0.08,
            max_crop_frac=0.35,
        )

        # 4.d) Margen blanco
        hh, ww = warped_final.shape[:2]
        base = min(hh, ww)
        pad = int(base * margin_ratio)

        warped_padded = cv2.copyMakeBorder(
            warped_final,
            pad,
            pad,
            pad,
            pad,
            borderType=cv2.BORDER_CONSTANT,
            value=(255, 255, 255),
        )

        # 4.e) Rotar a landscape (tarjeta acostada) si hace falta
        if rotate_portrait:
            h2, w2 = warped_padded.shape[:2]
            if h2 > w2:
                warped_padded = cv2.rotate(
                    warped_padded,
                    cv2.ROTATE_90_COUNTERCLOCKWISE,
                )

        processed_images.append(warped_padded)

    return processed_images