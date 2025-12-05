import cv2
import numpy as np


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

    dst = np.array([
        [0, 0],
        [maxWidth - 1, 0],
        [maxWidth - 1, maxHeight - 1],
        [0, maxHeight - 1]
    ], dtype="float32")

    # matriz de transformación y warp
    M = cv2.getPerspectiveTransform(rect, dst)
    warped = cv2.warpPerspective(image, M, (maxWidth, maxHeight))

    return warped


def find_document_contour(edged: np.ndarray, min_area: float = 5000) -> np.ndarray | None:
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
    rotate_portrait: bool = True,   # NUEVO
    enhance_mode: str = "soft",   # "soft" o "hard"
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
        image_bgr = cv2.resize(image_bgr, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)

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
            35,   # tamaño del bloque (ajustable: debe ser impar)
            10    # constante que baja/sube el umbral
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
        pad, pad, pad, pad,
        borderType=cv2.BORDER_CONSTANT,
        value=(255, 255, 255),
    )

    # 7. Rotar a portrait (forzado si querés)
    if rotate_portrait:
        warped_padded = cv2.rotate(warped_padded, cv2.ROTATE_90_COUNTERCLOCKWISE)

    return warped_padded