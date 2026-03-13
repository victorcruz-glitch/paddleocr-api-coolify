from fastapi import FastAPI, File, UploadFile, Query
from pydantic import BaseModel
from cv2 import dnn_superres
from paddleocr import PaddleOCR
import base64
import cv2
import json
import numpy as np
import time


app = FastAPI(title="PaddleOCR API - PP-OCRv5 (3.x)")


def load_upscalers():
    try:
        print("Carregando modelos de Upscale (FSRCNN 2x e 4x)...")
        sr_x2 = dnn_superres.DnnSuperResImpl_create()
        sr_x2.readModel("/app/FSRCNN_x2.pb")
        sr_x2.setModel("fsrcnn", 2)

        sr_x4 = dnn_superres.DnnSuperResImpl_create()
        sr_x4.readModel("/app/FSRCNN_x4.pb")
        sr_x4.setModel("fsrcnn", 4)
        return sr_x2, sr_x4, True
    except Exception as e:
        print(f"Erro ao carregar upscaler: {e}")
        return None, None, False


sr_x2, sr_x4, has_upscaler = load_upscalers()


def load_engines():
    common = {
        "lang": "pt",
        "device": "cpu",
        "use_doc_orientation_classify": False,
        "use_doc_unwarping": False,
        "use_textline_orientation": False,
    }

    print("Carregando modelo PP-OCRv5 server real (PaddleOCR 3.x)...")
    server = PaddleOCR(**common)

    print("Carregando modelo PP-OCRv5 mobile real (PaddleOCR 3.x)...")
    mobile = PaddleOCR(
        **common,
        text_detection_model_name="PP-OCRv5_mobile_det",
        text_recognition_model_name="latin_PP-OCRv5_mobile_rec",
    )
    return mobile, server


ocr_mobile, ocr_server = load_engines()


class OCRResponse(BaseModel):
    success: bool
    processing_time_ms: float
    model_used: str
    preprocess_applied: bool
    upscale_applied: bool
    strategy_used: str
    data: list
    full_text: str = ""
    message: str = ""


class ImagePayload(BaseModel):
    image_base64: str
    model: str
    strategy: str
    preprocess: bool = False
    upscale: int = 0


def preprocess_for_ocr(img_array):
    try:
        gray = cv2.cvtColor(img_array, cv2.COLOR_BGR2GRAY)
        clahe = cv2.createCLAHE(clipLimit=1.5, tileGridSize=(8, 8))
        contrast = clahe.apply(gray)
        kernel = np.array([[-1, -1, -1], [-1, 9, -1], [-1, -1, -1]])
        sharpened = cv2.filter2D(contrast, -1, kernel)
        return cv2.cvtColor(sharpened, cv2.COLOR_GRAY2RGB)
    except Exception as e:
        print(f"Erro no pré-processamento: {e}")
        return img_array


def normalize_json(res):
    data = res.json
    if isinstance(data, str):
        return json.loads(data)
    return data


def run_ocr(engine, img_array):
    results = engine.predict(img_array)
    extracted = []
    for res in results:
        data = normalize_json(res)
        payload = data.get("res", data)
        polys = payload.get("rec_polys") or payload.get("dt_polys") or []
        texts = payload.get("rec_texts") or []
        scores = payload.get("rec_scores") or []
        for idx, text in enumerate(texts):
            if idx >= len(polys):
                continue
            score = float(scores[idx]) if idx < len(scores) else 0.0
            extracted.append({
                "box": polys[idx],
                "text": text,
                "confidence": score,
            })
    return extracted


def shift_box(box, offset_x, offset_y):
    shifted = []
    for point in box:
        shifted.append([point[0] + offset_x, point[1] + offset_y])
    return shifted


def dedupe_data(items):
    deduped = []
    seen = set()
    for item in items:
        text = item["text"].strip().lower()
        x = int(item["box"][0][0] / 10)
        y = int(item["box"][0][1] / 10)
        key = (text, x, y)
        if key in seen:
            continue
        seen.add(key)
        deduped.append(item)
    return deduped


def format_full_text(extracted_data):
    if not extracted_data:
        return ""
    sorted_data = sorted(extracted_data, key=lambda x: x["box"][0][1])
    lines = []
    current_line = []
    last_y = sorted_data[0]["box"][0][1]
    for item in sorted_data:
        current_y = item["box"][0][1]
        if abs(current_y - last_y) < 15:
            current_line.append(item)
        else:
            current_line.sort(key=lambda x: x["box"][0][0])
            lines.append(" ".join([i["text"] for i in current_line]))
            current_line = [item]
            last_y = current_y
    if current_line:
        current_line.sort(key=lambda x: x["box"][0][0])
        lines.append(" ".join([i["text"] for i in current_line]))
    return "\n".join(lines)


def split_quadrants(img_array, overlap_ratio=0.12):
    h, w = img_array.shape[:2]
    overlap_x = int(w * overlap_ratio)
    overlap_y = int(h * overlap_ratio)
    mid_x = w // 2
    mid_y = h // 2
    regions = [
        (0, 0, min(w, mid_x + overlap_x), min(h, mid_y + overlap_y)),
        (max(0, mid_x - overlap_x), 0, w, min(h, mid_y + overlap_y)),
        (0, max(0, mid_y - overlap_y), min(w, mid_x + overlap_x), h),
        (max(0, mid_x - overlap_x), max(0, mid_y - overlap_y), w, h),
    ]
    return [(img_array[y1:y2, x1:x2], x1, y1) for x1, y1, x2, y2 in regions]


def process_image(img_array, model_type, apply_preprocess, upscale_multiplier, strategy):
    start_time = time.time()
    upscale_status = False
    preprocess_status = False

    img_to_ocr = img_array
    if upscale_multiplier > 0 and has_upscaler:
        try:
            if upscale_multiplier <= 2:
                print("Aplicando AI Upscale FSRCNN 2x...")
                img_to_ocr = sr_x2.upsample(img_to_ocr)
            else:
                print("Aplicando AI Upscale FSRCNN 4x...")
                img_to_ocr = sr_x4.upsample(img_to_ocr)
            upscale_status = True
        except Exception as e:
            print(f"Erro no Upscale: {e}")

    if apply_preprocess:
        img_to_ocr = preprocess_for_ocr(img_to_ocr)
        preprocess_status = True

    if model_type.lower() == "server":
        engine = ocr_server
        used = "server"
    elif model_type.lower() == "mobile":
        engine = ocr_mobile
        used = "mobile"
    else:
        raise ValueError("model deve ser 'mobile' ou 'server'")

    strategy_used = strategy.lower()
    if strategy_used not in ("whole", "quadrants"):
        raise ValueError("strategy deve ser 'whole' ou 'quadrants'")

    if strategy_used == "whole":
        extracted_data = run_ocr(engine, img_to_ocr)
    else:
        extracted_data = []
        extracted_data.extend(run_ocr(engine, img_to_ocr))
        for tile, offset_x, offset_y in split_quadrants(img_to_ocr):
            tile_data = run_ocr(engine, tile)
            for item in tile_data:
                item["box"] = shift_box(item["box"], offset_x, offset_y)
                extracted_data.append(item)
        extracted_data = dedupe_data(extracted_data)

    processing_time = (time.time() - start_time) * 1000
    full_text = format_full_text(extracted_data)
    return extracted_data, full_text, processing_time, used, preprocess_status, upscale_status, strategy_used


@app.post("/predict/base64", response_model=OCRResponse)
async def ocr_base64(payload: ImagePayload):
    try:
        img_bytes = base64.b64decode(payload.image_base64)
        np_arr = np.frombuffer(img_bytes, np.uint8)
        img_cv2 = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
        if img_cv2 is None:
            raise ValueError("Falha ao decodificar imagem. Verifique se o base64 está correto.")

        data, full_text_str, proc_time, used, prep_status, up_status, strategy_used = process_image(
            img_cv2,
            payload.model,
            payload.preprocess,
            payload.upscale,
            payload.strategy,
        )

        return OCRResponse(
            success=True,
            processing_time_ms=proc_time,
            model_used=used,
            preprocess_applied=prep_status,
            upscale_applied=up_status,
            strategy_used=strategy_used,
            data=data,
            full_text=full_text_str,
            message="Sucesso",
        )
    except Exception as e:
        return OCRResponse(
            success=False,
            processing_time_ms=0,
            model_used="",
            preprocess_applied=False,
            upscale_applied=False,
            strategy_used="",
            data=[],
            message=str(e),
        )


@app.post("/predict/file", response_model=OCRResponse)
async def ocr_file(
    file: UploadFile = File(...),
    model: str = Query(..., description="Obrigatório: Escolha 'mobile' ou 'server'"),
    strategy: str = Query(..., description="Obrigatório: 'whole' ou 'quadrants'"),
    preprocess: str = Query("false", description="Opcional: 'true' ou 'false'"),
    upscale: int = Query(0, description="Opcional: Multiplicador de Upscale. Escolha 0, 2 ou 4"),
):
    try:
        contents = await file.read()
        np_arr = np.frombuffer(contents, np.uint8)
        img_cv2 = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
        if img_cv2 is None:
            raise ValueError("Falha ao processar arquivo.")

        do_preprocess = preprocess.lower() == "true"
        data, full_text_str, proc_time, used, prep_status, up_status, strategy_used = process_image(
            img_cv2,
            model,
            do_preprocess,
            upscale,
            strategy,
        )

        return OCRResponse(
            success=True,
            processing_time_ms=proc_time,
            model_used=used,
            preprocess_applied=prep_status,
            upscale_applied=up_status,
            strategy_used=strategy_used,
            data=data,
            full_text=full_text_str,
            message="Sucesso",
        )
    except Exception as e:
        return OCRResponse(
            success=False,
            processing_time_ms=0,
            model_used="",
            preprocess_applied=False,
            upscale_applied=False,
            strategy_used="",
            data=[],
            message=str(e),
        )


@app.get("/health")
def health_check():
    return {"status": "ok", "service": "PaddleOCR-CPU", "stack": "paddleocr-3.x"}
