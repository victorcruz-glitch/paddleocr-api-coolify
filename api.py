from fastapi import FastAPI, File, UploadFile, HTTPException
from pydantic import BaseModel
import numpy as np
import cv2
from paddleocr import PaddleOCR
import base64
import time

app = FastAPI(title="PaddleOCR API - CPU Lightweight Version")

# Inicializa o PaddleOCR usando o modelo "Mobile" (Tiny)
# lang="pt" -> Suporte a português, mas mantendo suporte geral a inglês tb
# use_angle_cls=True -> Ajusta a rotação automática de textos
# show_log=False -> Diminui os logs que pesam
# use_gpu=False -> Como vc quer ir só de CPU
ocr = PaddleOCR(use_angle_cls=True, lang="pt")

class OCRResponse(BaseModel):
    success: bool
    processing_time_ms: float
    data: list
    message: str = ""

class ImagePayload(BaseModel):
    image_base64: str

def process_image(img_array):
    """
    Função base para processar a imagem (numpy array) no PaddleOCR
    """
    start_time = time.time()
    
    # O PaddleOCR espera uma imagem em formato numpy RGB (OpenCV)
    result = ocr.ocr(img_array)
    
    processing_time = (time.time() - start_time) * 1000
    
    extracted_data = []
    
    if result and result[0] is not None:
        for idx in range(len(result)):
            res = result[idx]
            for line in res:
                box = line[0]  # Coordenadas da bounding box
                text = line[1][0] # O Texto extraído em si
                confidence = line[1][1] # Nível de confiança (0 a 1)
                
                extracted_data.append({
                    "box": box,
                    "text": text,
                    "confidence": float(confidence)
                })
                
    return extracted_data, processing_time

@app.post("/predict/base64", response_model=OCRResponse)
async def ocr_base64(payload: ImagePayload):
    try:
        # Decodifica base64 para imagem
        img_bytes = base64.b64decode(payload.image_base64)
        np_arr = np.frombuffer(img_bytes, np.uint8)
        img_cv2 = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
        
        if img_cv2 is None:
             raise ValueError("Falha ao decodificar imagem. Verifique se o base64 está correto.")

        data, proc_time = process_image(img_cv2)
        
        return OCRResponse(
            success=True, 
            processing_time_ms=proc_time,
            data=data,
            message="Sucesso"
        )
        
    except Exception as e:
        return OCRResponse(
            success=False, 
            processing_time_ms=0,
            data=[],
            message=str(e)
        )

@app.post("/predict/file", response_model=OCRResponse)
async def ocr_file(file: UploadFile = File(...)):
    try:
        contents = await file.read()
        np_arr = np.frombuffer(contents, np.uint8)
        img_cv2 = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
        
        if img_cv2 is None:
             raise ValueError("Falha ao processar arquivo.")

        data, proc_time = process_image(img_cv2)
        
        return OCRResponse(
            success=True, 
            processing_time_ms=proc_time,
            data=data,
            message="Sucesso"
        )
        
    except Exception as e:
        return OCRResponse(
            success=False, 
            processing_time_ms=0,
            data=[],
            message=str(e)
        )

@app.get("/health")
def health_check():
    return {"status": "ok", "service": "PaddleOCR-CPU"}
