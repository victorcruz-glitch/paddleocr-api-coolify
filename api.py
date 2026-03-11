from fastapi import FastAPI, File, UploadFile, HTTPException, Query
from pydantic import BaseModel
import numpy as np
import cv2
from paddleocr import PaddleOCR
import base64
import time

app = FastAPI(title="PaddleOCR API - CPU Multi-Model")

# --- Instancia os DOIS modelos na memória na hora que a API sobe ---

# 1. Modelo Mobile (Leve e rápido)
print("Carregando modelo Mobile...")
ocr_mobile = PaddleOCR(use_angle_cls=True, lang="pt")

# 2. Modelo Server (Pesado e preciso)
print("Carregando modelo Server...")
ocr_server = PaddleOCR(
    use_angle_cls=True, 
    lang="pt",
    det_algorithm="DB",
    rec_algorithm="SVTR_LCNet", # Força arquitetura server v4/v5
    det_limit_side_len=1280     # Lê imagens maiores sem redimensionar tanto (mais precisão)
)

class OCRResponse(BaseModel):
    success: bool
    processing_time_ms: float
    model_used: str
    data: list
    full_text: str = ""
    message: str = ""

class ImagePayload(BaseModel):
    image_base64: str
    model: str = "mobile" # Valor default (mobile ou server)

def process_image(img_array, model_type="mobile"):
    """
    Função base para processar a imagem. Escolhe o modelo baseado no parâmetro.
    """
    start_time = time.time()
    
    # Seleciona o motor de inferência
    if model_type.lower() == "server":
        engine = ocr_server
        used = "server"
    else:
        engine = ocr_mobile
        used = "mobile"
    
    # O PaddleOCR espera uma imagem em formato numpy RGB (OpenCV)
    result = engine.ocr(img_array)
    
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
                
    # --- Lógica para formatar o full_text inteligentemente ---
    full_text = ""
    if extracted_data:
        # 1. Ordena os blocos de texto usando a coordenada Y inicial do box (para ler de cima para baixo)
        sorted_data = sorted(extracted_data, key=lambda x: x['box'][0][1])
        
        lines = []
        current_line = []
        # Agrupa textos que estão na mesma "linha" horizontal (com margem de erro Y de 10 pixels)
        last_y = sorted_data[0]['box'][0][1]
        
        for item in sorted_data:
            current_y = item['box'][0][1]
            # Se a diferença Y for menor que 15 pixels, consideramos a mesma linha
            if abs(current_y - last_y) < 15:
                current_line.append(item)
            else:
                # Ordena a linha atual da esquerda para a direita (X inicial: box[0][0])
                current_line.sort(key=lambda x: x['box'][0][0])
                lines.append(" ".join([i['text'] for i in current_line]))
                current_line = [item]
                last_y = current_y
                
        # Adiciona a última linha
        if current_line:
            current_line.sort(key=lambda x: x['box'][0][0])
            lines.append(" ".join([i['text'] for i in current_line]))
            
        full_text = "\n".join(lines)
                
    return extracted_data, full_text, processing_time, used

@app.post("/predict/base64", response_model=OCRResponse)
async def ocr_base64(payload: ImagePayload):
    try:
        # Decodifica base64 para imagem
        img_bytes = base64.b64decode(payload.image_base64)
        np_arr = np.frombuffer(img_bytes, np.uint8)
        img_cv2 = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
        
        if img_cv2 is None:
             raise ValueError("Falha ao decodificar imagem. Verifique se o base64 está correto.")

        data, full_text_str, proc_time, used = process_image(img_cv2, payload.model)
        
        return OCRResponse(
            success=True, 
            processing_time_ms=proc_time,
            model_used=used,
            data=data,
            full_text=full_text_str,
            message="Sucesso"
        )
        
    except Exception as e:
        return OCRResponse(
            success=False, 
            processing_time_ms=0,
            model_used="",
            data=[],
            message=str(e)
        )

@app.post("/predict/file", response_model=OCRResponse)
async def ocr_file(file: UploadFile = File(...), model: str = Query("mobile", description="Escolha 'mobile' ou 'server'")):
    try:
        contents = await file.read()
        np_arr = np.frombuffer(contents, np.uint8)
        img_cv2 = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
        
        if img_cv2 is None:
             raise ValueError("Falha ao processar arquivo.")

        data, full_text_str, proc_time, used = process_image(img_cv2, model)
        
        return OCRResponse(
            success=True, 
            processing_time_ms=proc_time,
            model_used=used,
            data=data,
            full_text=full_text_str,
            message="Sucesso"
        )
        
    except Exception as e:
        return OCRResponse(
            success=False, 
            processing_time_ms=0,
            model_used="",
            data=[],
            message=str(e)
        )

@app.get("/health")
def health_check():
    return {"status": "ok", "service": "PaddleOCR-CPU"}
