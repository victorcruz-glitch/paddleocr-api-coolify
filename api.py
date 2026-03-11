from fastapi import FastAPI, File, UploadFile, HTTPException, Query
from pydantic import BaseModel
import numpy as np
import cv2
from paddleocr import PaddleOCR
import base64
import time

app = FastAPI(title="PaddleOCR API - CPU Multi-Model")

# Se a opção upscale for ativada (apenas para o endpoint /predict/file por enquanto)
from cv2 import dnn_superres
try:
    print("Carregando modelo de Upscale (FSRCNN)...")
    sr = dnn_superres.DnnSuperResImpl_create()
    sr.readModel('/app/FSRCNN_x4.pb')
    sr.setModel('fsrcnn', 4)
    has_upscaler = True
except Exception as e:
    print(f"Erro ao carregar upscaler: {e}")
    has_upscaler = False

# 1. Modelo Mobile (Leve e rápido)
print("Carregando modelo Mobile (v5)...")
ocr_mobile = PaddleOCR(use_angle_cls=True, lang="pt", ocr_version="PP-OCRv4")

# 2. Modelo Server (Pesado e preciso)
print("Carregando modelo Server (v5)...")
ocr_server = PaddleOCR(
    use_angle_cls=True, 
    lang="pt",
    ocr_version="PP-OCRv4", # Força a pipeline V5 por debaixo dos panos 
    det_algorithm="DB",
    rec_algorithm="SVTR_LCNet", 
    det_limit_side_len=1280
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
    model: str # Obrigatório (mobile ou server)
    preprocess: bool = False # Valor default (desativado por segurança)

def preprocess_for_ocr(img_array):
    """
    Tratamento de imagem suave: Apenas converte para escala de cinza, 
    aumenta o contraste linearmente (para fotos apagadas) e aplica um sharpening.
    Ideal para não destruir imagens digitais, mas ajudar fotos ruins.
    """
    try:
        # 1. Converte para escala de cinza (remove cores distrativas como fundos vermelhos ou azuis)
        gray = cv2.cvtColor(img_array, cv2.COLOR_BGR2GRAY)
        
        # 2. Aumento de Contraste Linear com clipping de histograma (CLAHE leve)
        # Isso escurece o que é texto e clareia o fundo, sem transformar em blocos pretos
        clahe = cv2.createCLAHE(clipLimit=1.5, tileGridSize=(8,8))
        contrast = clahe.apply(gray)
        
        # 3. Sharpening (Filtro de nitidez)
        # Realça as bordas das letras que estão borradas ou em baixa resolução
        kernel = np.array([[-1,-1,-1], [-1,9,-1], [-1,-1,-1]])
        sharpened = cv2.filter2D(contrast, -1, kernel)
        
        # Converte de volta para 3 canais (RGB) porque o PaddleOCR espera esse formato
        final_img = cv2.cvtColor(sharpened, cv2.COLOR_GRAY2RGB)
        return final_img
    except Exception as e:
        print(f"Erro no pré-processamento: {e}")
        return img_array

def process_image(img_array, model_type="mobile", apply_preprocess=True, apply_upscale=False):
    """
    Função base para processar a imagem.
    """
    start_time = time.time()
    
    # Aplica IA de Super Resolução (Upscale 4x) se solicitado e disponível
    img_to_ocr = img_array
    if apply_upscale and has_upscaler:
        try:
            print("Aplicando AI Upscale FSRCNN...")
            img_to_ocr = sr.upsample(img_to_ocr)
        except Exception as e:
            print(f"Erro no Upscale: {e}")
            
    # Aplica o filtro de limpeza visual clássico se solicitado
    if apply_preprocess:
        img_to_ocr = preprocess_for_ocr(img_to_ocr)
    
    # Seleciona o motor de inferência
    if model_type.lower() == "server":
        engine = ocr_server
        used = "server"
    else:
        engine = ocr_mobile
        used = "mobile"
    
    # Roda o OCR na imagem tratada
    result = engine.ocr(img_to_ocr)
    
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

        data, full_text_str, proc_time, used = process_image(img_cv2, payload.model, payload.preprocess)
        
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
async def ocr_file(
    file: UploadFile = File(...), 
    model: str = Query(..., description="Obrigatório: Escolha 'mobile' ou 'server'"),
    preprocess: str = Query("false", description="Opcional: 'true' ou 'false'"),
    upscale: str = Query("false", description="Opcional: Ativa IA Super Resolução 4x (Aumenta o tempo de processamento)")
):
    try:
        contents = await file.read()
        np_arr = np.frombuffer(contents, np.uint8)
        img_cv2 = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
        
        if img_cv2 is None:
             raise ValueError("Falha ao processar arquivo.")
             
        # Converte string explícita para boolean
        do_preprocess = preprocess.lower() == "true"
        do_upscale = upscale.lower() == "true"

        data, full_text_str, proc_time, used = process_image(img_cv2, model, do_preprocess, do_upscale)
        
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
