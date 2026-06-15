# main.py
from fastapi import FastAPI, Form, HTTPException
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field
from dotenv import load_dotenv
from pathlib import Path
import asyncio
import json
import logging
import os
import uuid
import warnings

# Imports del LLM (ajusta si estás usando otra integración)
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import PromptTemplate
from fastapi.middleware.cors import CORSMiddleware

# Cargar .env
load_dotenv()

# Configuración básica
API_KEY = os.getenv("GOOGLE_API_KEY")
if not API_KEY:
    warnings.warn("GOOGLE_API_KEY no está cargada. Asegúrate de ponerla en .env")

OUTPUT_DIR = Path("outputs")
RES_DIR = Path("res")

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("travel_api")

app = FastAPI(title="Travel Post Generator API")

# Asegurarse de que los directorios de salida existen
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
RES_DIR.mkdir(parents=True, exist_ok=True)

# Registrar CORS (usar "*" o lista concreta)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:4200"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ---- Modelos de request / response ----
class TravelRequest(BaseModel):
    lugar: str = Field(..., example="Rumania")
    transporte: str = Field(..., example="coche")
    dias: int = Field(..., ge=1, example=7)

class TravelResponse(BaseModel):
    id: str
    post: str

# ---- Prompt y LLM (global para reutilizar) ----
template = PromptTemplate(
    input_variables=["lugar", "transporte", "dias"],
    template="""Crea un viaje por {lugar} sobre {transporte} durante {dias}.

Rol:
    Eres un guía turístico experto. Tu trabajo es ayudar al viajero a descubrir lugares, organizar visitas y proporcionar información cultural, histórica y práctica.
    Tu objetivo es hacer que cada experiencia sea interesante, clara y agradable.

REQUISITOS:
- Llamado a la acción claro
- Enumera actividades y lugares para visitar
- Consejos útiles para el viajero
- Habla de forma cercana y profesional.
- Explica la información de manera sencilla.
- Adapta las recomendaciones al perfil del viajero.
- Prioriza la utilidad y la experiencia del visitante.
- Evita respuestas excesivamente técnicas.

PRIORIDADES:
Determina cuál es el objetivo principal del viaje:
- "imprescindibles": visitar los lugares más importantes.
- "experiencia_local": conocer la vida cotidiana y lugares menos turísticos.
- "eficiencia": aprovechar el tiempo disponible.
- "presupuesto": minimizar gastos.
- "comodidad": reducir desplazamientos y esfuerzo.

Selecciona siempre una prioridad principal.

FORMATO DE RESPUESTA:
- Comienza con un breve resumen del viaje (1-2 frases).
Para cada recomendación incluye:

- Nombre del lugar.
- Motivo por el que merece la pena.
- Tiempo aproximado de visita.
- Mejor momento del día para visitarlo.
- Consejos útiles.
- Nivel de interés:

POSIBLES VALORES:
- "imprescindible"
- "muy recomendable"
- "opcional"

POST:
"""
)

# Pasa api_key al constructor (si tu paquete lo acepta)
llm = ChatGoogleGenerativeAI(model="gemini-3.5-flash", temperature=0.8, api_key=API_KEY)
chain = template | llm | StrOutputParser()

# ---- Función auxiliar para generar (sin bloquear el event loop) ----
def _invoke_chain_sync(input_dict: dict) -> str:
    """
    Ejecuta el chain de forma sincrónica.
    LangChain invoke puede ser síncrono; como lo llamaremos desde async,
    lo envolvemos para ejecutar en thread pool si es necesario.
    """
    return chain.invoke(input_dict)

async def invoke_chain(input_dict: dict) -> str:
    # Ejecutar en hilo para no bloquear el event loop si la llamada es I/O bloqueante
    result = await asyncio.to_thread(_invoke_chain_sync, input_dict)
    return result

# ---- Endpoint POST: crear plan ----
@app.post("/plan-viaje", response_model=TravelResponse)
async def plan_viaje(payload: TravelRequest):
    """
    Recibe JSON con {lugar, transporte, dias}, genera un post y devuelve JSON con id y post.
    Guarda también el resultado en outputs/{id}.json
    """
    input_dict = {
        "lugar": payload.lugar,
        "transporte": payload.transporte,
        "dias": f"{payload.dias} días"
    }

    # Registrar el payload recibido para verificar desde el front
    logger.info("Payload recibido desde front: %s", input_dict)

    try:
        post_viaje = await invoke_chain(input_dict)
    except Exception as e:
        logger.exception("Error al invocar el LLM")
        raise HTTPException(status_code=500, detail="Error generando el post.") from e

    # crear id único y guardar como JSON
    item_id = uuid.uuid4().hex[:4] 
    out_obj = {"id": item_id, "input": input_dict, "post": post_viaje}
    out_path = OUTPUT_DIR / f"{item_id}.json"
    try:
        out_path.write_text(json.dumps(out_obj, ensure_ascii=False, indent=2), encoding="utf-8")
    except Exception as e:
        logger.exception("No se pudo guardar el JSON de salida")
        # no impedimos devolver la respuesta; sólo avisamos con log

    res_path = RES_DIR / f"{item_id}.md"
    try:
        res_content = (
            f"# Plan de viaje {item_id}\n\n"
            f"**Lugar:** {payload.lugar}\n\n"
            f"**Transporte:** {payload.transporte}\n\n"
            f"**Días:** {payload.dias}\n\n"
            f"---\n\n"
            f"{post_viaje}\n"
        )
        res_path.write_text(res_content, encoding="utf-8")
    except Exception as e:
        logger.exception("No se pudo guardar el markdown de salida")

    logger.info("Generado post id=%s", item_id)

    return JSONResponse(content={"id": item_id, "post": post_viaje})

# ---- Endpoint GET: obtener plan por id (para que otra app lo lea) ----
@app.get("/plan-viaje/{item_id}", response_model=TravelResponse)
async def get_plan(item_id: str):
    out_path = OUTPUT_DIR / f"{item_id}.json"
    if not out_path.exists():
        raise HTTPException(status_code=404, detail="Plan no encontrado")
    try:
        content = json.loads(out_path.read_text(encoding="utf-8"))
        return JSONResponse(content={"id": content.get("id"), "post": content.get("post")})
    except Exception as e:
        logger.exception("Error leyendo el JSON de salida")
        raise HTTPException(status_code=500, detail="Error leyendo el plan")
