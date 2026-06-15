from fastapi import FastAPI, Form
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import PromptTemplate
from dotenv import load_dotenv
import os, warnings

# Cargar variables de entorno desde .env
load_dotenv()

api_key = os.getenv("GOOGLE_API_KEY")
if not api_key:
    warnings.warn("GOOGLE_API_KEY no está cargada.")

# Crear la aplicación FastAPI
app = FastAPI(title="Generador de Posts de Viaje")

# Definir el template del prompt
template = PromptTemplate(
    input_variables=["lugar", "transporte", "dias"],
    template="""Crea un viaje por {lugar} sobre {transporte} durante {dias}.

Requisitos:
- Descripción del viaje
- Actividades recomendadas
- Lugares para visitar
- Consejos útiles
Criterios del post:
- Tono amigable y entusiasta
- Longitud entre 150 y 200 palabras
- Formato de lista para actividades y lugares

POST:
"""
)

# Inicializar el modelo LLM de Google Generative AI
llm = ChatGoogleGenerativeAI(
    model="gemini-2.0-flash",
    temperature=0.8,
    api_key=api_key
)

# Crear la secuencia ejecutable
chain3 = template | llm | StrOutputParser()

# Endpoint de FastAPI
@app.post("/plan-viaje")
async def plan_viaje(
    lugar: str = Form(...),
    transporte: str = Form(...),
    dias: int = Form(...)
):
    """
    Genera un post de viaje personalizado usando LLM.
    """
    # Preparar los datos de entrada
    input_dict = {
        "lugar": lugar,
        "transporte": transporte,
        "dias": f"{dias} días"
    }
    

    # Ejecutar la secuencia
    post_viaje = chain3.invoke(input_dict)

    # Imprimir en consola para depuración
    print("Post generado:", post_viaje)

    return {"mensaje": post_viaje}
