# travel-ia-back

## creamos un entorno virtual:
python -m venv venv

### Instalamos FASTAPI 
pip install fastapi[standard]

## Instalar langchain langchain-google-genai google-generativeai
pip install -q langchain langchain-google-genai google-generativeai

## Crear una variable de configuración
Creamos un archivo .env en la carpeta raiz

Dentro del archivo, escribe tus variables (una por línea):
DEBUG=True
DATABASE_URL=sqlite:///db.sqlite3
SECRET_KEY=mi_clave_secreta

## Instala la librería para leer el archivo .env desde Python:
pip install python-dotenv

## En tu código Python (por ejemplo app.py), carga las variables:
pip freeze > requirements.txt

## Activamos el entorno virtual:
venv\Scripts\activate

## Cuando el entorno esté activo, deberías ver algo como:
(venv) C:\ruta\del\proyecto>

### Ejecutamos el servidor
fastapi dev main.py