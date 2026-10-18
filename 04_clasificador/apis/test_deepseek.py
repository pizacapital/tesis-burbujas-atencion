# Prueba de conexion al API de DeepSeek
# Requiere: pip install openai python-dotenv

import os
from dotenv import load_dotenv
from openai import OpenAI

# Carga las variables del archivo deepseek.env que esta en esta misma carpeta
load_dotenv("deepseek.env")

api_key = os.environ.get("DEEPSEEK_API_KEY")
if not api_key:
    raise SystemExit("No se encontro DEEPSEEK_API_KEY. Revisa el archivo deepseek.env")

print(f"Clave cargada: {api_key[:8]}... (ok)")

client = OpenAI(
    api_key=api_key,
    base_url="https://api.deepseek.com"
)

respuesta = client.chat.completions.create(
    model="deepseek-chat",  # o "deepseek-reasoner" para razonamiento
    messages=[{"role": "user", "content": "Hola, responde con una sola frase."}]
)

print(respuesta.choices[0].message.content)
