# Prueba de conexion al API de Kimi (Moonshot AI)
# Requiere: pip install openai python-dotenv

import os
from dotenv import load_dotenv
from openai import OpenAI

# Carga las variables del archivo kimi.env que esta en esta misma carpeta
load_dotenv("/Users/ppizam/Claude/Master Thesis/Desarrollo/Metodologia/APIS/kimi.env")

api_key = os.environ.get("MOONSHOT_API_KEY")
if not api_key or api_key.startswith("PEGA-AQUI"):
    raise SystemExit("Falta tu clave real en kimi.env. Editala con: nano kimi.env")

print(f"Clave cargada: {api_key[:8]}... (ok)")

client = OpenAI(
    api_key=api_key,
    base_url="https://api.moonshot.ai/v1"
)

# Lista los modelos disponibles para confirmar el nombre exacto vigente
print("\nModelos disponibles:")
for m in client.models.list().data:
    print(" -", m.id)

# Llamada de prueba
respuesta = client.chat.completions.create(
    model="kimi-k3",  # tambien disponibles: kimi-k2.6, kimi-k2.7-code, kimi-k2.7-code-highspeed
    messages=[{"role": "user", "content": "Hola, responde con una sola frase."}]
)

print("\nRespuesta:", respuesta.choices[0].message.content)
