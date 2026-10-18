# Paso 1 del arranque StockTwits - prueba de conexion con la API publica
#
# Corre en iTerm:
#   cd 'Code/stocktwits'
#   python3 scripts/test_connection.py
#
# Que hace: pide la pagina mas reciente del stream publico de $GME, verifica que
# vengan los campos que el esquema necesita (id, fecha, texto, sentimiento nativo)
# e imprime una muestra + la tasa de etiquetado de la pagina. Sin credenciales:
# la lectura publica funciona sin token (con limites mas estrictos); el registro
# formal de apps esta pausado (ver plan v2, seccion 11 y correo a developers@).
import json
import urllib.request

URL = 'https://api.stocktwits.com/api/2/streams/symbol/GME.json'
HEADERS = {
    'User-Agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/126.0.0.0 Safari/537.36',
    'Accept': 'application/json',
    'Accept-Language': 'en-US,en;q=0.9',
}
req = urllib.request.Request(URL, headers=HEADERS)
with urllib.request.urlopen(req, timeout=30) as r:
    datos = json.loads(r.read().decode('utf-8'))

msgs = datos.get('messages', [])
print(f"status api: {datos.get('response', {}).get('status')} | mensajes en la pagina: {len(msgs)}")
if not msgs:
    raise SystemExit('sin mensajes: revisar si el endpoint cambio o hay bloqueo')

etiquetados = 0
for m in msgs[:5]:
    senti = (m.get('entities', {}).get('sentiment') or {})
    basic = senti.get('basic') if isinstance(senti, dict) else None
    print(f"- id {m['id']} | {m['created_at']} | @{m['user']['username']} | "
          f"sentimiento: {basic or 'sin etiqueta'} | {m['body'][:70]!r}")
for m in msgs:
    senti = (m.get('entities', {}).get('sentiment') or {})
    if isinstance(senti, dict) and senti.get('basic'):
        etiquetados += 1
print(f"\ntasa de etiquetado de la pagina: {etiquetados}/{len(msgs)} ({etiquetados/len(msgs):.0%})")
print(f"cursor para paginar hacia atras (max): {datos.get('cursor', {}).get('max')}")
print("\nconexion ok: la lectura publica funciona. siguiente paso: backfill_symbol.py GME")
