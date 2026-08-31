# Backfill historico de un cashtag por paginacion de cursor - PILOTO StockTwits
#
# Corre en iTerm (el piloto del plan, seccion 20.1):
#   cd 'Code/stocktwits'
#   python3 scripts/backfill_symbol.py GME
#
# Que hace: pagina hacia atras el stream publico del simbolo (parametro max =
# id mas antiguo de la pagina anterior) guardando cada mensaje en
# data/raw/<SIMBOLO>.jsonl (una linea por mensaje, crudo completo).
# - Reanudable: si el jsonl ya existe, retoma desde el id mas antiguo guardado.
# - Cliente PACIENTE (leccion GDELT): pausa de crucero entre paginas y esperas
#   escalonadas ante el 429 (30 s, 2, 5, 15 y 30 min); los reintentos rapidos
#   profundizan el castigo. Correr desde UNA sola maquina.
# - Registra data/log_backfill.csv y al final imprime los DOS NUMEROS del
#   piloto: fecha del mensaje mas antiguo alcanzable y paginas/hora sostenidas.
#   Esos dos numeros deciden si hacen falta las vias 2 a 4 del plan.
# - Detener con Ctrl+C es seguro: todo lo bajado queda guardado.
import csv
import json
import sys
import time
import urllib.error
import urllib.parse
import urllib.request
from datetime import datetime, timezone
from pathlib import Path

class BloqueoTemporal(Exception):
    pass


CODE = Path(__file__).resolve().parents[1]
RAW = CODE / 'data' / 'raw'
LOG = CODE / 'data' / 'log_backfill.csv'
API = 'https://api.stocktwits.com/api/2/streams/symbol/{sym}.json'
HEADERS = {
    'User-Agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/126.0.0.0 Safari/537.36',
    'Accept': 'application/json',
    'Accept-Language': 'en-US,en;q=0.9',
}
PAUSA_S = 4.0            # crucero entre paginas (conservador para la via sin token)
ESPERAS_429 = [30, 120, 300, 900, 1800]
LIMITE_FECHA = '2019-12-01'   # hasta donde interesa retroceder (ventana + colchon)
TOPE_PAGINAS_SESION = 1500    # ~5-6 h por sesion; la corrida de 19 h continuas
                              # del 9-10 ago termino en 429 escalando a 403:
                              # sesiones acotadas con descanso entre ellas


def llamar(sym, max_id=None):
    params = {'max': max_id} if max_id else {}
    url = API.format(sym=sym) + ('?' + urllib.parse.urlencode(params) if params else '')
    for intento in range(1, 7):
        try:
            req = urllib.request.Request(url, headers=HEADERS)
            with urllib.request.urlopen(req, timeout=45) as r:
                return json.loads(r.read().decode('utf-8'))
        except urllib.error.HTTPError as e:
            if e.code == 403:
                raise BloqueoTemporal('403 Forbidden: la API corto el acceso '
                                      '(escalada tras racionamiento). Alto elegante.')
            if e.code == 429 and intento <= len(ESPERAS_429):
                espera = ESPERAS_429[intento - 1]
                print(f'   ...limite de tasa (429): espero {espera // 60 or espera} '
                      f'{"min" if espera >= 60 else "s"} y reintento', flush=True)
                time.sleep(espera)
                continue
            if intento == 6:
                raise
            time.sleep(15 * intento)
        except Exception:
            if intento == 6:
                raise
            time.sleep(15 * intento)


def registrar(sym, paginas, mensajes, mas_antiguo, estado):
    nuevo = not LOG.exists()
    with open(LOG, 'a', newline='', encoding='utf-8') as f:
        w = csv.writer(f)
        if nuevo:
            w.writerow(['ts_utc', 'simbolo', 'paginas', 'mensajes', 'mas_antiguo', 'estado'])
        w.writerow([datetime.now(timezone.utc).isoformat(timespec='seconds'),
                    sym, paginas, mensajes, mas_antiguo, estado])


def main():
    if len(sys.argv) < 2:
        sys.exit('uso: python3 scripts/backfill_symbol.py SIMBOLO  (ej. GME)')
    sym = sys.argv[1].upper()
    RAW.mkdir(parents=True, exist_ok=True)
    destino = RAW / f'{sym}.jsonl'

    max_id = None
    ya = 0
    if destino.exists():
        with open(destino, encoding='utf-8') as f:
            for linea in f:
                ya += 1
                try:
                    mid = json.loads(linea)['id']
                    max_id = mid if max_id is None else min(max_id, mid)
                except (json.JSONDecodeError, KeyError):
                    continue
        print(f'reanudando {sym}: {ya:,} mensajes ya guardados; retomo desde id {max_id}')

    t0 = time.time()
    paginas = nuevos = 0
    mas_antiguo = ''
    try:
        with open(destino, 'a', encoding='utf-8') as f:
            while True:
                datos = llamar(sym, max_id)
                msgs = datos.get('messages', [])
                if not msgs:
                    print('sin mas mensajes: fin del historico alcanzable por la API')
                    break
                for m in msgs:
                    f.write(json.dumps(m, ensure_ascii=False) + '\n')
                nuevos += len(msgs)
                paginas += 1
                mas_antiguo = msgs[-1].get('created_at', '')
                max_id = datos.get('cursor', {}).get('max') or (msgs[-1]['id'] - 1)
                if paginas % 25 == 0:
                    ph = paginas / ((time.time() - t0) / 3600)
                    print(f'{paginas:5d} paginas | {nuevos:7,} mensajes nuevos | '
                          f'mas antiguo: {mas_antiguo[:10]} | {ph:.0f} pag/h', flush=True)
                if mas_antiguo[:10] and mas_antiguo[:10] < LIMITE_FECHA:
                    print(f'alcanzada la ventana de estudio ({LIMITE_FECHA}): alto')
                    break
                if paginas >= TOPE_PAGINAS_SESION:
                    print(f'tope de sesion alcanzado ({TOPE_PAGINAS_SESION} paginas, '
                          f'~{TOPE_PAGINAS_SESION * PAUSA_S / 3600:.0f}h+): alto elegante. '
                          'Dejar descansar unas horas y relanzar (reanudable).')
                    break
                time.sleep(PAUSA_S)
    except BloqueoTemporal as e:
        print(f'\n{e}')
        print('todo lo bajado queda guardado. NO relanzar de inmediato: dejar '
              'descansar la IP varias horas (los reintentos profundizan el castigo).')
    except KeyboardInterrupt:
        print('\ninterrumpido con Ctrl+C: todo lo bajado queda guardado (reanudable)')

    horas = (time.time() - t0) / 3600
    ph = paginas / horas if horas > 0 else 0
    registrar(sym, paginas, nuevos, mas_antiguo[:10], 'ok')
    print(f'\n===== piloto {sym} =====')
    print(f'paginas: {paginas:,} | mensajes nuevos: {nuevos:,} (total en disco: {ya + nuevos:,})')
    print(f'NUMERO 1 - mensaje mas antiguo alcanzado: {mas_antiguo[:10] or "(sin cambio)"}')
    print(f'NUMERO 2 - velocidad sostenida: {ph:.0f} paginas/hora (~{ph * 30:.0f} mensajes/hora)')
    print(f'archivo: {destino}')
    print('pegar estos numeros en el chat: deciden si la API alcanza 2020 o si '
          'entran los datasets/terceros del plan (vias 2-4).')


if __name__ == '__main__':
    main()
