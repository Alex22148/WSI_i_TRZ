import asyncio
import websockets

async def hello():
    uri = "ws://192.168.50.78:8765"
    try:
        async with websockets.connect(uri) as websocket:
            print("🔗 Połączono z serwerem!")
            while True:
                msg = await websocket.recv()
                print(f"Odebrano: {msg}")
    except Exception as e:
        print(f"❌ Błąd klienta: {e}")

asyncio.run(hello())

