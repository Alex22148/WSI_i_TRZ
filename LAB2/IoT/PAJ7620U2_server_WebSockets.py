import asyncio
import websockets
import json
from gesture_sensor import gesture, get_gesture_dict

gestures = gesture()
gesture.init()

async def send_gesture_name(websocket):
    try:
        while True:
            gesture_num = gestures.return_gesture()
            data = get_gesture_dict(gesture_num)
            await websocket.send(json.dumps(data))  # Send JSON data
            await asyncio.sleep(0.02)  # 50Hz update rate (~20ms)
    except websockets.exceptions.ConnectionClosed as e:
        print("Client disconnected")

async def main():
    serwer_ip = "192.168.50.76"
    """Starts WebSocket server on Windows."""
    server = await websockets.serve(send_gesture_name, serwer_ip, 8765)
    print(f"WebSocket Server Started on ws://{serwer_ip}:8765")
    await server.wait_closed()

if __name__ == "__main__":
    # Ensures compatibility with Windows asyncio behavior
    asyncio.run(main())

