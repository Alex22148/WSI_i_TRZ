import time
import board
import busio
import adafruit_bno08x
from adafruit_bno08x.i2c import BNO08X_I2C
from adafruit_bno08x import BNO_REPORT_ROTATION_VECTOR

i2c = busio.I2C(board.SCL, board.SDA)
bno = BNO08X_I2C(i2c)

bno.enable_feature(BNO_REPORT_ROTATION_VECTOR)

def get_orientation():
    try:
        quat = bno.quaternion  # Pobieranie kwaternionu
        if quat is None:
            raise ValueError("Kwaternion nie został zwrócony.")
    
        return {
            "w": quat[0],  # Część rzeczywista
            "x": quat[1],  # Oś X
            "y": quat[2],  # Oś Y
            "z": quat[3]   # Oś Z
        }

    except KeyError as e:
        print(f"Nieznany raport: {e}. Ignoruję raport i kontynuuję.")
        return None  # Zwracanie None lub domyślnych wartości
    except ValueError as e:
        print(f"Błąd podczas odczytu quaternion: {e}")
        return None  # Zwracanie None lub domyślnych wartości
    except OSError as e:
        print(f"Błąd I/O: {e}. Sprawdź połączenie z urządzeniem.")
        return None  # Możesz spróbować ponownie lub zakończyć program
'''
if __name__ == "__main__":
    while True:
        print(get_orientation())  # Print orientation data
        time.sleep(0.05)  # Adjust refresh rate'''
