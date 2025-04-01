import requests
import time

raspberry_pi_ip = "192.168.50.76"  # Change this to your actual IP
url = f"http://{raspberry_pi_ip}:5000/gesture"# orientation
print(url)
response = requests.get(url)
data = response.json()
while True:
    response = requests.get(url)
    data = response.json()
    if data != "nothing":
        print(data)

