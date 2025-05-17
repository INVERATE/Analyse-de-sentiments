import requests

r = requests.post("http://127.0.0.1:8000/predict", json={"text": "very medium product"})
print(r.json())
