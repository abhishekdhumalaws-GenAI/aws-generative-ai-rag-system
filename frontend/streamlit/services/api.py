import requests
from services.config import API_URL, API_KEY

def ask_ai(question):
    response = requests.post(
        API_URL,
        headers={
            "Content-Type": "application/json",
            "x-api-key": API_KEY
        },
        json={"query": question},
        timeout=120
    )
    return response.json()

