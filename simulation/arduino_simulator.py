import requests
import time
import random

# URL de votre API FastAPI (locale ou Docker)
API_URL = "http://localhost:8000/predict"

def simulate_sensor_data():
    while True:
        # Génération de données réalistes
        payload = {
            "soil_moisture": round(random.uniform(10, 60), 2),
            "temperature": round(random.uniform(20, 45), 2),
            "humidity": round(random.uniform(30, 70), 2),
            "rainfall": round(random.uniform(0, 5), 2),
            "soil_type": "Sandy",
            "crop_type": "Cotton"
        }

        try:
            response = requests.post(API_URL, json=payload)
            if response.status_code == 200:
                result = response.json()
                print(f"Données envoyées : {payload['soil_moisture']}% humidité")
                print(f"Décision : {result['label']} | Pompe : {result['statut_pompe']}")
            else:
                print(f"Erreur API : {response.status_code}")
        except Exception as e:
            print(f"Connexion impossible à l'API : {e}")

        # Pause de 5 secondes entre chaque lecture
        time.sleep(5)

if __name__ == "__main__":
    simulate_sensor_data()