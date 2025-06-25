import os, json
import requests
from PIL import Image
import matplotlib.pyplot as plt
import numpy as np
from tqdm.notebook import tqdm
import base64
import io
from dotenv import load_dotenv, dotenv_values
from images_func import *

load_dotenv()
config = dotenv_values(".env")

image_dir = "./assets/top_influenceurs_2024/IMG/"
max_images = 3  
api_token = os.getenv("API_TOKEN")
url = "https://api-inference.huggingface.co/models/sayeed99/segformer_b3_clothes"
headers = {
    "Authorization": f"Bearer {api_token}",
    "Content-Type": "image/png"
}
image_paths = [] # A vous de jouer !

for path, _, files in os.walk(image_dir):
    for file in files:
        image_paths.append(f"{path}/{file}")

if not image_paths:
    print(f"Aucune image trouvée dans '{image_dir}'. Veuillez y ajouter des images.")
else:
    print(f"{len(image_paths)} image(s) à traiter")
    
CLASS_MAPPING = {
    "Background": 0,
    "Hat": 1,
    "Hair": 2,
    "Sunglasses": 3,
    "Upper-clothes": 4,
    "Skirt": 5,
    "Pants": 6,
    "Dress": 7,
    "Belt": 8,
    "Left-shoe": 9,
    "Right-shoe": 10,
    "Face": 11,
    "Left-leg": 12,
    "Right-leg": 13,
    "Left-arm": 14,
    "Right-arm": 15,
    "Bag": 16,
    "Scarf": 17
}

if image_paths:
    single_image_path = image_paths[0] # Prenons la première image de notre liste
    print(f"Traitement de l'image : {single_image_path}")

    try:
        image_data = None
        
        with open(single_image_path, "rb") as data:
            image_data = data
            response = requests.post(url=url, data=data, headers=headers)
            
            if response.status_code != 200:
                response.raise_for_status()
                
            print(response.json())

    except Exception as e:
        print(f"Une erreur est survenue : {e}")
else:
    print("Aucune image à traiter. Vérifiez la configuration de 'image_dir' et 'max_images'.")
