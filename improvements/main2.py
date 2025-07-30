import pandas as pd
import requests
import os
from dotenv import load_dotenv

# Charger les variables d'environnement
load_dotenv()

df = pd.read_csv('data_ratings_v2.csv')

df.drop(columns=['Truthfulness Rating', 'Completeness Rating'])

# Fonction pour envoyer une requête au serveur Ollama
def query_ollama_server(comment, response):
    print(f"Préparation de la requête pour le commentaire: {comment}")

    # Récupérer l'adresse IP du serveur Ollama depuis les variables d'environnement
    ollama_server_ip = os.getenv("OLLAMA_SERVER_IP")
    if not ollama_server_ip:
        raise ValueError("L'adresse IP du serveur Ollama n'est pas définie dans les variables d'environnement.")

    print(f"Adresse IP du serveur Ollama récupérée: {ollama_server_ip}")

    # Structure de la requête
    payload = {
        "model": "nemotron",  # Assurez-vous que ce modèle est disponible sur votre serveur
        "messages": [
            {"role": "system", "content": "Eres un asistente especializado en mejorar las respuestas del chatbot basándose en los comentarios."},
            {"role": "user", "content": f"Commentaire: {comment}\nRespuesta inicial: {response}\nMejorar la respuesta de acuerdo con el comentario:"}
        ],
        "stream": False
    }

    print("Envoi de la requête au serveur Ollama...")
    # Envoyer la requête à l'API Ollama
    try:
        response = requests.post(
            url=f"http://{ollama_server_ip}/api/chat",
            json=payload
        )
        response.raise_for_status()  # Vérifie les erreurs HTTP
        print("Requête envoyée avec succès, réception de la réponse...")

        # Vérifier si la requête a réussi
        if response.status_code == 200:
            improved_response = response.json().get("message", {}).get("content", response.text)
            print("Réponse améliorée reçue.")
            return improved_response
        else:
            print(f"Erreur lors de la réception de la réponse: {response.status_code}")
            return response.text
    except requests.exceptions.RequestException as e:
        print(f"Erreur lors de l'envoi de la requête: {e}")
        return str(e)

# Appliquer la fonction pour chaque paire commentaire/réponse
print("Début du traitement des commentaires et réponses...")
df['Improved_Response'] = df.apply(lambda row: query_ollama_server(row['Comments'], row['Answers']), axis=1)
print("Toutes les requêtes ont été traitées.")

# Afficher le DataFrame mis à jour
print("\nDataFrame final avec réponses améliorées:")
print(df)

df.to_csv('data_ratings_v2_improved.csv', index=False)