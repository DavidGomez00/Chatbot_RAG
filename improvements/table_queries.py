import pandas as pd
import spacy
import json
import os
import torch
from transformers import pipeline, BartTokenizer

# Assurez-vous que PyTorch utilise le CPU
device = torch.device('cpu')

# Charger le modèle de langage de spaCy
nlp = spacy.load("es_core_news_sm")

# Charger le modèle de résumé extractif et forcer l'utilisation du CPU
summarizer = pipeline("summarization", model="sshleifer/distilbart-cnn-12-6", device=device)

# Charger le tokeniseur pour le modèle
tokenizer = BartTokenizer.from_pretrained("sshleifer/distilbart-cnn-12-6")

# Lire le fichier CSV existant
df = pd.read_csv('data_ratings.csv')

# Chemin vers le dossier contenant les fichiers JSON
json_folder = '/home/maxime/gesida-rag-huil/results/nemotron/BGE-M3/nat'

# Liste pour stocker les réponses
answers = []

# Parcourir les fichiers JSON dans le dossier
for i in range(78):  # De 0 à 77
    json_file = os.path.join(json_folder, f'query_{i}.json')
    try:
        with open(json_file, 'r', encoding='utf-8') as file:
            data = json.load(file)
            answers.append(data['answer'])
    except FileNotFoundError:
        answers.append(None)  # Ajouter None si le fichier n'est pas trouvé
answers=answers+answers
# Ajouter les réponses au DataFrame
df['Answer'] = answers

# Fonction pour nettoyer le texte
def clean_text(text):
    # Supprimer les espaces supplémentaires et les caractères spéciaux
    return ' '.join(text.split())

# Fonction pour tronquer le texte à une longueur maximale de tokens
def truncate_to_max_tokens(text, tokenizer, max_length=1024):
    text = clean_text(text)
    tokens = tokenizer.encode(text, truncation=True, max_length=max_length)
    return tokenizer.decode(tokens)

# Fonction pour vérifier si un comment concerne la longueur
def is_length_related(comment):
    if pd.isna(comment):
        return False
    doc = nlp(str(comment))
    length_related_words = ["largo", "corto", "extenso", "breve", "reducir", "acortar", "alargar"]
    return any(token.text in length_related_words for token in doc)

# Fonction pour améliorer les réponses
def improve_response(response, comment):
    if pd.isna(comment) or pd.isna(response):
        return response

    if is_length_related(comment):
        try:
            # Tronquer la réponse à la longueur maximale autorisée
            response_truncated = truncate_to_max_tokens(response, tokenizer)
            # Utiliser le modèle de summary pour raccourcir la réponse
            summary = summarizer(response_truncated, max_length=130, min_length=30, truncation=True)
            new_response = summary[0]['summary_text']
        except Exception as e:
            print(f"Erreur lors du summary: {e}")
            new_response = response
    else:
        new_response = response

    return new_response

# Appliquer la fonction d'amélioration à chaque ligne du DataFrame
df['Améliored Response'] = df.apply(lambda row: improve_response(row['Answer'], row['Comments']), axis=1)

# Sauvegarder le DataFrame avec les réponses améliorées
df.to_csv('improved_responses.csv', index=False)

# Afficher le DataFrame avec les réponses améliorées
print(df.head())
