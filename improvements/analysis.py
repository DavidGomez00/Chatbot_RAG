## Le script permet de séparer les requêtes qui ont une note moyenne inférieure à 3
# des requêtes qui ont une note moyenne supérieure ou égale à 3.

import os
import pandas as pd
import json

# Charger le fichier CSV
df = pd.read_csv('data_ratings.csv')
df = df.loc[:, ['Query', 'Truthfulness Rating', 'Completeness Rating']]

# Afficher les premières lignes du DataFrame
#print(df.head())

###################################
# Chargement des questions et des réponses.

# Chemin vers le fichier texte dans un autre dossier
file_path = os.path.join('..', '/home/maxime/gesida-rag-huil/', 'nuevas_preguntas.txt')

# Lire le contenu du fichier texte
with open(file_path, 'r', encoding='utf-8') as file:
    questions = file.readlines()

# Nettoyer les questions en supprimant les sauts de ligne
questions = [q.strip() for q in questions]

# et pour les réponses, on va lire les fichiers JSON dans le dossier results/nemotron/BGE-M3/nat-sem

def read_json_files(directory):
    '''Read JSON files from a directory and return a list of queries and answers.'''
    answers = []
    for filename in os.listdir(directory):
        if filename.endswith('.json'):
            filepath = os.path.join(directory, filename)
            with open(filepath, 'r') as f:
                data = json.load(f)
                answers.append((data['answer']))
    return answers

json_directory = "/home/maxime/gesida-rag-huil/results/nemotron/BGE-M3/nat-sem"

# Lire les fichiers JSON

answers = read_json_files(json_directory)

##################################


# Ajouter une nouvelle colonne au DataFrame avec les questions correspondantes
df['Question Text'] = df['Query'].apply(lambda x: questions[(x-1) % 78]) #modulo 78 car il y a 78 questions mais 2*78 ratings (un par user)
df['Answers'] = df['Query'].apply(lambda x: answers[(x-1) % 78])
df2 = pd.read_csv('data_ratings.csv')

# Ajouter une nouvelle colonne au DataFrame avec les comments correspondants
df['Comments'] = df2['Comments']

average_ratings_per_query = df.groupby('Query')[['Truthfulness Rating', 'Completeness Rating']].mean()

# Calculate the overall mean rating for each query
average_ratings_per_query['Mean Rating'] = average_ratings_per_query.mean(axis=1)
average_ratings_per_query["Query"] = df['Query']
average_ratings_per_query['Comments'] = df['Comments']
average_ratings_per_query['Question Text'] = df['Question Text']
average_ratings_per_query['Answers'] = df['Answers']


# Filter queries where the mean rating is strictly less than 3
low_rated_queries = average_ratings_per_query[average_ratings_per_query['Mean Rating'] < 3]

high_rated_queries = average_ratings_per_query[average_ratings_per_query['Mean Rating'] >= 3]

print(low_rated_queries.head())
print(high_rated_queries.head())

#df.to_csv('data_ratings_v2.csv', index=False)
low_rated_queries.to_csv('low_rated_queries.csv', index=False)
high_rated_queries.to_csv('high_rated_queries.csv', index=False)
