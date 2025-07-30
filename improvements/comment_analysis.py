import pandas as pd

df = pd.read_csv('high_rated_queries.csv')

df = df.dropna(subset=['Comments']).copy() # Supprimer les lignes avec des commentaires vides
# car on ne va pas les améliorer.

# Supprimer les colonnes score1 et score2
df = df.drop(columns=['Truthfulness Rating', 'Completeness Rating'])

# Fonction pour analyser les commentaires et améliorer les réponses

# Fonction pour catégoriser les commentaires
def categorize_comment(comment):
    comment = comment.lower()
    if any(word in comment for word in ['largo', 'larga', 'extenso', 'acortar', 'reducir', 'corto', 'mas corta']):
        return 'Shorten the response'
    elif any(word in comment for word in ['error', 'obsoleto', 'actualizada', 'antiguas', 'incorrecto']):
        return 'Regenerate response'
    elif any(word in comment for word in ['confuso', 'difusa', 'inespecífico', 'dispersión', 'complicado']):
        return 'Clarify the response'
    elif any(word in comment for word in ['explicación', 'detalles']):
        return 'Provide more details'
    else:
        return 'No specific improvement'

df['improvement_suggestion'] = df['Comments'].apply(categorize_comment)

# Affichage des suggestions d'amélioration
print(df[['Question Text', 'Answers', 'Mean Rating', 'Comments', 'improvement_suggestion']])

verification = pd.DataFrame()
verification['Comments'] = df['Comments']
verification['improvement_suggestion'] = df['improvement_suggestion']
verification.to_csv('improvement_suggestions.csv', index=False)
