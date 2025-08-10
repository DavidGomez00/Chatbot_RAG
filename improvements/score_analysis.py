import pandas as pd

# Charger le fichier CSV
file_path = 'data_ratings.csv'
data = pd.read_csv(file_path)

# Séparer les données en deux groupes en fonction des deux notateurs
# Supposons que les notateurs soient identifiés par une colonne 'Notateur'
group1 = data[data['User'] == 0]
group2 = data[data['User'] == 1]

# Fonction pour calculer les statistiques descriptives
def calculate_statistics(df, group_name):
    stats = {
        'Moyenne Truthfulness': df['Truthfulness Rating'].mean(),
        'Écart-type Truthfulness': df['Truthfulness Rating'].std(),
        'Moyenne Completeness': df['Completeness Rating'].mean(),
        'Écart-type Completeness': df['Completeness Rating'].std()
    }
    print(f"Statistiques pour le groupe {group_name}:")
    for stat, value in stats.items():
        print(f"{stat}: {value:.2f}")
    print("\n")

# Calculer les statistiques pour chaque groupe
calculate_statistics(group1, 'User 0')
calculate_statistics(group2, 'User 1')

# Calculer les statistiques pour l'ensemble des données
calculate_statistics(data, 'Tous les notateurs')
