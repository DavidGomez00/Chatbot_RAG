

import os
import json

def read_json_files(directory):
    '''Read JSON files from a directory and return a list of queries and answers.'''
    queries_answers = []
    for filename in os.listdir(directory):
        if filename.endswith('.json'):
            filepath = os.path.join(directory, filename)
            with open(filepath, 'r') as f:
                data = json.load(f)
                queries_answers.append((data['query'], data['answer']))
    return queries_answers

def generate_html_form(queries_answers, output_file):
    '''Generate an HTML form from a list of queries and answers.'''
    html_content = """
    <!DOCTYPE html>
    <html lang="en">
    <head>
        <meta charset="UTF-8">
        <meta name="viewport" content="width=device-width, initial-scale=1.0">
        <title>Chatbot Validation Form</title>
        <style>
            body { font-family: Arial, sans-serif; margin: 20px; }
            .query { margin-bottom: 20px; }
            .query p { font-weight: bold; }
            .rating { margin: 10px 0; }
            .rating label { margin-right: 10px; }
        </style>
    </head>
    <body>
        <h1>Chatbot Validation Form</h1>
        <form>
    """

    for idx, (query, answer) in enumerate(queries_answers):
        html_content += f"""
            <div class="query">
                <p>Query {idx + 1}: {query}</p>
                <p>Answer: {answer}</p>
                <div class="rating">
                    <label>Truthfulness Rating:</label>
                    <select name="truthfulness_rating_{idx}">
                        <option value="" selected disabled>Select a rating</option>
                        <option value="1">1 - Deficiente</option>
                        <option value="2">2</option>
                        <option value="3">3</option>
                        <option value="4">4</option>
                        <option value="5">5 - Excelente</option>
                    </select>
                </div>
                <div class="rating">
                    <label>Completeness Rating:</label>
                    <select name="completeness_rating_{idx}">
                        <option value="" selected disabled>Select a rating</option>
                        <option value="1">1 - Deficiente</option>
                        <option value="2">2</option>
                        <option value="3">3</option>
                        <option value="4">4</option>
                        <option value="5">5 - Excelente</option>
                    </select>
                </div>
            </div>
        """

    html_content += """
        </form>
    </body>
    </html>
    """

    with open(output_file, 'w') as f:
        f.write(html_content)

if __name__ == "__main__":
    # Chemin vers le répertoire contenant les fichiers JSON
    json_directory = "results/nemotron/BGE-M3/nat-sem"
    output_file = "validation_form.html"

    # Lire les fichiers JSON
    queries_answers = read_json_files(json_directory)

    # Générer le formulaire HTML
    generate_html_form(queries_answers, output_file)

    print(f"Form generated and saved to {output_file}")
