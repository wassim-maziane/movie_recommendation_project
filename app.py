import flask
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import difflib

# Importer le module Flask
app = flask.Flask(__name__, template_folder='templates')

# Charger les données à partir du fichier CSV
df2 = pd.read_csv('./model/tmdb.csv')

# Initialiser le vecteur TF-IDF
tfidf = TfidfVectorizer(stop_words='english', analyzer='word')

# Construire la matrice TF-IDF requise en ajustant et transformant les données
tfidf_matrix = tfidf.fit_transform(df2['soup'])
print(tfidf_matrix.shape)

# Construire la matrice de similarité cosinus
cosine_sim = cosine_similarity(tfidf_matrix, tfidf_matrix)
print(cosine_sim.shape)

# Réinitialiser l'index du dataframe
df2 = df2.reset_index()

# Créer une série d'indices pour les titres des films
indices = pd.Series(df2.index, index=df2['title']).drop_duplicates()

# Créer une liste avec tous les titres de films
all_titles = [df2['title'][i] for i in range(len(df2['title']))]

# Fonction pour obtenir les recommandations de films
def get_recommendations(title):
    # Obtenir l'indice du film correspondant au titre
    idx = indices[title]
    # Obtenir les scores de similarité par paire de tous les films avec ce film
    sim_scores = list(enumerate(cosine_sim[idx]))
    # Trier les films en fonction des scores de similarité
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
    # Obtenir les scores des 10 films les plus similaires
    sim_scores = sim_scores[1:11]
    # Obtenir les détails des films recommandés
    movie_indices = [i[0] for i in sim_scores]
    tit = df2['title'].iloc[movie_indices]
    dat = df2['release_date'].iloc[movie_indices]
    rating = df2['vote_average'].iloc[movie_indices]
    moviedetails = df2['overview'].iloc[movie_indices]
    movietypes = df2['keywords'].iloc[movie_indices]
    movieid = df2['id'].iloc[movie_indices]

    # Créer un nouveau dataframe avec les informations des films recommandés
    return_df = pd.DataFrame(columns=['Title', 'Year'])
    return_df['Title'] = tit
    return_df['Year'] = dat
    return_df['Ratings'] = rating
    return_df['Overview'] = moviedetails
    return_df['Types'] = movietypes
    return_df['ID'] = movieid
    return return_df

# Définir la route principale
@app.route("/")
@app.route("/index")
def index():
    return flask.render_template('index.html')

# Définir la route pour les résultats positifs
@app.route('/positive', methods=['GET', 'POST'])
def main():
    if flask.request.method == 'GET':
        return(flask.render_template('index.html'))

    if flask.request.method == 'POST':
        # Obtenir le nom du film saisi dans le formulaire
        m_name = flask.request.form['movie_name']
        m_name = m_name.title().strip()
        if m_name not in all_titles:
            # Si le titre du film n'est pas dans la liste des titres, proposer des suggestions
            suggestions = difflib.get_close_matches(m_name, all_titles)
            return flask.render_template('negative.html', name=m_name, suggestions=suggestions)
        else:
            # Obtenir les recommandations de films basées sur le titre donné
            result_final = get_recommendations(m_name)
            names = []
            dates = []
            ratings = []
            overview = []
            types = []
            mid = []
            for i in range(len(result_final)):
                names.append(result_final.iloc[i][0])
                dates.append(result_final.iloc[i][1])
                ratings.append(result_final.iloc[i][2])
                overview.append(result_final.iloc[i][3])
                types.append(result_final.iloc[i][4])
                mid.append(result_final.iloc[i][5])

            # Rendre le modèle positif avec les résultats des recommandations
            return flask.render_template('positive.html', movie_type=types[5:], movieid=mid,
                                         movie_overview=overview, movie_names=names, movie_date=dates,
                                         movie_ratings=ratings, search_name=m_name)

if __name__ == '__main__':
    # Exécuter l'application Flask sur le serveur local
    app.run(host="127.0.0.1", port=8080, debug=True)
