import ast
import pickle
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from nltk.stem.porter import PorterStemmer

def load_data():
    movies = pd.read_csv('tmdb_5000_movies.csv')
    credit = pd.read_csv('tmdb_5000_credits.csv')
    movies = movies.merge(credit, on='title')
    movies = movies[['movie_id', 'title', 'overview', 'genres', 'keywords', 'cast', 'crew']]
    movies.dropna(inplace=True)
    return movies

def preprocess_data(movies):
    def convert(obj):
        L = []
        for i in ast.literal_eval(obj):
            L.append(i['name'])
        return L

    def convert3(obj):
        L = []
        counter = 0
        for i in ast.literal_eval(obj):
            if counter != 3:
                L.append(i['name'])
                counter += 1
            else:
                break
        return L

    def fetch_director(obj):
        L = []
        for i in ast.literal_eval(obj):
            if i['job'] == 'Director':
                L.append(i['name'])
                break
        return L

    def stem(text):
        ps = PorterStemmer()
        y = []
        for i in text.split():
            y.append(ps.stem(i))
        return " ".join(y)

    movies['genres'] = movies['genres'].apply(convert)
    movies['keywords'] = movies['keywords'].apply(convert)
    movies['cast'] = movies['cast'].apply(convert3)
    movies['crew'] = movies['crew'].apply(fetch_director)
    movies['overview'] = movies['overview'].apply(lambda x: x.split())
    movies['tags'] = movies['overview'] + movies['genres'] + movies['keywords'] + movies['cast'] + movies['crew']
    movies['tags'] = movies['tags'].apply(lambda x: " ".join(x))
    movies['tags'] = movies['tags'].apply(lambda x: x.lower())
    movies['tags'] = movies['tags'].apply(stem)
    return movies

def generate_similarity_matrix(movies):
    cv = CountVectorizer(max_features=5000, stop_words='english')
    vectors = cv.fit_transform(movies['tags']).toarray()
    similarity = cosine_similarity(vectors)
    return similarity

def recommend(movie_input, movies, similarity):

    movie_input = movie_input.lower()

    movies['title_lower'] = movies['title'].str.lower()

    movie_index = movies[movies['title_lower'] == movie_input].index

    if len(movie_index) == 0:
        return []  # Return empty list if movie not found

    movie_index = movie_index[0]  # Get first index if multiple matches
    distances = similarity[movie_index]
    movies_list = sorted(list(enumerate(distances)), reverse=True, key=lambda x: x[1])[1:6]
    recommendations = [movies.iloc[i[0]].title for i in movies_list]
    return recommendations

