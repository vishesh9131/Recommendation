import streamlit as st
from predict import load_data, preprocess_data, generate_similarity_matrix, recommend
import ast


movies = load_data()
movies = preprocess_data(movies)

similarity = generate_similarity_matrix(movies)

def main():
    st.title('Movie Recommendation System')
    st.title('Predict')

    movie_input = st.text_input('Enter a movie title:')
    
    if st.button('Predict'):
        recommendations = recommend(movie_input, movies, similarity)
        if recommendations:
            st.write('Recommended Movies:')
            for movie in recommendations:
                st.write(movie)
        else:
            st.write('No recommendations found.')

if __name__ == '__main__':
    main()
