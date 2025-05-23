import pandas as pd
from flask import Flask, render_template, request
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

# Initialize Flask app
app = Flask(__name__)

# Load the SBERT model for sentence embeddings
model = SentenceTransformer('all-MiniLM-L6-v2')

# Load the book and rating datasets
books = pd.read_csv('goodbooks-10k-master/books.csv')
ratings = pd.read_csv('goodbooks-10k-master/ratings.csv')

# Example: Ensure books contains 'book_id' and 'title'
books = books[['book_id', 'title', 'authors']]

# Get embeddings for book titles
book_titles = books['title'].tolist()
book_embeddings = model.encode(book_titles)

# Function to recommend books based on user query
def recommend_books_dynamic(query, books, book_embeddings, top_n=5):
    # Encode the user query
    query_embedding = model.encode([query])
    
    # Compute cosine similarities between query and all book titles
    similarities = cosine_similarity(query_embedding, book_embeddings)
    
    # Get the indices of the most similar books
    similar_indices = similarities.argsort()[0][-top_n:][::-1]
    
    # Return top recommended books
    recommended_books = books.iloc[similar_indices]
    
    return recommended_books

# Define the route for the web app
@app.route('/')
def home():
    return render_template('index.html')

# Define the route for the search and recommendations
@app.route('/recommend', methods=['POST'])
def recommend():
    query = request.form['query']
    
    # Get recommendations based on the query
    recommended_books = recommend_books_dynamic(query, books, book_embeddings)
    
    # Prepare data for display
    recommended_books_data = recommended_books[['title', 'authors']].to_dict(orient='records')
    
    return render_template('index.html', query=query, recommended_books=recommended_books_data)

# Run the web app
if __name__ == '__main__':
    app.run(debug=True)

books = pd.read_csv('/Users/mohammadrafiquekuwari/Projects/Codsoft/task4/goodbooks-10k-master/books.csv')
ratings = pd.read_csv('/Users/mohammadrafiquekuwari/Projects/Codsoft/task4/goodbooks-10k-master/ratings.csv')



# use python rs.py to run the flask app