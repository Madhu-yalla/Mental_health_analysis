import re
import pandas as pd
from flask import Flask, render_template, request
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix
import nltk
import seaborn as sns
import matplotlib.pyplot as plt
import time

nltk.download('stopwords')
from nltk.corpus import stopwords

# Initialize Flask app
app = Flask(__name__)

# Mapping for numeric labels to actual mental health conditions
label_mapping = {
    0: 'Stress',
    1: 'Depression',
    2: 'Bipolar disorder',
    3: 'Personality disorder',
    4: 'Anxiety'
}

# Preprocess text function
def preprocess_text(text):
    text = re.sub(r'http\S+', '', text)  # Remove URLs
    text = re.sub(r'[^a-zA-Z\s]', '', text)  # Remove special characters
    text = text.lower()  # Convert to lowercase
    stop_words = set(stopwords.words('english'))
    return ' '.join([word for word in text.split() if word not in stop_words])

# Load and preprocess the dataset
def load_and_preprocess_data():
    print("Loading and preprocessing data...")
    start_time = time.time()

    # Load dataset
    data = pd.read_csv('data_to_be_cleansed.csv')

    # Drop rows with missing text data
    data.dropna(subset=['text'], inplace=True)

    # Preprocess the text column
    data['cleaned_text'] = data['text'].apply(preprocess_text)

    # Extract features and labels (replace 'target' with your actual label column)
    X = data['cleaned_text']
    y = data['target'] 

    end_time = time.time()
    print(f"Data loading and preprocessing completed. Time taken: {end_time - start_time:.2f} seconds.")
    return X, y

# Train and validate the machine learning model
def train_and_validate_model():
    print("Starting model training and validation...")
    start_time = time.time()

    X, y = load_and_preprocess_data()

    # Vectorize text using TF-IDF
    print("Vectorizing text using TF-IDF...")
    tfidf = TfidfVectorizer(max_features=5000)
    X_tfidf = tfidf.fit_transform(X).toarray()

    # Split data into train and test sets
    print("Splitting data into train and test sets...")
    X_train, X_test, y_train, y_test = train_test_split(X_tfidf, y, test_size=0.2, random_state=42)

    # Train a logistic regression model
    print("Training logistic regression model...")
    model = LogisticRegression(max_iter=1000)
    model.fit(X_train, y_train)

    # Test the model on validation data
    print("Testing model on validation data...")
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    print("Accuracy on validation data:", accuracy)
    print("Classification Report:\n", classification_report(y_test, y_pred))

    # plot confusion matrix to see how well the model performs
    print("Plotting confusion matrix...")
    cm = confusion_matrix(y_test, y_pred)
    sns.heatmap(cm, annot=True, cmap='Blues', fmt='g')
    plt.title('Confusion Matrix')
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.savefig('confusion_matrix.png')
    plt.close()

    end_time = time.time()
    print(f"Model training and validation completed. Time taken: {end_time - start_time:.2f} seconds.")
    
    return model, tfidf, accuracy

# Initialize the trained model, TF-IDF vectorizer, and accuracy
model, tfidf, model_accuracy = train_and_validate_model()

# Predict mental health condition
@app.route('/predict', methods=['POST'])
def predict():
    if request.method == 'POST':
        print("Received POST request for prediction.")
        user_input = request.form['comment']

        # Preprocess and vectorize the input text
        cleaned_text = preprocess_text(user_input)
        vectorized_text = tfidf.transform([cleaned_text]).toarray()

        # Predict the mental health condition using the trained model
        predicted_condition_numeric = model.predict(vectorized_text)[0]

        # Map the numeric prediction to the actual mental health condition
        predicted_condition = label_mapping[predicted_condition_numeric]

        print(f"Prediction completed. User input: {user_input}, Predicted condition: {predicted_condition}")

        # Render the results in the HTML template
        return render_template('index.html', prediction=predicted_condition, comment=user_input, accuracy=model_accuracy)

# Home route to display the form
@app.route('/')
def index():
    print("Rendering home page...")
    return render_template('index.html')

if __name__ == "__main__":
    print("Starting Flask app...")
    app.run(debug=True, host='0.0.0.0', port=8080)

