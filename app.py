import pandas as pd

# Replace with the actual paths to your files
true_path = r"C:\Users\USER\Downloads\fake-news-classifier\True.csv"
fake_path = r"C:\Users\USER\Downloads\fake-news-classifier\Fake.csv"

# Load datasets
true_df = pd.read_csv(true_path)
fake_df = pd.read_csv(fake_path)
# Add labels
true_df['label'] = 'REAL'
fake_df['label'] = 'FAKE'

# Combine into one dataset
news_df = pd.concat([true_df, fake_df])

# Shuffle dataset
news_df = news_df.sample(frac=1).reset_index(drop=True)

# Save as news.csv
news_df.to_csv('news.csv', index=False)
print("Dataset saved as news.csv")
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split

# Load dataset
df = pd.read_csv('news.csv')

# Extract features and labels
X = df['text']
y = df['label']

# Vectorize text data
tfidf = TfidfVectorizer(stop_words='english', max_df=0.7)
X = tfidf.fit_transform(X)

# Split data into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report

# Train logistic regression model
model = LogisticRegression()
model.fit(X_train, y_train)

# Evaluate model
y_pred = model.predict(X_test)
print("Accuracy:", accuracy_score(y_test, y_pred))
print(classification_report(y_test, y_pred))
import pickle

# Save model and vectorizer
with open('model.pkl', 'wb') as model_file:
    pickle.dump(model, model_file)

with open('tfidf.pkl', 'wb') as tfidf_file:
    pickle.dump(tfidf, tfidf_file)
    import streamlit as st
import pickle

# Load the saved model and vectorizer
model = pickle.load(open('model.pkl', 'rb'))
tfidf = pickle.load(open('tfidf.pkl', 'rb'))

# Streamlit interface
st.title("Fake News Classifier")
user_input = st.text_area("Enter news content:")
if st.button("Predict"):
    input_data = tfidf.transform([user_input])
    prediction = model.predict(input_data)
    st.write("The news is:", "Real" if prediction[0] == "REAL" else "Fake")