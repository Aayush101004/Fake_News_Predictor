# 📰 Fake News Detector (Deep Learning & Wikipedia Search)
This is a Streamlit web application that leverages a deep learning model to classify news article titles as either "Real" or "Fake." To aid in verification, it integrates with the Wikipedia API to provide summaries and links for related information.
## ✨ Features
- Deep Learning Model: Utilizes a TensorFlow/Keras deep learning model (Sequential with Embedding and Global Average Pooling layers) for text classification.
- Text Preprocessing: Includes stemming and stopword removal for cleaner text features.
- Interactive Web Interface: Built with Streamlit for an easy-to-use and responsive user experience.
- Wikipedia Search Integration: Fetches and displays summaries and links from Wikipedia based on the input news title.
- Google Search Fallback: If no Wikipedia summary is found, it provides a direct Google search link for further investigation.
- Confidence Scores: Displays the model's confidence for both "Real" and "Fake" predictions.
- Confusion Matrix Output: Prints the model's confusion matrix to the terminal upon startup for quick evaluation.
- Cached Performance: Uses Streamlit's caching (@st.cache_data, @st.cache_resource) to speed up data loading, preprocessing, and model training.
## 🚀 Technologies Used
- Python 3.x
- Streamlit: For building the web application.
- TensorFlow/Keras: For the deep learning model.
- Pandas: For data manipulation.
- NLTK: For natural language processing (stemming, stopwords).
- Wikipedia: For searching and retrieving summaries from Wikipedia.
- Scikit-learn: For data splitting and confusion matrix.
## 🛠️ Setup Instructions
Follow these steps to get the application up and running on your local machine.
1. Clone the Repository
First, clone this GitHub repository to your local machine:
git clone https://github.com/Aayush101004/Fake_News_Predictor.git
cd Fake_News_Predictor
2. Create a Virtual Environment
It's highly recommended to use a virtual environment to manage dependencies:
python -m venv my_project_env
3. Activate the Virtual Environment
On Windows:
.\my_project_env\Scripts\activate
On macOS/Linux:
source my_project_env/bin/activate
4. Install Dependencies
Install all required Python packages. You can create a requirements.txt file by running pip freeze > requirements.txt after installing them manually, or install them directly:
pip install pandas numpy streamlit tensorflow nltk wikipedia scikit-learn
5. Download NLTK Stopwords
NLTK requires a one-time download of the 'stopwords' corpus:
python -c "import nltk; nltk.download('stopwords')"
6. Place Data Files
Ensure True.csv and Fake.csv are in the same directory as app.py. These files contain your training data.
7. Run the Streamlit Application
With your virtual environment active, run the app:
streamlit run app.py
Your browser should automatically open to the Streamlit application (usually at http://localhost:8501).
## 💡 How to Use
- Enter News Title: Type or paste a news article title into the text area.
- Analyze: Click the "Analyze News" button.
- View Prediction: The app will display whether the news is likely "REAL" or "FAKE" along with a confidence score.
- Explore Related Information: The app will attempt to fetch a summary from Wikipedia related to your input. If successful, it will display the summary and a link to the Wikipedia page. If not, it will provide a direct Google search link for further investigation.
## ⚠️ Disclaimer
This model is for demonstrative and educational purposes only and may not be 100% accurate. The accuracy of the predictions heavily depends on the quality, diversity, and size of the training dataset. Always verify information from multiple reliable sources.
## 📈 Model Evaluation (Terminal Output)
Upon starting the Streamlit application, the confusion matrix for the model's performance on the test set will be printed directly to your terminal. This provides a quick overview of how well the model is classifying real vs. fake news.
## 📚 Further Improvements
- Larger/More Diverse Dataset: Training on a significantly larger and more varied dataset would greatly improve accuracy.
- Advanced NLP Techniques: Explore more sophisticated text embeddings (e.g., Word2Vec, GloVe, FastText) or pre-trained language models (e.g., BERT, RoBERTa) for feature extraction.
- More Complex Deep Learning Architectures: Experiment with Bidirectional LSTMs, GRUs, or Transformer-based models.
- Hyperparameter Tuning: Use techniques like GridSearchCV or RandomizedSearchCV to find optimal model hyperparameters.
- Full Article Analysis: Extend the model to analyze the full text of articles, not just titles.
- Deployment: Deploy the application to cloud platforms like Streamlit Cloud, Vercel, or AWS for public access.
