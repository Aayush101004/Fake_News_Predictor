# FAKE-NEWS-DETECTOR
## 📰 Fake News Detector
This is a Streamlit web application that leverages a deep learning model to classify news article titles as either "Real" or "Fake." It also integrates with the NewsAPI to provide real-time search results for similar news or fact-checks, enhancing the verification process.
## ✨ Features
- Deep Learning Model: Utilizes a TensorFlow/Keras deep learning model (Sequential with Embedding and Global Average Pooling layers) for text classification.
- Text Preprocessing: Includes stemming and stopword removal for cleaner text features.
- Interactive Web Interface: Built with Streamlit for an easy-to-use and responsive user experience.
- Real-time News Search (NewsAPI):
  - If the news is predicted as REAL, it fetches and displays similar news articles from the web using NewsAPI.
  - If the news is predicted as FAKE, it provides a Google search link for fact-checking the given title.
- Confidence Scores: Displays the model's confidence for both "Real" and "Fake" predictions.
- Confusion Matrix Output: Prints the model's confusion matrix to the terminal upon startup for quick evaluation.
- Cached Performance: Uses Streamlit's caching (@st.cache_data, @st.cache_resource) to speed up data loading, preprocessing, and model training.
## 🚀 Technologies Used
- Python 3.x
- Streamlit: For building the web application.
- TensorFlow/Keras: For the deep learning model.
- Pandas: For data manipulation.
- NLTK: For natural language processing (stemming, stopwords).
- Requests: For making API calls to NewsAPI.
- Scikit-learn: For data splitting and confusion matrix.
## 🛠️ Setup Instructions
Follow these steps to get the application up and running on your local machine.
- Clone the Repository
First, clone this GitHub repository to your local machine:
git clone https://github.com/Aayush101004/FAKE-NEWS-DETECTOR.git
cd FAKE-NEWS-DETECTOR
- Create a Virtual Environment
It's highly recommended to use a virtual environment to manage dependencies:
python -m venv my_project_env
- Activate the Virtual Environment
On Windows:
.\my_project_env\Scripts\activate
On macOS/Linux:
source my_project_env/bin/activate
- Install Dependencies
Install all required Python packages:
pip install -r requirements.txt
(If you don't have a requirements.txt yet, create one by running pip freeze > requirements.txt after installing all dependencies manually, or install them one by one: pip install pandas numpy streamlit tensorflow nltk requests scikit-learn)
- Download NLTK Stopwords
NLTK requires a one-time download of the 'stopwords' corpus:
python -c "import nltk; nltk.download('stopwords')"
- Configure NewsAPI Key (Crucial for Search Functionality)
The application uses the NewsAPI for real-time news search. You need to obtain a free API key and set it as an environment variable.
  - Get your NewsAPI Key:
    - Go to https://newsapi.org/ and sign up for a free developer account.
    - Your API key will be available on your dashboard. Copy it.
  - Set the API Key as an Environment Variable:
  Before running the app, set the NEWS_API_KEY environment variable in your terminal.
    - On Windows (Command Prompt):
    bash set NEWS_API_KEY=YOUR_NEWS_API_KEY_HERE
    - On Windows (PowerShell):
    powershell $env:NEWS_API_KEY="YOUR_NEWS_API_KEY_HERE"
    - On macOS/Linux (Bash/Zsh):
    bash export NEWS_API_KEY="YOUR_NEWS_API_KEY_HERE"
  (Replace YOUR_NEWS_API_KEY_HERE with your actual NewsAPI key. This variable needs to be set in every new terminal session or added to your shell's profile for persistence.)
- Place Data Files
Ensure True.csv and False.csv are in the same directory as app.py. These files contain your training data. Take any datasets that you find.
- Run the Streamlit Application
With your virtual environment active and the API key set, run the app:
streamlit run app.py
Your browser should automatically open to the Streamlit application (usually at http://localhost:8501).
## 💡 How to Use
- Enter News Title: Type or paste a news article title into the text area.
- Analyze: Click the "Analyze News" button.
- View Prediction: The app will display whether the news is likely "REAL" or "FAKE" along with a confidence score.
- Explore Related News/Fact Checks:
  - If predicted as REAL, it will show similar news articles from NewsAPI and provide a Google search link.
  - If predicted as FAKE, it will provide a Google search link for fact-checking the title.
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

Deployment: Deploy the application to cloud platforms like Streamlit Cloud, Heroku, or AWS for public access.
