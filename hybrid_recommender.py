import joblib
import re
import nltk
import numpy as np
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize

# Load all pre-trained models and the vectorizer
print("--- Loading SVD and SVM models ---")
try:
    svd_model = joblib.load('SVD_model.pkl')
    svm_payload = joblib.load('SVM_model.pkl')
    svm_model = svm_payload['model']
    tfidf_vectorizer = svm_payload['vectorizer']
    print("All models and vectorizer loaded successfully.")
except FileNotFoundError as e:
    print(f"Error loading models: {e}. Make sure 'SVD_model.pkl' and 'SVM_model.pkl' are in the correct directory.")
    svd_model = svm_model = tfidf_vectorizer = None

# Initialize text processing tools
try:
    lemmatizer = WordNetLemmatizer()
    stop_words = set(stopwords.words('english'))
except LookupError:
    print("NLTK data not found. Downloading...")
    nltk.download('punkt')
    nltk.download('stopwords')
    nltk.download('wordnet')
    lemmatizer = WordNetLemmatizer()
    stop_words = set(stopwords.words('english'))

# Define the text cleaning function
def clean_text(text):
    """A single function to perform all text cleaning steps."""
    if not isinstance(text, str):
        return ""
    text = text.lower()
    text = re.sub(r'[^\x00-\x7F]+', ' ', text)
    text = re.sub(r'\s+', ' ', text).strip()
    tokens = word_tokenize(text)
    lemmatized_tokens = [
        lemmatizer.lemmatize(word) for word in tokens if word not in stop_words and len(word) > 1
    ]
    return ' '.join(lemmatized_tokens)

# Define the main Hybrid Recommender Prediction Function
def predict_hybrid_rating(user_id, hospital_id, review_text, alpha=0.5):
    """
    Calculates a hybrid recommendation rating by combining SVD and SVM.

    Args:
        user_id (str): The author's name or ID.
        hospital_id (str): The hospital's name or ID.
        review_text (str): The text content to be analyzed by the content model.
        alpha (float): The weight for the content-based (SVM) model score.
                       The SVD model weight will be (1 - alpha).

    Returns:
        A dictionary containing the final hybrid rating and individual model scores.
    """
    if not all([svd_model, svm_model, tfidf_vectorizer]):
        return {"error": "Models are not loaded. Cannot make a prediction."}
    
    # === 1. Get SVD (Collaborative Filtering) Prediction ===
    # This is R_SVD, the predicted rating on a 1-5 scale.
    svd_prediction = svd_model.predict(uid=user_id, iid=hospital_id)
    R_SVD = svd_prediction.est

    # === 2. Get SVM (Content-Based) Prediction ===
    # This is P_SVM, the predicted probability on a 0-1 scale.
    processed_text = clean_text(review_text)
    vectorized_text = tfidf_vectorizer.transform([processed_text])
    P_SVM = svm_model.predict_proba(vectorized_text)[0][1]

    # === 3. Convert SVM Probability to a 1-5 Rating Scale ===
    # This step creates R_SVM by mapping the [0, 1] probability to the [1, 5] rating scale.
    R_SVM = (P_SVM * 4.0) + 1.0

    # === 4. Calculate the Final Hybrid Score using a Direct Weighted Average ===
    # Both R_SVM and R_SVD are now on the same 1-5 scale.
    final_predicted_rating = (alpha * R_SVM) + ((1 - alpha) * R_SVD)

    # === 5. Final Clipping (Safety Net) ===
    # Ensures the final score is strictly within the 1-5 range.
    final_predicted_rating = np.clip(final_predicted_rating, 1, 5)

    # === 6. Explanation Generation ===

        # === XAI LOGIC ===
    # Get the words that actually influenced the SVM prediction
    feature_names = tfidf_vectorizer.get_feature_names_out()
    vectorized_text = tfidf_vectorizer.transform([clean_text(review_text)])
    # Find words present in the input that have the highest weight in the model
    # (Simplified XAI: showing which input words the TF-IDF recognized)
    keywords_detected = []
    feature_index = vectorized_text.nonzero()[1]
    for i in feature_index:
        keywords_detected.append(feature_names[i])

    explanation = {
        "logic": f"This score is {alpha*100}% based on your symptom description and {(1-alpha)*100}% based on historical patient ratings.",
        "top_features": keywords_detected[:3], # Show the top 3 words detected
        "svd_contribution": "High" if R_SVD > 3.5 else "Moderate",
        "svm_contribution": "High" if R_SVM > 3.5 else "Moderate"
    }

    return {
        'user_id': user_id,
        'hospital_id': hospital_id,
        'svd_predicted_rating (R_SVD)': R_SVD,
        'svm_rating_equivalent (R_SVM)': R_SVM,
        'final_hybrid_rating': final_predicted_rating,
        'explanation':explanation
    }
