# InsureSmart-Personalized-Policy-Recommendation-System
InsureSmart is an AI-driven system that recommends personalized insurance policies based on customer profiles like age, income, health, and lifestyle. It streamlines decision-making, boosts satisfaction, and helps insurers offer tailored, data-backed coverage options.

# InsureSmart – Personalized Policy Recommendation System

InsureSmart is an AI-powered Streamlit web application that recommends personalized insurance policies based on individual user profiles. By leveraging synthetic data and a machine learning model (Random Forest Classifier), the system provides real-time, data-driven insurance policy suggestions tailored to user characteristics such as age, income, smoking habits, dependents, occupation, lifestyle, and previous policies.

## Features

- Interactive Streamlit UI for user-friendly data entry
- Machine learning-based personalized policy recommendation
- Synthetic data generation to simulate realistic scenarios
- Top two policy recommendations with matching scores
- Encodes categorical variables using LabelEncoder
- Fast and responsive backend using scikit-learn's Random Forest

## How It Works

1. **Data Generation**: A synthetic dataset is created with customer details such as age, income, occupation, lifestyle, and insurance history.
2. **Policy Assignment**: A rule-based function assigns a suitable policy label to each record.
3. **Model Training**: The data is used to train a Random Forest Classifier to predict recommended policies.
4. **User Input**: The user fills in a form with their personal and professional details.
5. **Prediction**: The trained model predicts the top two most suitable insurance policies.

## Tech Stack

- **Frontend/UI**: Streamlit
- **Backend/ML**: Python, scikit-learn
- **Data Handling**: Pandas, NumPy
- **Model**: Random Forest Classifier
- **Encoding**: LabelEncoder from sklearn

## Installation
Step 1: Set Up Virtual Environment
bash
Copy
Edit
python -m venv venv
# On Windows:
venv\Scripts\activate
# On macOS/Linux:
source venv/bin/activate

Step 2: Install Dependencies
pip install -r requirements.txt
streamlit
pandas
numpy
scikit-learn

Running the Application
streamlit run app.py

Sample User Input
Age: 35
Income: ₹12,00,000
Smoker: No
Dependents: 2
Occupation: Software Developer
Previous Policy: Health Insurance
Lifestyle Score: 4

Sample Output
Recommendations:
• Recommendation 1: Comprehensive Family Floater Plan G (Matching score: 92%)
• Recommendation 2: Life + Accidental Combo Plan B (Matching score: 88%)
