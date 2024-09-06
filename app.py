from flask import Flask, request, render_template
import pandas as pd
import joblib
from sklearn.preprocessing import StandardScaler, PolynomialFeatures
import firebase_admin
from firebase_admin import credentials, storage
from io import BytesIO

# Initialize Firebase app with your credentials
cred = credentials.Certificate('firebase_key.json')  # Path to your Firebase service account key
firebase_admin.initialize_app(cred, {
    'storageBucket': 'emotion-predictor-96406.appspot.com'  # Replace with your Firebase project ID
})

# Function to load model from Firebase Storage
def load_model_from_firebase(file_name):
    bucket = storage.bucket()
    blob = bucket.blob(file_name)
    model_data = blob.download_as_bytes()
    model = joblib.load(BytesIO(model_data))
    return model

# Load the models dynamically from Firebase
best_model = load_model_from_firebase('random_forest_model.pkl')
scaler = load_model_from_firebase('scaler.pkl')
poly = load_model_from_firebase('poly_transform.pkl')
model_columns = load_model_from_firebase('model_columns.pkl')

# Initialize Flask app
app = Flask(__name__)

@app.route('/')
def index():
    return render_template('index.html', result=None, error=None)

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Extract input data from form
        age = float(request.form['age'])
        usage_time = float(request.form['usage_time'])
        posts_per_day = float(request.form['posts_per_day'])
        likes_per_day = float(request.form['likes_per_day'])
        comments_per_day = float(request.form['comments_per_day'])
        messages_per_day = float(request.form['messages_per_day'])
        gender = request.form['gender']
        platform = request.form['platform']

        # Build the feature dictionary with input values
        features = {
            'Age': age,
            'Daily_Usage_Time (minutes)': usage_time,
            'Posts_Per_Day': posts_per_day,
            'Likes_Received_Per_Day': likes_per_day,
            'Comments_Received_Per_Day': comments_per_day,
            'Messages_Sent_Per_Day': messages_per_day,
            'Gender': gender,
            'Platform': platform
        }

        # Convert features into a DataFrame for one-hot encoding
        input_df = pd.DataFrame([features])
        input_df = pd.get_dummies(input_df, columns=['Gender', 'Platform'], drop_first=True)

        # Ensure all columns are present
        for col in model_columns:
            if col not in input_df.columns:
                input_df[col] = 0

        # Reorder the columns to match the training set
        input_df = input_df[model_columns]

        # Scale and transform the input features
        X_scaled = scaler.transform(input_df)
        X_poly = poly.transform(X_scaled)

        # Make the prediction
        prediction = best_model.predict(X_poly)

        # Define the emotions list before accessing it
        emotions = ['Neutral', 'Happiness', 'Anxiety', 'Sadness', 'Boredom', 'Anger', 'Aggression']

        # Check if prediction is valid
        if prediction[0] < 0 or prediction[0] >= len(emotions):
            return render_template('index.html', result=None, error="Prediction index out of range.")

        # Map prediction to emotion
        predicted_emotion = emotions[prediction[0]]

        return render_template('index.html', result=f'Predicted Emotion: {predicted_emotion}', error=None)

    except ValueError:
        return render_template('index.html', result=None, error="Invalid input, please check your values.")
    except Exception as e:
        print(f"Error: {e}")
        return render_template('index.html', result=None, error="An error occurred while processing the prediction.")

if __name__ == '__main__':
    app.run(debug=True)
