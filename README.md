# ANN-Classification-Customer-Churn

## Overview
This project aims to predict whether a customer will churn or not using an Artificial Neural Network (ANN). The model is trained on customer data and deployed using **Streamlit**, providing an interactive web interface for predictions.

## Features
- **Deep Learning Model:** Built with TensorFlow/Keras.
- **Data Preprocessing:** Uses **StandardScaler**, **OneHotEncoder**, and **LabelEncoder**.
- **User-friendly Web App:** Developed with **Streamlit**.
- **Real-time Predictions:** Enter customer details and get an instant churn prediction.

## Technologies Used
- **Python** (pandas, numpy, tensorflow, scikit-learn, pickle, streamlit)
- **Machine Learning:** Deep Neural Networks (DNN)
- **Deployment:** Streamlit Web App

## Installation
1. Clone the repository:
   ```bash
   git clone https://github.com/bassamejaz/ANN-Classification-Customer-Churn.git
   cd ANN-Classification-Customer-Churn
   ```
2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
3. Run the Streamlit app:
   ```bash
   streamlit run app.py
   ```

## How It Works
1. **User Input:** The user provides customer details such as age, balance, credit score, tenure, etc.
2. **Preprocessing:**
   - Categorical features (Gender, Geography) are encoded.
   - Numerical features are scaled using **StandardScaler**.
3. **Model Prediction:**
   - The preprocessed input is fed into the trained **ANN model**.
   - The model outputs a probability of churn.
   - If the probability > 0.5, the customer is predicted to churn; otherwise, they will not churn.

## Files and Directories
- **model.h5** → Pre-trained ANN model
- **label_encoder_gender.pkl** → Label encoder for gender
- **onehot_encoder_geo.pkl** → One-hot encoder for geography
- **scalar.pkl** → Standard scaler for feature normalization
- **app.py** → Streamlit app script

## Example Usage
1. Run the app using Streamlit.
2. Select customer attributes such as **Geography, Gender, Age, Balance, Credit Score, Estimated Salary, etc.**
3. Click on **Predict** to get the probability of churn.
4. The app will display whether the customer is likely to churn or not.

## Future Improvements
- Adding more customer behavior features for better accuracy.
- Implementing model explainability with SHAP or LIME.
- Deploying the app on a cloud platform (AWS, GCP, Heroku).

## Author
**Bassam Ejaz**  
For queries, reach out via [GitHub](https://github.com/bassamejaz) or email.

## License
This project is licensed under the MIT License.

