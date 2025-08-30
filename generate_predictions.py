import pandas as pd
import joblib

# Load your batch input data
input_df = pd.read_csv('test_farm.csv')
features=['N','P','K','temperature','humidity','ph','rainfall']
# Load the trained model and label encoder
model = joblib.load('crop_recommender.pkl')
le = joblib.load('label_encoder.pkl')

# Predict for all rows in the dataset

pred_encoded = model.predict(input_df[features])
pred_crops = le.inverse_transform(pred_encoded)

# Add predictions to the dataframe
input_df['predicted_crop'] = pred_crops

# Save results to CSV
input_df.to_csv('predictions_output.csv', index=False)

print("Predictions saved to predictions_output.csv")
