import matplotlib.pyplot as plt
# import joblib
# import pandas as pd
importances = model.feature_importances_
features = ['N', 'P', 'K', 'temperature', 'humidity', 'ph', 'rainfall']

plt.barh(features, importances)
plt.xlabel('Feature Importance')
plt.title('Global Feature Importance for Crop Prediction')
plt.show()
