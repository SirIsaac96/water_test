# Necessary libraries
from fastapi import FastAPI
import pandas as pd
import pickle
from data_model import WaterPotability

# Initialize instance FastAPI with name app
app = FastAPI(
    title="Water Potability Prediction",
    description="Predicting Water Potability",
)

# Load the pre-trained model
with open("model.pkl", "rb") as model_file:
    model = pickle.load(model_file)

# Create a root endpoint (first endpoint) - acts as the home page of the API
@app.get("/")
def index():
    return "Welcome to Water Potability Prediction FastAPI"

# Create a prediction endpoint
@app.post("/predict")
def model_predict(water: WaterPotability):
    sample = pd.DataFrame(
        {
            "ph": [water.ph],
            "Hardness": [water.Hardness],
            "Solids": [water.Solids],
            "Chloramines": [water.Chloramines],
            "Sulfate": [water.Sulfate],
            "Conductivity": [water.Conductivity],
            "Organic_carbon": [water.Organic_carbon],
            "Trihalomethanes": [water.Trihalomethanes],
            "Turbidity": [water.Turbidity]
        }
    )
    # Perform prediction on the data entered by the user, using the pre-trained model
    predicted_value = model.predict(sample)

    if predicted_value[0] == 1:
        return "Water is Consumable"
    else:

        return "Water is Not Consumable"
