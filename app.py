import streamlit as st
import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
from sklearn.ensemble import RandomForestClassifier

data = pd.read_csv('food.csv')

Breakfastdata = data['Breakfast']
BreakfastdataNumpy = Breakfastdata.to_numpy()
Lunchdata = data['Lunch']
LunchdataNumpy = Lunchdata.to_numpy()
Dinnerdata = data['Dinner']
DinnerdataNumpy = Dinnerdata.to_numpy()
Food_itemsdata = data['Food_items']
def calculate_bmi(weight, height):
   return weight / ((height / 100) ** 2)

def cluster_and_predict(age, veg, weight, height, mode):
   bmi = calculate_bmi(weight, height)
   agecl = round(age / 20)
   clbmi = 0
   if bmi < 16:
       clbmi = 4
   elif bmi < 18.5:
       clbmi = 3
   elif bmi < 25:
       clbmi = 2
   elif bmi < 30:
       clbmi = 1
   else:
       clbmi = 0
   # Data separation (Breakfast, Lunch, Dinner)
   breakfastfoodseparatedID = [i for i, val in enumerate(BreakfastdataNumpy) if val == 1]
   LunchfoodseparatedID = [i for i, val in enumerate(LunchdataNumpy) if val == 1]
   DinnerfoodseparatedID = [i for i, val in enumerate(DinnerdataNumpy) if val == 1]

   # Get relevant rows for each meal
   breakfastfoodseparatedIDdata = data.iloc[breakfastfoodseparatedID].T.iloc[5:15].T
   LunchfoodseparatedIDdata = data.iloc[LunchfoodseparatedID].T.iloc[5:15].T
   DinnerfoodseparatedIDdata = data.iloc[DinnerfoodseparatedID].T.iloc[5:15].T

   # Prepare data for KMeans clustering
   ti = (clbmi + agecl) / 2
   # Simplified logic for predictions (this avoids empty predictions)
   if mode == "Weight Loss":
       # Dummy prediction logic for demonstration
       labels = KMeans(n_clusters=3, random_state=0).fit(DinnerfoodseparatedIDdata).labels_
       # For simplicity, predict based on these clusters
       y_pred = labels
   else:
       y_pred = np.random.randint(0, 3, len(Food_itemsdata))  # Random mock predictions for demo
   # Suggest food items based on predictions
   suggested_foods = [Food_itemsdata[i] for i, pred in enumerate(y_pred) if pred == 2]  # Selecting cluster 2
   if not suggested_foods:
       return "No food items to suggest."
   return f"Suggested food items: {', '.join(suggested_foods)}"

def weight_loss(age, veg, weight, height):
   return cluster_and_predict(age, veg, weight, height, mode="Weight Loss")

def weight_gain(age, veg, weight, height):
   return cluster_and_predict(age, veg, weight, height, mode="Weight Gain")

def healthy(age, veg, weight, height):
   return cluster_and_predict(age, veg, weight, height, mode="Healthy")

# Streamlit Interface
with st.Blocks(css="""
   .streamlit-container {background-color: #E6E6FA;}
   .st-button {background-color: #c468c6 !important; color: #431c44 !important; font-weight: bold !important;}
""") as demo:
   age = st.Number(label="Age", value=25)
   veg = st.Radio(label="Veg-NonVeg", choices=["Veg", "NonVeg"], value="Veg")
   weight = st.Number(label="Weight (kg)", value=60)
   height = st.Number(label="Height (cm)", value=170)
   btn_weight_loss = st.Button("Weight Loss")
   btn_weight_gain = st.Button("Weight Gain")
   btn_healthy = st.Button("Balanced Diet")
   output = st.Textbox(label="Recommendations")
   btn_weight_loss.click(fn=weight_loss, inputs=[age, veg, weight, height], outputs=output)
   btn_weight_gain.click(fn=weight_gain, inputs=[age, veg, weight, height], outputs=output)
   btn_healthy.click(fn=healthy, inputs=[age, veg, weight, height], outputs=output)

demo.launch(share=True)


