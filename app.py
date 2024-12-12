import streamlit as st
import pandas as pd
import numpy as np
from sklearn.cluster import KMeans

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

    data = pd.read_csv('food.csv')
    Breakfastdata = data['Breakfast'].to_numpy()
    Lunchdata = data['Lunch'].to_numpy()
    Dinnerdata = data['Dinner'].to_numpy()
    Food_itemsdata = data['Food_items']

    # Data separation (Breakfast, Lunch, Dinner)
    breakfastfoodseparatedID = [i for i, val in enumerate(Breakfastdata) if val == 1]
    LunchfoodseparatedID = [i for i, val in enumerate(Lunchdata) if val == 1]
    DinnerfoodseparatedID = [i for i, val in enumerate(Dinnerdata) if val == 1]

    breakfastfoodseparatedIDdata = data.iloc[breakfastfoodseparatedID].T.iloc[5:15].T
    LunchfoodseparatedIDdata = data.iloc[LunchfoodseparatedID].T.iloc[5:15].T
    DinnerfoodseparatedIDdata = data.iloc[DinnerfoodseparatedID].T.iloc[5:15].T

    ti = (clbmi + agecl) / 2

    if mode == "Weight Loss":
        labels = KMeans(n_clusters=3, random_state=0).fit(DinnerfoodseparatedIDdata).labels_
        y_pred = labels
    else:
        y_pred = np.random.randint(0, 3, len(Food_itemsdata))

    suggested_foods = [Food_itemsdata[i] for i, pred in enumerate(y_pred) if pred == 2]
    if not suggested_foods:
        return pd.DataFrame({"Message": ["No food items to suggest."]})
    return pd.DataFrame({"Suggested Food Items": suggested_foods})

def main():
    st.title("Personalized Diet Recommendation")

    st.sidebar.header("User Input Parameters")
    age = st.sidebar.number_input("Age", min_value=1, max_value=120, value=25)
    veg = st.sidebar.radio("Veg or Non-Veg", ("Veg", "NonVeg"), index=0)
    weight = st.sidebar.number_input("Weight (kg)", min_value=1, max_value=200, value=60)
    height = st.sidebar.number_input("Height (cm)", min_value=50, max_value=250, value=170)

    mode = st.selectbox("Select Mode", ("Weight Loss", "Weight Gain", "Balanced Diet"))

    if st.button("Get Recommendations"):
        recommendations = cluster_and_predict(age, veg, weight, height, mode)
        st.table(recommendations)

if __name__ == "__main__":
    main()
