import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score

# Load the diabetes dataset
@st.cache_data
def load_data():
    diabetes_data = pd.read_csv("C:\\Users\\ABHAY\\anaconda3\\diabetes.csv")
    return diabetes_data

# Function to train and evaluate the k-NN model
def train_evaluate_model(data):
    # Split the dataset into features (X) and target variable (y)
    X = data.drop(columns=['Outcome'])  # Features
    y = data['Outcome']  # Target variable

    # Splitting the dataset into the training set and test set
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Feature scaling
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # Initialize the k-NN classifier
    k = 5  # Number of neighbors to consider
    knn_classifier = KNeighborsClassifier(n_neighbors=k)

    # Train the classifier
    knn_classifier.fit(X_train_scaled, y_train)

    # Predict the test set results
    y_pred = knn_classifier.predict(X_test_scaled)

    # Evaluate the model
    accuracy = accuracy_score(y_test, y_pred)
    accuracy_percentage = accuracy * 100
    st.write("Accuracy:", "{:.2f}%".format(accuracy_percentage))

    return knn_classifier, scaler

# Predict diabetes for new data
def predict_diabetes(model, scaler, data):
    # Scale the input features
    data_scaled = scaler.transform(data)

    # Predict using the trained model
    prediction = model.predict(data_scaled)

    return prediction

# Visualize the distribution of diabetes risk levels
def visualize_diabetes_distribution(data):
    st.write("Distribution of Diabetes Risk Levels:")
    fig, ax = plt.subplots()
    sns.countplot(x='Outcome', data=data, ax=ax)
    st.pyplot(fig)

# Visualize correlations between features and diabetes risk
def visualize_feature_correlation(data):
    st.write("Correlation between Features and Diabetes Risk:")
    corr = data.corr()
    fig, ax = plt.subplots(figsize=(10, 8))  # Adjust figure size as needed
    sns.heatmap(corr, annot=True, cmap='coolwarm', fmt=".2f", ax=ax, annot_kws={"size": 10})
    st.pyplot(fig)

# Identify patterns and trends in patient data
def visualize_pair_plot(data):
    st.write("Pair Plot of Features:")
    fig = sns.pairplot(data, hue='Outcome')
    st.pyplot(fig)
    
# Visualize boxplot for each feature
def visualize_boxplot(data):
    st.write("Boxplot of Features:")
    fig, ax = plt.subplots(1, 2, figsize=(20, 8))  # Create subplots for two boxplots side by side
    sns.boxplot(x='Glucose', data=data, ax=ax[0])  # Boxplot for Glucose
    ax[0].set_title('Boxplot of Glucose')
    sns.boxplot(x='Pregnancies', data=data, ax=ax[1])  # Boxplot for Pregnancies
    ax[1].set_title('Boxplot of Pregnancies')
    st.pyplot(fig)




# Main function to run the Streamlit app
def main():
    # Load the data
    diabetes_data = load_data()
    
    # Sidebar options
    st.sidebar.title("NAVIGATE")

    option = st.sidebar.radio("Select Option", ["Home", "View Dataset", "Visualization", "Prediction"])

    if option == "Home":
        st.title("DIABETES RISK PREDICTION")
        st.image("https://assets.clevelandclinic.org/transform/LargeFeatureImage/f52ef1a5-1310-4cec-bc56-c1759042d1f7/glucometer-diabetes-stress-1433938010", 
         caption="Understanding Diabetes Risk", 
         width=200,  # Adjust width as needed
         use_column_width=True)
        
        st.write("Diabetes Risk Prediction Dashboard offers a multifaceted approach to understanding and mitigating the risk of diabetes through machine learning methodologies.This comprehensive platform delves into an array of health metrics, ranging from pregnancies and glucose levels to blood pressure, skin thickness, insulin levels, BMI, diabetes pedigree function, and age. By amalgamating these diverse data points, the dashboard facilitates an insightful exploration of diabetes risk factors, providing users with a nuanced understanding of their health profile.")
        st.write("Through data visualization tools, users can discern patterns, trends, and correlations within the dataset, gaining invaluable insights into the complex interplay of various health parameters in predisposing individuals to diabetes. Furthermore, leveraging machine learning techniques, the dashboard empowers users to make informed predictions regarding their diabetes risk, enabling proactive measures for prevention and management.")
        st.write("Ultimately, the Diabetes Risk Prediction Dashboard serves as a pivotal tool in the arsenal against diabetes, fostering awareness, education, and proactive healthcare practices. By equipping individuals with the knowledge and tools to assess and mitigate their risk, it aims to facilitate early intervention, healthier lifestyles, and improved overall well-being.")

        


    elif option == "View Dataset":
        st.title("DIABETES DATASET")
        st.write("<b>Pregnancies:</b> This feature represents the number of times an individual has been pregnant.",unsafe_allow_html=True)
        st.write("<b>Glucose:</b> This represents the plasma glucose concentration at 2 hours in an oral glucose tolerance test.", unsafe_allow_html=True)   
        st.write("<b>BloodPressure:</b> This feature signifies diastolic blood pressure (in mm Hg).",unsafe_allow_html=True)
        st.write("<b>SkinThickness:</b> This refers to the triceps skin fold thickness (in mm).",unsafe_allow_html=True)
        st.write("<b>Insulin:</b> This is the 2-Hour serum insulin (in mu U/ml).",unsafe_allow_html=True)
        st.write("<b>BMI: </b>This feature is the Body Mass Index (weight in kg/(height in m)^2).",unsafe_allow_html=True)
        st.write("<b>DiabetesPedigreeFunction:</b> This is a function that scores likelihood of diabetes based on family history.",unsafe_allow_html=True)
        st.write("<b>Age:</b> This represents age in years.",unsafe_allow_html=True)
        st.write("<b>Outcome:</b> This is the class variable (0 or 1).",unsafe_allow_html=True)
     
        st.write("Viewing Diabetes Dataset:")
        st.write(diabetes_data)
        
    elif option == "Visualization":
        st.title("VISUALIZATION")
        visualization_option = st.sidebar.selectbox("Choose Visualization", ["Distribution of Diabetes Risk Levels", "Correlation between Features and Diabetes Risk", "Pair Plot of Features", "Boxplot of Features"])

        if visualization_option == "Distribution of Diabetes Risk Levels":
            visualize_diabetes_distribution(diabetes_data)
        elif visualization_option == "Correlation between Features and Diabetes Risk":
            visualize_feature_correlation(diabetes_data)
        elif visualization_option == "Pair Plot of Features":
            visualize_pair_plot(diabetes_data)
        elif visualization_option == "Boxplot of Features":
            visualize_boxplot(diabetes_data) 
            
    elif option == "Prediction":
        st.title("PREDICTION")
        # Train the model
        st.write(diabetes_data)
        st.write("Training the Model...")
        model, scaler = train_evaluate_model(diabetes_data)
        st.write("Model trained successfully!")

        # Input features for prediction
        st.sidebar.subheader("Predict Diabetes")
        pregnancies = st.sidebar.number_input("Pregnancies", min_value=0, max_value=17, value=0)
        glucose = st.sidebar.number_input("Glucose", min_value=0, max_value=199, value=0)
        blood_pressure = st.sidebar.number_input("Blood Pressure", min_value=0, max_value=122, value=0)
        skin_thickness = st.sidebar.number_input("Skin Thickness", min_value=0, max_value=99, value=0)
        insulin = st.sidebar.number_input("Insulin", min_value=0, max_value=846, value=0)
        bmi = st.sidebar.number_input("BMI", min_value=0.0, max_value=67.1, value=0.0)
        diabetes_pedigree_function = st.sidebar.number_input("Diabetes Pedigree Function", min_value=0.078, max_value=2.42, value=0.078)
        age = st.sidebar.number_input("Age", min_value=21, max_value=81, value=21)

        # Create a dataframe with input features
        input_data = pd.DataFrame({
            "Pregnancies": [pregnancies],
            "Glucose": [glucose],
            "BloodPressure": [blood_pressure],
            "SkinThickness": [skin_thickness],
            "Insulin": [insulin],
            "BMI": [bmi],
            "DiabetesPedigreeFunction": [diabetes_pedigree_function],
            "Age": [age]
        })

        # Predict diabetes
        if st.sidebar.button("Predict"):
            prediction = predict_diabetes(model, scaler, input_data)
            if prediction[0] == 1:
                st.write("The person is diabetic.")
            else:
                st.write("The person is not diabetic.")

# Run the main function
if __name__ == "__main__":
    main()