#  Flight Ticket Price Prediction using Machine Learning
##  Project Overview.

### This project predicts flight ticket prices using Machine Learning Regression algorithms. The system analyzes various flight details such as airline, source, destination, duration, and total stops to estimate ticket prices accurately.

### The project follows the complete Machine Learning lifecycle, including data preprocessing, exploratory data analysis (EDA), model training, evaluation, pipeline creation, and deployment using Streamlit.

##  Objectives
### * Analyze factors affecting flight ticket prices.
### * Perform data cleaning and preprocessing.
### * Build and compare multiple regression models.
### * Select the best-performing model.
### * Create an automated machine learning pipeline.
### * Deploy the model using Streamlit for real-time predictions.

##  Dataset Features
### Airline
### Source
### Destination
### Total Stops
### Duration
### Journey Day
### Journey Month
### Price (Target Variable)

##  Exploratory Data Analysis (EDA)

## The following visualizations were used:

### * Bar Charts
### * Histograms
### * Box Plots
### * Scatter Plots
### * Correlation Heatmaps
### * Distribution Plots
## Key Insights
### Flight duration significantly impacts ticket prices.
### Flights with more stops generally cost more.
### Ticket prices vary across airlines.
### Seasonal and date-related factors influence prices.

##  Machine Learning Models Used
### 1- Linear Regression
### 2- Decision Tree Regressor
### 3- Random Forest Regressor
### 4-Gradient Boosting Regressor
### 5-K-Nearest Neighbors (KNN) Regressor

##  Model Performance
### Model	R² Score
### Linear Regression	0.44
### Decision Tree Regressor	0.81
### Random Forest Regressor	0.86
### Gradient Boosting Regressor	0.84
### KNN Regressor	0.57
###  Best Model

### Random Forest Regressor achieved the highest prediction accuracy and was selected as the final model.

##  Pipeline Integration

### A complete machine learning pipeline was created to automate:

### Data preprocessing
### Encoding categorical variables
### Feature transformation
### Model prediction

## The final pipeline was saved as:

### "flight_price_pipeline.pkl"

##  Streamlit Deployment

### The trained model was deployed using Streamlit.

## Application Workflow
### User Inputs Flight Details
            ↓
     Streamlit Interface
            ↓
      Saved ML Pipeline
            ↓
      Data Preprocessing
            ↓
 ### Random Forest Prediction
            ↓
   ### Display Ticket Price
## User Inputs
### Airline
### Source
### Destination
### Total Stops
### Duration
### Journey Day
### Journey Month
### Output
### Predicted Ticket Price: ₹XXXX
##  Technologies Used
### Python
### Pandas
### NumPy
### Matplotlib
### Seaborn
### Scikit-Learn
### Joblib
### Streamlit
### Jupyter Notebook

##  Project Structure
### Flight-Ticket-Price-Prediction/
│
├── Flight_Ticket_Price_Prediction.ipynb
├── app.py
├── flight_price_pipeline.pkl
├── dataset.csv
├── requirements.txt
└── README.md
##  How to Run
### Install Required Libraries
### pip install -r requirements.txt
### Run Streamlit App
### streamlit run app.py
## Open Browser
### (http://localhost:8507)

streamli cloud community 
https://ceq75jknsylmsignrsniq2.streamlit.app/

##  Conclusion

### This project successfully developed a Flight Ticket Price Prediction system using Machine Learning. Five regression algorithms were evaluated, and Random Forest Regressor achieved the highest accuracy. The final model was integrated into an automated pipeline and deployed using Streamlit, enabling users to obtain real-time flight ticket price predictions through an interactive web application. This project demonstrates practical machine learning implementation from data preprocessing to deployment in a real-world environment.

### GitHub Topics (Tags)
### machine-learning
### regression
### flight-price-prediction
### random-forest
### streamlit
### python
### data-science
