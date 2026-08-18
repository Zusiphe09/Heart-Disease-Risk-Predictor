<img src="https://cdn.prod.website-files.com/677c400686e724409a5a7409/6790ad949cf622dc8dcd9fe4_nextwork-logo-leather.svg" alt="NextWork" width="300" />

# Heart Disease Risk Predictor

**Project Link:** [View Project](https://nextwork.ai/projects/9b8fd880-ade2-4b67-b2e4-c8f9b4b8b580)

**Author:** Inga Nguse  
**Email:** inganguse09@gmail.com

---

![Image](https://nextwork.ai/proud_blue_vibrant_quince/uploads/9b8fd880-ade2-4b67-b2e4-c8f9b4b8b580_8eevmkpd)

## Building a Heart Disease Risk Predictor

### Project goals and motivation

In this project, I'm building a machine learning web application that predicts a patient's heart disease risk using health information such as cholesterol levels, blood pressure, heart rate, and other medical data. The app uses a trained Random Forest model to generate instant predictions, display risk probabilities, classify risk levels, and visualize the most important contributing factors.

This is more useful than a terminal script because it provides an easy-to-use browser interface that anyone can access without programming knowledge. Healthcare professionals or users can enter patient data through a form, receive immediate results, view charts and explanations, and interact with the model in a more intuitive and user-friendly way than running commands in a terminal.

## Training the Machine Learning Model

### Loading data and fitting the classifier

In this step, I'm training a Random Forest machine learning model using the heart disease dataset and evaluating its performance on unseen data.

So that I can verify that the dataset is loaded correctly, confirm the model can learn meaningful patterns from the patient health records, measure its baseline accuracy, and test a sample prediction before building the web interface. This ensures the core prediction system is working reliably before it is integrated into the Streamlit application.

![Image](https://nextwork.ai/proud_blue_vibrant_quince/uploads/9b8fd880-ade2-4b67-b2e4-c8f9b4b8b580_65p4aszo)

### Model accuracy and the limits of raw output

My model achieved 83.61% accuracy. The probability array is not useful on its own because it only shows the model's confidence scores for each class. Without the corresponding class labels and context, we cannot tell what the probabilities represent or how they translate into the final prediction. In this[0.75 0.0** (no heart disease ** class is therefore 0, because it has the higher probability.

the on its own because the numbers only represent the model's confidence for each class. class each probability corresponds to, or understanding the class labels and prediction threshold, the values have little meaning and cannot be interpreted correctly. 83.61% accuracy own because it only shows the model's confidence the probabilities represent or how they translate into example, [0.75, 0.25] means there is a 75% chance of class 0 and a 25% chance of class 1, resulting in a prediction of class 0.

## Creating the Interactive Prediction Interface

### Building the patient data form

In this step, I'm building a Streamlit web application so that users can enter patient health information through a simple interface and receive a heart disease risk prediction without having to run Python code or interpret terminal output. The app will load and train the model, collect patient data using form inputs, and display the raw prediction result when the user clicks the Predict button.

![Image](https://nextwork.ai/proud_blue_vibrant_quince/uploads/9b8fd880-ade2-4b67-b2e4-c8f9b4b8b580_tvuko3c1)

### Why raw predictions fail health app users

The bare prediction is a problem because users cannot tell what the numbers mean, whether the result indicates high or low heart disease risk, or how confident the model is in its prediction. A simple "0" or "1" provides no explanation, context, or guidance, which can be confusing and unhelpful in a health application where users need clear and understandable information.

## Transforming Predictions into Meaningful Health Insights

### Adding probability scores, risk bands, and visualizations

In this step, I'm transforming the raw 0/1 prediction into a more informative health risk assessment by showing the model's confidence as a percentage, assigning a risk level (Low, Moderate, or High), and visualizing the most important factors with a feature importance chart. This helps users better understand the prediction, how reliable it is, and which health indicators contributed most to the result. Additionally, model accuracy and dataset information will be displayed in the sidebar to provide transparency about the model's performance.

### Understanding feature importance

The top features in my chart are the variables with the highest feature importance scores, particularly the feature with an importance of about 0.18 and the next highest features around 0.14 and 0.11. This means the model relies most heavily on these health factors when predicting heart disease risk because they have the greatest influence on the model's decision-making process. Features with lower importance scores still contribute to the prediction, but they have less impact than the top-ranked factors.

## Comparing Multiple ML Algorithms

![Image](https://nextwork.ai/proud_blue_vibrant_quince/uploads/9b8fd880-ade2-4b67-b2e4-c8f9b4b8b580_bxts1yeo)

### Which algorithm performed best and why

In this project extension, the best performer was Logistic Regression with an accuracy of 85.2%, compared to Random Forest and Gradient Boosting, which both achieved 83.6% accuracy. Logistic Regression outperformed the other two models by 1.6 percentage points.

## Reflections and Key Takeaways

### Tools and concepts mastered

The key tools I used include Python, pandas, NumPy, scikit-learn, Streamlit, and a Python virtual environment (venv). Pandas was used for data loading and preparation, NumPy for numerical operations, scikit-learn for building and evaluating machine learning models, and Streamlit for creating an interactive web application for predictions.Key concepts I learnt include data preprocessing, machine learning model training, model evaluation using accuracy scores, probability-based predictions, feature importance analysis, risk classification, and comparing multiple algorithms such as Logistic Regression, Random Forest, and Gradient Boosting. I also learned how to transform technical model outputs into user-friendly insights by displaying risk levels, confidence percentages, and visual charts that make predictions easier to understand.

### Time and challenges

This project took me approximately 5 hours to complete. The most challenging part was building the Streamlit web application and connecting it to the machine learning model so that user inputs could be processed correctly for predictions. Another challenge was transforming the raw prediction output into a meaningful result by adding probability scores, risk levels, and a feature importance chart. However, overcoming these challenges helped me gain a better understanding of machine learning deployment, data preprocessing, and creating user-friendly interfaces for healthcare applications.

### Looking ahead

I did this project today to learn how to build and deploy a machine learning application, train and evaluate predictive models, compare different algorithms, and create a user-friendly web interface with Streamlit. I also learned how to turn raw model outputs into meaningful insights using probabilities, risk levels, and feature importance visualizations.

Another skill I want to learn is how to deploy machine learning applications to the cloud so that they can be accessed online by multiple users. I would also like to learn more about improving model performance through feature engineering, hyperparameter tuning, and working with larger real-world datasets.

---

*Built with [NextWork](https://nextwork.ai) - [View this project](https://nextwork.ai/projects/9b8fd880-ade2-4b67-b2e4-c8f9b4b8b580)*
