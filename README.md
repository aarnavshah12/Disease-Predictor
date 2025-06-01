# Disease-Predictor
This project uses machine learning and logic to predict the user's disease based on their clicked symptoms.

# Disease Prediction System

This Disease Prediction System uses machine learning models to predict possible diseases based on user-provided symptoms. The application provides a graphical user interface (GUI) for easy interaction and includes features such as model selection, user authentication, a searchable symptom list, and integration with online medical resources.

## Features

![Screenshot 2024-07-31 003758](https://github.com/user-attachments/assets/2ab7b941-32d1-45c5-8248-86c318f5e63e)

1. **Model Selection**: Choose between Random Forest, Decision Tree, or Naive Bayes for prediction.
2. **User Authentication**: Login system to save user history and provide personalized experiences.
3. **Help Section**: Instructions on how to use the application and interpret results.
4. **Searchable Symptom List**: Search bar to filter the list of symptoms.
5. **Integration with Medical Resources**: Link predictions to online medical resources for more information.
6. **Feedback System**: Submit feedback on user experience and suggestions for improvement.
7. **Save/Load Symptoms**: Save your symptoms to a file and load them later.

## Installation

1. Clone the repository:
    ```bash
    git clone <repository-url>
    ```
2. Navigate to the project directory:
    ```bash
    cd disease-prediction-system
    ```

## Usage

1. Run the application:
    ```bash
    python Disease_Prediction.py
    ```
2. Login with the following credentials:
    - **Username**: `admin`
    - **Password**: `password`
3. Select your symptoms from the list and choose the prediction model.
4. Click "Predict Disease" to see the results.
5. Save your symptoms using "Save Input" and load them using "Load Input".
6. Click "Open Medical Resource" to learn more about the predicted diseases.
7. Provide feedback using the feedback section.

## Code Overview

- **disease_prediction_gui.py**: Main script to run the GUI application.
- **Data.csv**: Dataset used for training the machine learning models.
- **requirements.txt**: List of required Python packages.

## Contributing

1. Fork the repository.
2. Create a new branch:
    ```bash
    git checkout -b feature-branch
    ```
3. Make your changes and commit them:
    ```bash
    git commit -m "Description of changes"
    ```
4. Push to the branch:
    ```bash
    git push origin feature-branch
    ```
5. Open a pull request.

## License

This project is licensed under the MIT License.

## Acknowledgements

- Inspired by various open-source disease prediction projects.
- This project isn't 100% accurate due to the limitation of training and testing data. Therefore, it is still being worked on.

## Contact

For questions or feedback, please contact aarnav.shah.12@gmail.com.

