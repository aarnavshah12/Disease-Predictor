import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import classification_report
from tkinter import *
from tkinter import messagebox, ttk
import webbrowser

# Load and Prepare Data
# Importing the dataset
dataset = pd.read_csv('ML Models\\Disease Prediction\\Data.csv')

# Dropping the unnecessary column
dataset = dataset.drop(columns=['Unnamed: 133'])

# Encoding Categorical Data: Dependent Variable
le = LabelEncoder()
dataset['prognosis'] = le.fit_transform(dataset['prognosis'])

# Splitting into independent and dependent variables
X = dataset.iloc[:, :-1].values
y = dataset.iloc[:, -1].values

X = np.nan_to_num(X)
y = np.nan_to_num(y)

# Splitting the dataset into the Training set and Test set
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=24)

# Feature scaling
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)

# Training the models
rf_classifier = RandomForestClassifier(n_estimators=10, criterion='entropy', random_state=18)
rf_classifier.fit(X_train, y_train)

dt_classifier = DecisionTreeClassifier(criterion='entropy', random_state=0)
dt_classifier.fit(X_train, y_train)

nb_classifier = GaussianNB()
nb_classifier.fit(X_train, y_train)

# User Authentication (simple implementation)
def login():
    username = username_entry.get()
    password = password_entry.get()
    if username == "admin" and password == "password":
        login_frame.pack_forget()
        main_frame.pack(fill='both', expand=True)
    else:
        messagebox.showerror("Login Error", "Invalid username or password")

def efficiency_report(model):
    if model == 'Random Forest':
        print("Random Forest Classification Report")
        print(classification_report(y_test, rf_classifier.predict(X_test)))
    elif model == 'Decision Tree':
        print("Decision Tree Classification Report")
        print(classification_report(y_test, dt_classifier.predict(X_test)))
    elif model == 'Naive Bayes':
        print("Naive Bayes Classification Report")
        print(classification_report(y_test, nb_classifier.predict(X_test)))

def final_answer(rf_pred, dt_pred, nb_pred):
    rf_disease_name = le.inverse_transform([rf_pred[0]])
    dt_disease_name = le.inverse_transform([dt_pred[0]])
    nb_disease_name = le.inverse_transform([nb_pred[0]])

    return rf_disease_name[0], dt_disease_name[0], nb_disease_name[0]

def find_consecutive_ones(lst):
    positions = []
    for i in range(len(lst) - 1):
        if lst[i] == 1 and lst[i + 1] == 1:
            positions.append((i, i + 1))
    return positions

def compare_consecutive_ones(lst, dataset):
    user_positions = find_consecutive_ones(lst)
    matched_prognoses = set()
    for index, row in dataset.iterrows():
        row_array = row[:-1].values
        for start, end in user_positions:
            if np.array_equal(row_array[start:end+1], np.array(lst[start:end+1])):
                matched_prognoses.add(row['prognosis'])
                break
    return list(matched_prognoses)

# GUI Application
def predict_disease():
    user_input = [symptom_listbox.get(idx) for idx in symptom_listbox.curselection()]
    x = [symptom.lower() for symptom in user_input]
    global lst
    lst = [0] * 132
    symptom_names = dataset.columns.tolist()[:-1]

    try:
        for symptom in x:
            symptom_index = symptom_names.index(symptom)
            lst[symptom_index] = 1

        results = compare_consecutive_ones(lst, dataset)
        if len(results) > 0:
            results = le.inverse_transform(results)
            result_label.config(text=f"Prognosis: {', '.join(results)}")
        else:
            selected_model = model_var.get()
            if selected_model == 'Random Forest':
                prediction = rf_classifier.predict(sc.transform([lst])).astype(int)
            elif selected_model == 'Decision Tree':
                prediction = dt_classifier.predict(sc.transform([lst])).astype(int)
            elif selected_model == 'Naive Bayes':
                prediction = nb_classifier.predict(sc.transform([lst])).astype(int)

            efficiency_report(selected_model)
            rf_result, dt_result, nb_result = final_answer(prediction, prediction, prediction)
            result_label.config(text=f"{selected_model}: {rf_result}")

    except Exception as e:
        result_label.config(text=f"Error: {str(e)}")
        messagebox.showerror("Input Error", "Enter a valid symptom from the list provided to you. Ensure correct formatting and spelling of symptoms!")

def clear_input():
    symptom_listbox.selection_clear(0, END)
    result_label.config(text="")

def save_input():
    user_input = [symptom_listbox.get(idx) for idx in symptom_listbox.curselection()]
    with open("user_input.txt", "w") as file:
        file.write(", ".join(user_input))
    messagebox.showinfo("Save Input", "Symptoms saved successfully!")

def load_input():
    try:
        with open("user_input.txt", "r") as file:
            user_input = file.read().split(", ")
            symptom_listbox.selection_clear(0, END)
            for symptom in user_input:
                idx = symptom_names.index(symptom.lower())
                symptom_listbox.selection_set(idx)
        messagebox.showinfo("Load Input", "Symptoms loaded successfully!")
    except Exception as e:
        messagebox.showerror("Load Error", f"Error loading symptoms: {str(e)}")

def search_symptom(var, idx, mode):
    search_term = search_var.get().lower()
    symptom_listbox.selection_clear(0, END)
    for idx, symptom in enumerate(symptom_names):
        if search_term in symptom:
            symptom_listbox.selection_set(idx)

def open_medical_resource():
    selected_model = model_var.get()
    rf_pred = rf_classifier.predict(sc.transform([lst])).astype(int)
    disease_name = le.inverse_transform([rf_pred[0]])
    webbrowser.open(f"https://www.webmd.com/search/search_results/default.aspx?query={disease_name[0]}")

def submit_feedback():
    feedback = feedback_text.get("1.0", END)
    with open("feedback.txt", "a") as file:
        file.write(feedback)
    messagebox.showinfo("Feedback", "Thank you for your feedback!")
    feedback_text.delete("1.0", END)

# Create main window
root = Tk()
root.title("Disease Prediction System")
root.geometry("800x600")

# Login frame
login_frame = Frame(root)
login_frame.pack()

username_label = Label(login_frame, text="Username:")
username_label.grid(row=0, column=0)
username_entry = Entry(login_frame)
username_entry.grid(row=0, column=1)

password_label = Label(login_frame, text="Password:")
password_label.grid(row=1, column=0)
password_entry = Entry(login_frame, show="*")
password_entry.grid(row=1, column=1)

login_button = Button(login_frame, text="Login", command=login)
login_button.grid(row=2, column=0, columnspan=2)

# Main frame with tabs
main_frame = Frame(root)
notebook = ttk.Notebook(main_frame)
notebook.pack(fill='both', expand=True)

# Prediction Tab
prediction_frame = Frame(notebook)
notebook.add(prediction_frame, text='Prediction')

# Model selection
model_var = StringVar(value='Random Forest')
model_label = Label(prediction_frame, text="Select Model:")
model_label.pack()
model_dropdown = ttk.Combobox(prediction_frame, textvariable=model_var, values=['Random Forest', 'Decision Tree', 'Naive Bayes'])
model_dropdown.pack()

# Searchable symptom list
search_var = StringVar()
search_var.trace("w", search_symptom)
search_entry = Entry(prediction_frame, textvariable=search_var)
search_entry.pack()
symptom_names = dataset.columns.tolist()[:-1]
symptom_listbox = Listbox(prediction_frame, selectmode=MULTIPLE, width=50, height=20)
for symptom in symptom_names:
    symptom_listbox.insert(END, symptom)
symptom_listbox.pack()

button_frame = Frame(prediction_frame)
button_frame.pack()

predict_button = Button(button_frame, text="Predict Disease", command=predict_disease)
predict_button.grid(row=0, column=0)

clear_button = Button(button_frame, text="Clear Input", command=clear_input)
clear_button.grid(row=0, column=1)

save_button = Button(button_frame, text="Save Input", command=save_input)
save_button.grid(row=0, column=2)

load_button = Button(button_frame, text="Load Input", command=load_input)
load_button.grid(row=0, column=3)

resource_button = Button(button_frame, text="Open Medical Resource", command=open_medical_resource)
resource_button.grid(row=1, column=0, columnspan=4)

result_label = Label(prediction_frame, text="")
result_label.pack()

# Feedback Tab
feedback_frame = Frame(notebook)
notebook.add(feedback_frame, text='Feedback')

feedback_label = Label(feedback_frame, text="Submit Feedback:")
feedback_label.pack()
feedback_text = Text(feedback_frame, height=10, width=70)
feedback_text.pack()
submit_feedback_button = Button(feedback_frame, text="Submit Feedback", command=submit_feedback)
submit_feedback_button.pack()

# Help Tab
help_frame = Frame(notebook)
notebook.add(help_frame, text='Help')

help_text = """
1. Select symptoms from the list.
2. Choose a model for prediction.
3. Click 'Predict Disease' to get the prognosis.
4. Use 'Save Input' to save your symptoms and 'Load Input' to load them.
5. Click 'Open Medical Resource' to learn more about the predicted diseases.
6. Provide feedback in the section below.
"""
help_label = Label(help_frame, text=help_text, justify=LEFT)
help_label.pack()

# Run the GUI application
root.mainloop()
