import tkinter as tk
from tkinter import messagebox, simpledialog
import numpy as np
import joblib
from sklearn.preprocessing import StandardScaler
import threading
import time
from datetime import datetime, timedelta

# Load the pre-trained model and scaler
model = joblib.load('diabetes_model.pkl')
scaler = joblib.load('scaler.pkl')

# Global variable for reminder time
reminder_time = None

# Define acceptable ranges
RANGES = {
    'pregnancies': (0, 20),
    'glucose': (0, 200),
    'blood_pressure': (0, 122),
    'skin_thickness': (0, 99),
    'insulin': (0, 846),
    'bmi': (0, 67),
    'dpf': (0.0, 2.5),
    'age': (0, 120)
}

def validate_input(value, metric):
    """Validate the input against defined ranges."""
    min_val, max_val = RANGES[metric]
    if not (min_val <= value <= max_val):
        raise ValueError(f"{metric.capitalize()} must be between {min_val} and {max_val}.")

def predict_diabetes():
    global reminder_time
    try:
        # Gather data from input fields
        pregnancies = float(entry_pregnancies.get())
        glucose = float(entry_glucose.get())
        blood_pressure = float(entry_bp.get())
        skin_thickness = float(entry_skin.get())
        insulin = float(entry_insulin.get())
        bmi = float(entry_bmi.get())
        dpf = float(entry_dpf.get())
        age = float(entry_age.get())

        # Validate each input
        validate_input(pregnancies, 'pregnancies')
        validate_input(glucose, 'glucose')
        validate_input(blood_pressure, 'blood_pressure')
        validate_input(skin_thickness, 'skin_thickness')
        validate_input(insulin, 'insulin')
        validate_input(bmi, 'bmi')
        validate_input(dpf, 'dpf')
        validate_input(age, 'age')

        # Arrange the data for prediction
        input_data = np.array([[pregnancies, glucose, blood_pressure, skin_thickness, insulin, bmi, dpf, age]])
        scaled_input = scaler.transform(input_data)
        result = model.predict(scaled_input)

        # Display result and recommendations
        if result[0] == 1:
            prediction_message = "The patient is likely to have diabetes, 🙁🙁🙏 be safe and be healthy."
            recommendations = (
                "Recommended actions:\n"
                "- Medications: Metformin, Sulfonylureas\n"
                "- Food Habits: Eat a balanced diet, limit sugars, and stay hydrated.\n"
                "- Set a reminder for taking medications."
            )
            set_reminder()
        else:
            prediction_message = "The patient is unlikely to have diabetes, 😊😎🙂👍 keep smiling and enjoy."
            recommendations = (
                "Recommended actions:\n"
                "- Maintain a healthy lifestyle.\n"
                "- Regular check-ups are advised."
            )
        
        # Show prediction message and recommendations
        messagebox.showinfo("Prediction Result", f"{prediction_message}\n{recommendations}")

    except Exception as e:
        messagebox.showerror("Error", f"Invalid input: {e}")

def set_reminder():
    global reminder_time
    time_input = simpledialog.askstring("Input", "Enter time to take tablets (e.g., 08:00 AM):")
    
    if time_input:
        try:
            # Validate the time format
            reminder_time_obj = datetime.strptime(time_input, "%I:%M %p")
            reminder_time = reminder_time_obj.strftime("%I:%M %p")
            messagebox.showinfo("Reminder Set", f"Reminder set for taking tablets at {reminder_time}.")

            # Start the reminder thread if it's not already running
            if not any(t.name == "ReminderThread" for t in threading.enumerate()):
                threading.Thread(target=reminder_thread, name="ReminderThread", daemon=True).start()

            # Provide immediate feedback on time left until reminder
            now = datetime.now()
            reminder_datetime = datetime.combine(now.date(), reminder_time_obj.time())
            if reminder_datetime < now:  # If the time has already passed today
                reminder_datetime += timedelta(days=1)
            time_left = reminder_datetime - now
            messagebox.showinfo("Time Until Reminder", f"Reminder will trigger in {time_left.seconds // 60} minutes.")
        
        except ValueError:
            messagebox.showerror("Error", "Invalid time format. Please use HH:MM AM/PM.")
    else:
        messagebox.showwarning("Warning", "No time entered for the reminder.")

def reminder_thread():
    while True:
        if reminder_time:
            now = datetime.now().strftime("%I:%M %p")
            if now == reminder_time:
                messagebox.showinfo("Medication Reminder", "It's time to take your medication!\nDon't forget to eat healthily!")
                time.sleep(60)  # Wait a minute to avoid repeated notifications
            time.sleep(30)  # Check every 30 seconds
        else:
            time.sleep(30)  # Check every 30 seconds if reminder time is not set

# Create the GUI window
root = tk.Tk()
root.title("Diabetes Prediction using Machine Learning")
root.configure(bg='pink')

# Center the window
window_width = 500
window_height = 400
screen_width = root.winfo_screenwidth()
screen_height = root.winfo_screenheight()
x = (screen_width // 2) - (window_width // 2)
y = (screen_height // 2) - (window_height // 2)
root.geometry(f'{window_width}x{window_height}+{x}+{y}')

# Create input labels and entry fields
tk.Label(root, text="PREGNANCIES", bg='pink').grid(row=0, padx=5, pady=5)
tk.Label(root, text="GLUCOSE", bg='pink').grid(row=1, padx=5, pady=5)
tk.Label(root, text="BLOOD PRESSURE", bg='pink').grid(row=2, padx=5, pady=5)
tk.Label(root, text="SKIN THICKNESS", bg='pink').grid(row=3, padx=5, pady=5)
tk.Label(root, text="INSULIN", bg='pink').grid(row=4, padx=5, pady=5)
tk.Label(root, text="BMI", bg='pink').grid(row=5, padx=5, pady=5)
tk.Label(root, text="DIABETES PEDIGREE FUNCTION", bg='pink').grid(row=6, padx=5, pady=5)
tk.Label(root, text="AGE", bg='pink').grid(row=7, padx=5, pady=5)

entry_pregnancies = tk.Entry(root)
entry_glucose = tk.Entry(root)
entry_bp = tk.Entry(root)
entry_skin = tk.Entry(root)
entry_insulin = tk.Entry(root)
entry_bmi = tk.Entry(root)
entry_dpf = tk.Entry(root)
entry_age = tk.Entry(root)

entry_pregnancies.grid(row=0, column=1, padx=5, pady=5)
entry_glucose.grid(row=1, column=1, padx=5, pady=5)
entry_bp.grid(row=2, column=1, padx=5, pady=5)
entry_skin.grid(row=3, column=1, padx=5, pady=5)
entry_insulin.grid(row=4, column=1, padx=5, pady=5)
entry_bmi.grid(row=5, column=1, padx=5, pady=5)
entry_dpf.grid(row=6, column=1, padx=5, pady=5)
entry_age.grid(row=7, column=1, padx=5, pady=5)

# Create buttons
predict_button = tk.Button(root, text="PREDICT", command=predict_diabetes)
predict_button.grid(row=8, column=1, pady=10)

# Run the application
root.mainloop()
