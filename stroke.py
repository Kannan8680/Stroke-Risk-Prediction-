import pandas as pd
import joblib
import serial
import time
import tkinter as tk
from tkinter import Label, Button, Frame
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import numpy as np
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg

# Load the trained model and scaler
model = joblib.load(r"C:\Users\kannan\Downloads\stroke_model.pkl")
scaler = joblib.load(r"C:\Users\kannan\Downloads\scaler.pkl")

# Setup serial communication
ser = serial.Serial('COM4', 9600, timeout=1)
time.sleep(2)

# CSV setup
csv_file = "data.csv"
columns = ['Alpha', 'Beta', 'Theta', 'Delta', 'Stroke_Risk']
with open(csv_file, 'w') as f:
    f.write(','.join(columns) + '\n')

# Data Buffers
prediction_buffer = []
eeg_data = {'Alpha': [], 'Beta': [], 'Theta': [], 'Delta': []}

# GUI Setup
root = tk.Tk()
root.title("Stroke Prediction System")
root.geometry("1000x750")
root.configure(bg="#e0e0e0")

# Header Label
header = Label(root, text="Stroke Prediction System", font=("Arial", 22, "bold"), fg="#333", bg="#e0e0e0", pady=10)
header.pack()

frame = Frame(root, bg="#e0e0e0")
frame.pack(pady=10)

label = Label(frame, text="Press 'Start Test' to Begin", font=("Arial", 18, "bold"), fg="black", bg="#e0e0e0", padx=10, pady=10)
label.pack()

# Status Label (Dynamic)
status_label = Label(root, text="", font=("Arial", 18, "bold"), fg="black", bg="#e0e0e0", pady=10)
status_label.pack()

# Button Frame
button_frame = Frame(root, bg="#e0e0e0")
button_frame.pack(pady=5)

start_button = Button(button_frame, text="▶ Start Test", font=("Arial", 16, "bold"), bg="#4CAF50", fg="white", width=12, height=2, command=lambda: start_test())
start_button.grid(row=0, column=0, padx=20)

stop_button = Button(button_frame, text="⏹ Stop Test", font=("Arial", 16, "bold"), bg="#D32F2F", fg="white", width=12, height=2, command=lambda: stop_test())
stop_button.grid(row=0, column=1, padx=20)

# Matplotlib Setup
fig, axes = plt.subplots(4, 1, figsize=(8, 6), sharex=True)
fig.suptitle("EEG Waves (Real-time Visualization)", fontsize=16, fontweight="bold", color="#333")

x_data = np.linspace(0, 2 * np.pi, 100)
y_data = {wave: np.zeros(100) for wave in ['Alpha', 'Beta', 'Theta', 'Delta']}
colors = {'Alpha': '#1f77b4', 'Beta': '#ff7f0e', 'Theta': '#2ca02c', 'Delta': '#d62728'}

lines = {}
for i, wave in enumerate(['Alpha', 'Beta', 'Theta', 'Delta']):
    axes[i].set_ylim(-2, 2)
    axes[i].set_xlim(0, 2 * np.pi)
    axes[i].set_ylabel(wave, fontweight="bold")
    axes[i].grid(True, linestyle="--", alpha=0.7)
    lines[wave], = axes[i].plot(x_data, y_data[wave], color=colors[wave], label=wave, linewidth=2)
    axes[i].legend()

axes[-1].set_xlabel("Time", fontweight="bold")
is_running = False

def update_gui(message, color):
    label.config(text=message, fg=color)
    status_label.config(text=message, fg=color)
    root.update()

def update_plot(frame):
    for wave in ['Alpha', 'Beta', 'Theta', 'Delta']:
        if len(eeg_data[wave]) > 0:
            amplitude = eeg_data[wave][-1] / 10
            frequency = abs(eeg_data[wave][-1]) + 1
            y_data[wave] = amplitude * np.sin(frequency * x_data)
            lines[wave].set_ydata(y_data[wave])
    return lines.values()

def start_test():
    global is_running
    if not is_running:
        is_running = True
        update_gui("Listening for EEG Data...", "#1976D2")
        read_serial_data()

def stop_test():
    global is_running
    is_running = False
    update_gui("Test Stopped", "#333")

def read_serial_data():
    global is_running
    if not is_running:
        return

    try:
        if ser.in_waiting > 0:
            line = ser.readline().decode('utf-8').strip()
            if line:
                try:
                    values = list(map(float, line.split(',')))
                    if len(values) != 4:
                        print("Invalid data format. Skipping.")
                        return
                    
                    df_new = pd.DataFrame([values], columns=['Alpha', 'Beta', 'Theta', 'Delta'])
                    X_new_scaled = scaler.transform(df_new)
                    prediction = model.predict_proba(X_new_scaled)[0][1] * 100  # Get probability percentage
                    prediction_buffer.append(prediction)
                    
                    for i, wave in enumerate(['Alpha', 'Beta', 'Theta', 'Delta']):
                        eeg_data[wave].append(values[i])

                    print(f"Received Data: Alpha={values[0]}, Beta={values[1]}, Theta={values[2]}, Delta={values[3]}, Stroke Risk={prediction:.2f}%")

                    with open(csv_file, 'a') as f:
                        f.write(','.join(map(str, values + [prediction])) + '\n')

                    if len(prediction_buffer) == 10:
                        avg_prediction = sum(prediction_buffer) / 10
                        final_prediction = 1 if avg_prediction >= 70 else 0  # Threshold 70%

                        print(f"Final Stroke Prediction: {final_prediction}")  # Print 0 or 1

                        if final_prediction == 1:
                            update_gui("⚠️ Stroke Detected!", "#D32F2F")
                        else:
                            update_gui("✅ Normal", "#4CAF50")

                        prediction_buffer.clear()

                except ValueError as e:
                    print(f"Error processing data: {e}")

        root.after(500, read_serial_data)

    except KeyboardInterrupt:
        print("\nStopped by user.")
        ser.close()
        root.destroy()

ani = animation.FuncAnimation(fig, update_plot, interval=500, cache_frame_data=False)
canvas = FigureCanvasTkAgg(fig, master=root)
canvas_widget = canvas.get_tk_widget()
canvas_widget.pack(pady=20)

root.mainloop()

