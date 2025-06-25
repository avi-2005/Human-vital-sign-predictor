import tkinter as tk
from tkinter import ttk, messagebox
import requests
from PIL import Image, ImageTk, ImageSequence

class VitalSignsApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Vital Signs Analyzer")
        self.root.geometry("700x750")  # Increased width to fit the GIF
        self.root.configure(bg="#E6F0FA")  # Light blue background

        # Define styles
        style = ttk.Style()
        style.configure("TFrame", background="#E6F0FA")
        style.configure("TLabel", background="#E6F0FA", font=("Helvetica", 12))
        style.configure("TEntry", font=("Helvetica", 12))
        style.configure("Analyze.TButton", background="#4A90E2", foreground="black", font=("Helvetica", 12, "bold"))
        style.map("Analyze.TButton",
                  background=[("active", "#357ABD")],
                  foreground=[("active", "black")])

        # Create input fields and place beside the GIF
        main_frame = ttk.Frame(root)
        main_frame.pack(pady=10)

        frame = ttk.LabelFrame(main_frame, text="Enter Vital Signs", padding=10, style="TFrame")
        frame.grid(row=0, column=0, padx=10, pady=10, sticky="nw")

        self.entries = {}
        self.features = [
            "Heart Rate", "Oxygen Saturation", "Systolic Blood Pressure", "Diastolic Blood Pressure",
            "Derived_BMI", "Respiratory Rate", "Body Temperature", "Age", "Derived_HRV",
            "Derived_Pulse_Pressure", "Derived_MAP"
        ]

        for i, feature in enumerate(self.features):
            label = ttk.Label(frame, text=f"{feature}:", style="TLabel")
            label.grid(row=i, column=0, padx=5, pady=5, sticky="e")
            entry = ttk.Entry(frame, width=15, style="TEntry")
            entry.grid(row=i, column=1, padx=5, pady=5, sticky="w")
            self.entries[feature] = entry

        # GIF display frame
        gif_frame = ttk.LabelFrame(main_frame, text="Health Animation", padding=10, style="TFrame")
        gif_frame.grid(row=0, column=1, padx=10, pady=10, sticky="ne")

        self.gif_label = tk.Label(gif_frame, bg="#E6F0FA")
        self.gif_label.pack()

        # Load GIF
        self.load_gif(r"C:\Users\Avi\Downloads\gif heart.gif")  # Change to your GIF file

        # Submit button
        submit_btn = ttk.Button(root, text="Analyze", command=self.analyze_vitals, style="Analyze.TButton")
        submit_btn.pack(pady=10)

        # Results display
        result_frame = ttk.LabelFrame(root, text="Analysis Results", padding=10, style="TFrame")
        result_frame.pack(padx=10, pady=10, fill="both", expand=True)

        self.result_text = tk.Text(result_frame, height=15, width=70, wrap=tk.WORD, bg="#FFFFFF", font=("Arial", 10))
        self.result_text.pack(padx=5, pady=5, fill="both", expand=True)

    def load_gif(self, gif_path):
        self.gif = Image.open(gif_path)
        self.frames = [ImageTk.PhotoImage(frame) for frame in ImageSequence.Iterator(self.gif)]
        self.frame_index = 0
        self.animate_gif()

    def animate_gif(self):
        if self.frames:
            self.gif_label.configure(image=self.frames[self.frame_index])
            self.frame_index = (self.frame_index + 1) % len(self.frames)
            self.root.after(100, self.animate_gif)  # Adjust speed

    def analyze_vitals(self):
        patient_data = {}
        for feature, entry in self.entries.items():
            value = entry.get().strip()
            if value:
                try:
                    patient_data[feature] = float(value)
                except ValueError:
                    messagebox.showerror("Error", f"Invalid input for {feature}: {value}")
                    return

        try:
            response = requests.post("http://localhost:5000/api/analyze", json=patient_data)
            response.raise_for_status()
            result = response.json()
            
            self.result_text.delete(1.0, tk.END)
            self.result_text.insert(tk.END, "\n=== Predictions ===\n", "header")
            self.display_list(result.get('predictions', []), "No predictions available.")
            
            self.result_text.insert(tk.END, "\n=== Health Recommendations ===\n", "header")
            self.display_list(result.get('recommendations', []), "No specific recommendations.")
            
            self.result_text.insert(tk.END, "\n=== Updated Vital Signs ===\n", "header")
            for key, value in result.get('updatedData', {}).items():
                self.result_text.insert(tk.END, f"{key}: {value}\n")
            
        except requests.exceptions.RequestException as e:
            messagebox.showerror("Error", f"Failed to connect to API: {str(e)}")

    def display_list(self, items, default_message):
        if items:
            for item in items:
                self.result_text.insert(tk.END, f"- {item}\n")
        else:
            self.result_text.insert(tk.END, f"{default_message}\n")

if __name__ == "__main__":
    root = tk.Tk()
    app = VitalSignsApp(root)
    root.mainloop()
