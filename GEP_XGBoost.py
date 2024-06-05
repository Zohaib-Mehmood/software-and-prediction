import tkinter as tk
from tkinter import ttk
from math import pow, sqrt
from PIL import Image, ImageTk
import xgboost as xgb
import numpy as np
from sklearn.model_selection import train_test_split
import pandas as pd
from sklearn.multioutput import MultiOutputRegressor

class RangeInputGUI:
    def __init__(self, master):
        self.master = master
        master.title(
            "Graphical User Interface (GUI) for predicting the peak share strength of H-shaped RC squat walls")

        # Set a nice background color
        master.configure(background="#E8E8E8")

        # Add the main heading
        main_heading = tk.Label(master, text="Graphical User Interface (GUI) for predicting the peak share strength of H-shaped RC squat walls",
                                bg="#FFA500", fg="#0000FF", font=("Helvetica", 16, "bold"), pady=10)  # Orange background, Blue text
        main_heading.pack(side=tk.TOP, fill=tk.X)

        # Create a frame for the content with scrollbar
        self.content_frame = tk.Frame(master, bg="#E8E8E8")
        self.content_frame.pack(side=tk.TOP, fill=tk.BOTH,
                                expand=True, padx=20, pady=20)

        self.canvas = tk.Canvas(self.content_frame, bg="#E8E8E8")
        self.scrollbar = ttk.Scrollbar(
            self.content_frame, orient="vertical", command=self.canvas.yview)
        self.scrollable_frame = tk.Frame(self.canvas, bg="#E8E8E8")

        self.scrollable_frame.bind(
            "<Configure>",
            lambda e: self.canvas.configure(
                scrollregion=self.canvas.bbox("all")
            )
        )

        self.canvas.create_window(
            (0, 0), window=self.scrollable_frame, anchor="nw")
        self.canvas.configure(yscrollcommand=self.scrollbar.set)

        self.canvas.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        self.scrollbar.pack(side=tk.RIGHT, fill=tk.Y)

        # Input parameters frame
        self.input_frame = tk.Frame(
            self.scrollable_frame, bg="#FFFFFF", bd=2, relief=tk.RIDGE)
        self.input_frame.pack(side=tk.LEFT, fill="both",
                              padx=20, pady=20, expand=True)

        # Add a heading
        heading = tk.Label(self.input_frame, text="Input Parameters", bg="#4CAF50",
                           fg="white", font=("Helvetica", 16, "bold"), pady=10)  # Green heading
        heading.grid(row=0, column=0, columnspan=3, pady=10)

        # Constant variables
        self.G1C4 = -24.9941342003774
        self.G1C3 = 6.47694323705234
        self.G2C0 = 127.548920335475
        self.G2C2 = 8.60619685094863
        self.G4C7 = 7.14738074447545
        self.G4C4 = 577.183457643781

        # Create labels and sliders for d0 to d12
        self.create_slider("Loading Type (Categorical):", 0, 2.0, 0, 1)
        self.create_slider("Shear span ratio:", 0.25, 2.0, 0.5, 3)
        self.create_slider(
            "Ratio of Flange Thickness to web thickness:", 0.80, 1.874, 1, 5)
        self.create_slider("Flange length (mm):", 145, 3045, 609.6, 7)
        self.create_slider(
            "Concrete compressive strength (MPa):", 13.8, 110.70, 29, 9)
        self.create_slider(
            "Web longitudinal bar yield strength (MPa):", 0, 638, 543.3, 11)
        self.create_slider(
            "Horizontal steel bar yield (MPa):", 0, 610, 495.7, 13)
        self.create_slider(
            "Flange longitudinal bar yield strength (MPa):", 235, 638, 525.4, 15)
        self.create_slider(
            "Reinforcement ratio of the web longitudinal bars (%):", 0, 2.54, 0.5, 17)
        self.create_slider(
            "Reinforcement ratio of horizontal bars (%):", 0, 1.69, 0.5, 19)
        self.create_slider(
            "Reinforcement ratio of flange longitudinal bars (kN):", 0.35, 6.4, 1.8, 21)
        self.create_slider("Axial compressive force (kN):", 0, 2364, 0, 23)

        # Output frames for GEP and XGBoost
        self.output_frame = tk.Frame(
            self.scrollable_frame, bg="#FFFFFF", bd=2, relief=tk.RIDGE)
        self.output_frame.pack(side=tk.TOP, fill="both",
                               padx=20, pady=20)

        heading = tk.Label(self.output_frame, text="Output", bg="#4CAF50",
                           fg="white", font=("Helvetica", 16, "bold"), pady=10)  # Green heading
        heading.grid(row=0, column=0, columnspan=3, pady=10)

        # Output box for GEP
        self.calculate_button = tk.Button(self.output_frame, text="GEP", command=self.calculate_y,
                                          bg="blue", fg="white", font=("Helvetica", 12, "bold"), relief=tk.RAISED)
        self.calculate_button.grid(row=1, column=0, pady=10, padx=10)

        self.gep_output_text = tk.Text(self.output_frame, height=2, width=72)
        self.gep_output_text.grid(row=1, column=1, padx=10, pady=10)

        # Output box for XGBoost
        self.xgboost_button = tk.Button(self.output_frame, text="XGBoost", command=self.calculate_xgboost,
                                        bg="blue", fg="white", font=("Helvetica", 12, "bold"), relief=tk.RAISED)
        self.xgboost_button.grid(row=2, column=0, pady=10, padx=10)

        self.xgboost_output_text = tk.Text(
            self.output_frame, height=2, width=72)
        self.xgboost_output_text.grid(row=2, column=1, padx=10, pady=10)

        # Load and display the image on the right side below the output box
        image_frame = tk.Frame(self.scrollable_frame, bg="#E8E8E8")
        image_frame.pack(side=tk.TOP, padx=0, pady=0,
                         fill=tk.BOTH)
        
        # Add developer information below the image
        developer_info = tk.Label(image_frame, text="This GUI is developed by Zohaib Mehmood (zoohaibmehmood@gmail.com), COMSATS University Islamabad.",
                                  bg="light blue", fg="red", font=("Helvetica", 11, "bold"), pady=10)
        developer_info.pack()

        image = Image.open("final.jpg")
        self.photo = ImageTk.PhotoImage(
            image.resize((550, 450)))  # Adjust size as needed

        self.image_label = tk.Label(
            image_frame, image=self.photo, bg="#E8E8E8")
        self.image_label.pack(padx=20, pady=20)

        

    def create_slider(self, text, from_, to, initial, row):
        label = tk.Label(self.input_frame, text=text, font=(
        "Helvetica", 12, "bold"), fg="#00698f", bg="#FFFFFF")
        label.grid(row=row, column=0, padx=10, pady=10, sticky="w")

        # Create a custom style for the slider
        style = ttk.Style()
        style.configure("Custom.Horizontal.TScale", sliderlength=30)

        slider = ttk.Scale(self.input_frame, from_=from_,
                           to=to, orient="horizontal", length=200, style="Custom.Horizontal.TScale")
        slider.set(initial)
        slider.grid(row=row, column=1, columnspan=2, padx=10, pady=10)

        min_label = tk.Label(self.input_frame, text=f"Min: {from_}", font=(
        "Helvetica", 10), bg="#FFFFFF")
        min_label.grid(row=row+1, column=0, padx=10, pady=5, sticky="w")

        current_label = tk.Label(self.input_frame, text=f"Current: {initial}", font=(
        "Helvetica", 10), bg="#FFFFFF")
        current_label.grid(row=row+1, column=1, padx=10, pady=5, sticky="w")

        max_label = tk.Label(self.input_frame, text=f"Max: {to}", font=(
        "Helvetica", 10), bg="#FFFFFF")
        max_label.grid(row=row+1, column=2, padx=10, pady=5, sticky="w")

        slider.bind("<Motion>", lambda event, cl=current_label,
                s=slider: cl.config(text=f"Current: {s.get():.2f}"))

        setattr(self, f"d{row//2}", slider)

    def calculate_y(self):
        try:
            d = [getattr(self, f"d{i}").get() for i in range(12)]

            # Safeguard calculations to avoid division by zero
            if d[1] == 0 or d[3] == 0 or d[7] == 0 or d[8] == 0:
                raise ValueError("One of the inputs caused a division by zero")

            y = ((d[10] + (self.G1C3 / (d[8] + pow(((d[9] - d[0]) -
                 (d[11] / d[7])), 2.0)))) * (self.G1C4 / d[1]))
            y += ((d[6] + self.G2C0) - pow((sqrt((((d[3] * self.G2C2)
                  * (d[5] * d[0])) + ((d[11] * d[1]) * d[5])) / d[3])), 2.0))
            y += (((sqrt(sqrt((d[10] + pow((((d[9] * d[10]) * d[1])
                  * sqrt(d[8])), 2.0)))) / d[1]) - d[2]) * d[3])
            y += (d[0] * ((d[4] / (d[0] - (pow(pow(((self.G4C4 * d[6]) /
                  (d[7] * d[3])), 2.0), 2.0) * self.G4C7))) + d[6]))

            self.gep_output_text.delete(1.0, tk.END)
            self.gep_output_text.insert(
                tk.END, f"Predicted Peak Shear Strength using Gene Expression Programming: {y:.2f}")

        except Exception as e:
            self.gep_output_text.delete(1.0, tk.END)
            self.gep_output_text.insert(tk.END, f"Error: {str(e)}")

    def calculate_xgboost(self):
        try:
            d = [getattr(self, f"d{i}").get() for i in range(12)]
            # Read the Excel file using pandas
            base_dir = r"C:\Users\Zohaib Mehmood\Desktop\Saud Project"
            filename = r"data.xlsx"
            df = pd.read_excel(f"{base_dir}/{filename}")

            # Eliminate missing data from the dataframe
            df = df[df['V'] != 0]

            # Drop rows with missing values
            df.dropna(inplace=True)

            # Split data into input (x) and output (y)
            x = df.iloc[:, :-1]
            y = df.iloc[:, -1:]

            # Train-test split
            x_train, x_test, y_train, y_test = train_test_split(
                x, y, test_size=0.3, random_state=500)

            # Train XGBoost model
            regressor = MultiOutputRegressor(xgb.XGBRegressor(
                n_estimators=100,
                reg_lambda=0.01,
                gamma=1,
                max_depth=8
            ))
            model = regressor.fit(x_train, y_train)

            new_input = np.array(
                [[d[0], d[1], d[2], d[3], d[4], d[5], d[6], d[7], d[8], d[9], d[10], d[11]]])
            new_preds = model.predict(new_input)

            # Update output text with evaluation metrics
            self.xgboost_output_text.delete(1.0, tk.END)
            self.xgboost_output_text.insert(
                tk.END, f"Predicted Peak Shear Strength using XGBoost: {new_preds[0][0]:.2f}")

        except Exception as e:
            # Error handling
            self.xgboost_output_text.delete(1.0, tk.END)
            self.xgboost_output_text.insert(tk.END, "Error: XGBoost prediction failed")
            print("Error:", e)


if __name__ == "__main__":
    root = tk.Tk()
    app = RangeInputGUI(root)
    root.mainloop()
