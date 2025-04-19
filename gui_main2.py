import customtkinter as ctk
from tkinter import filedialog, messagebox
from PIL import Image
from ai_model.training.train import train_model
from ai_model.training.test import test_model
from ai_model.prediction.prediction import make_prediction
import os
import csv

ctk.set_appearance_mode("System")  # "Light", "Dark", or "System"
ctk.set_default_color_theme("blue")  # You can change this to "green", "dark-blue", etc.

class EmotionGUI(ctk.CTk):
    def __init__(self):
        super().__init__()

        self.title("Emotion Detection GUI")
        self.geometry("800x600")

        self.tabview = ctk.CTkTabview(self, width=760, height=500)
        self.tabview.pack(padx=20, pady=20)

        self.train_tab = self.tabview.add("Train")
        self.test_tab = self.tabview.add("Test")
        self.predict_tab = self.tabview.add("Predict")

        self.create_train_tab()
        self.create_test_tab()
        self.create_predict_tab()

    # ========== Train ==========
    def create_train_tab(self):
        self.train_dir = ctk.CTkEntry(self.train_tab, placeholder_text="Training Directory Path", width=500)
        self.train_dir.pack(pady=10)
        ctk.CTkButton(self.train_tab, text="Browse", command=self.browse_train_dir).pack()

        self.val_dir = ctk.CTkEntry(self.train_tab, placeholder_text="Validation Directory Path", width=500)
        self.val_dir.pack(pady=10)
        ctk.CTkButton(self.train_tab, text="Browse", command=self.browse_val_dir).pack()

        ctk.CTkButton(self.train_tab, text="Start Training", command=self.train).pack(pady=20)

    def browse_train_dir(self):
        path = filedialog.askdirectory()
        self.train_dir.delete(0, ctk.END)
        self.train_dir.insert(0, path)

    def browse_val_dir(self):
        path = filedialog.askdirectory()
        self.val_dir.delete(0, ctk.END)
        self.val_dir.insert(0, path)

    def train(self):
        try:
            train_model(self.train_dir.get(), self.val_dir.get())
            messagebox.showinfo("Success", "Training Complete!")
        except Exception as e:
            messagebox.showerror("Training Error", str(e))

    # ========== Test ==========
    def create_test_tab(self):
        self.test_path = ctk.CTkEntry(self.test_tab, placeholder_text="Image or Directory Path", width=500)
        self.test_path.pack(pady=10)
        ctk.CTkButton(self.test_tab, text="Select Image or Directory", command=self.select_test_path).pack()

        self.test_model = ctk.CTkEntry(self.test_tab, placeholder_text="Model Path", width=500)
        self.test_model.pack(pady=10)
        ctk.CTkButton(self.test_tab, text="Select Model", command=self.select_test_model).pack()

        ctk.CTkButton(self.test_tab, text="Run Test", command=self.test).pack(pady=20)

    def select_test_path(self):
        path = filedialog.askopenfilename() or filedialog.askdirectory()
        self.test_path.delete(0, ctk.END)
        self.test_path.insert(0, path)

    def select_test_model(self):
        path = filedialog.askopenfilename(filetypes=[("Kereas Models", "*.keras"),("H5 Model", "*.h5")])
        self.test_model.delete(0, ctk.END)
        self.test_model.insert(0, path)

    def test(self):
        try:
            result = test_model(
                image_path=self.test_path.get() if os.path.isfile(self.test_path.get()) else None,
                directory_path=self.test_path.get() if os.path.isdir(self.test_path.get()) else None,
                model_path=self.test_model.get()
            )
            messagebox.showinfo("Test Result", str(result))
        except Exception as e:
            messagebox.showerror("Testing Error", str(e))

    # Predict Tab
    def create_predict_tab(self):
        self.predict_mode = ctk.StringVar(value="single")

        mode_frame = ctk.CTkFrame(self.predict_tab)
        mode_frame.pack(pady=10)

        ctk.CTkLabel(mode_frame, text="Prediction Mode:").pack(side="left", padx=10)
        ctk.CTkRadioButton(mode_frame, text="Single", variable=self.predict_mode, value="single", command=self.toggle_predict_mode).pack(side="left")
        ctk.CTkRadioButton(mode_frame, text="Batch", variable=self.predict_mode, value="batch", command=self.toggle_predict_mode).pack(side="left")

        # Single Image Entry
        self.predict_image = ctk.CTkEntry(self.predict_tab, placeholder_text="Select Image", width=500)
        self.predict_image.pack(pady=10)
        self.image_browse_btn = ctk.CTkButton(self.predict_tab, text="Browse Image", command=self.browse_predict_image)
        self.image_browse_btn.pack()

        # Batch Directory Entry (hidden initially)
        self.batch_dir = ctk.CTkEntry(self.predict_tab, placeholder_text="Select Folder for Batch Prediction", width=500)
        self.batch_dir.pack(pady=10)
        self.batch_browse_btn = ctk.CTkButton(self.predict_tab, text="Browse Folder", command=self.browse_batch_dir)
        self.batch_browse_btn.pack()
        self.batch_dir.pack_forget()
        self.batch_browse_btn.pack_forget()

        self.predict_model = ctk.CTkEntry(self.predict_tab, placeholder_text="Select Model", width=500)
        self.predict_model.pack(pady=10)
        ctk.CTkButton(self.predict_tab, text="Browse Model", command=self.browse_predict_model).pack()


        self.prediction_result = ctk.CTkLabel(self.predict_tab, text="", font=ctk.CTkFont(size=20))
        self.prediction_result.pack(pady=10)
        ctk.CTkButton(self.predict_tab, text="Predict", command=self.predict).pack(pady=20)

        self.emoji_label = ctk.CTkLabel(self.predict_tab, text="", font=ctk.CTkFont(size=50))
        self.emoji_label.pack()

    def toggle_predict_mode(self):
        if self.predict_mode.get() == "single":
            self.predict_image.pack(pady=10)
            self.image_browse_btn.pack()
            self.batch_dir.pack_forget()
            self.batch_browse_btn.pack_forget()
        else:
            self.predict_image.pack_forget()
            self.image_browse_btn.pack_forget()
            self.batch_dir.pack(pady=10)
            self.batch_browse_btn.pack()

    def browse_predict_image(self):
        path = filedialog.askopenfilename(filetypes=[("Images", "*.jpg *.png *.jpeg")])
        self.predict_image.delete(0, ctk.END)
        self.predict_image.insert(0, path)

    def browse_batch_dir(self):
        path = filedialog.askdirectory()
        self.batch_dir.delete(0, ctk.END)
        self.batch_dir.insert(0, path)

    def browse_predict_model(self):
        path = filedialog.askopenfilename(filetypes=[("H5 Model", "*.h5")])
        self.predict_model.delete(0, ctk.END)
        self.predict_model.insert(0, path)

    def predict(self):
        try:
            if self.predict_mode.get() == "single":
                image_path = self.predict_image.get()
                if not os.path.exists(image_path):
                    raise Exception("Image path does not exist.")
                prediction = make_prediction(image_path=image_path, directory_path=None, model_path=self.predict_model.get())
                emotion = prediction[0].lower()

                emoji_map = {"happy": "😊", "sad": "😢", "angry": "😠"}
                emoji = emoji_map.get(emotion, "❓")

                self.prediction_result.configure(text=f"Detected Emotion: {emotion.capitalize()}")
                self.emoji_label.configure(text=emoji)

            else:
                folder_path = self.batch_dir.get()
                if not os.path.isdir(folder_path):
                    raise Exception("Batch directory does not exist.")

                predictions = make_prediction(image_path=None, directory_path=folder_path, model_path=self.predict_model.get())

                csv_path = os.path.join(folder_path, "batch_predictions.csv")
                with open(csv_path, "w", newline="") as f:
                    writer = csv.writer(f)
                    writer.writerow(["image_path", "prediction"])
                    for image_path, prediction in predictions.items():
                        writer.writerow([image_path, prediction])

                self.prediction_result.configure(text=f"Batch prediction saved to:\n{csv_path}")
                self.emoji_label.configure(text="📄")

        except Exception as e:
            messagebox.showerror("Prediction Error", str(e))


if __name__ == "__main__":
    app = EmotionGUI()
    app.mainloop()
