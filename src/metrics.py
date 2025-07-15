import tkinter as tk
from tkinter import filedialog, ttk, messagebox
from PIL import Image, ImageTk
import os
import numpy as np
from skimage.metrics import structural_similarity as ssim
from skimage.metrics import mean_squared_error, peak_signal_noise_ratio
import csv

class EvaluadorDDPM3ColsSingleCombo:
    def __init__(self, root):
        self.root = root
        self.root.title("Visualizer and Metrics")
        self.root.geometry("1550x520")

        # Frame superior para selección única de raíz y clase
        top_frame = tk.Frame(root)
        top_frame.pack(fill=tk.X, padx=10, pady=10)

        btn_select_root = tk.Button(top_frame, text="Seleccionar carpeta raíz", command=self.select_root)
        btn_select_root.pack(side=tk.LEFT)

        self.class_combobox = ttk.Combobox(top_frame, state="readonly")
        self.class_combobox.pack(side=tk.LEFT, padx=10, fill=tk.X, expand=True)
        self.class_combobox.bind("<<ComboboxSelected>>", self.on_class_change)

        self.image_count_label = tk.Label(top_frame, text="Condition: 0 | Ground Truth: 0 | Generated: 0", font=("Arial", 10))
        self.image_count_label.pack(side=tk.LEFT, padx=10)

        # Frames para las tres columnas
        self.condition_frame = tk.Frame(root)
        self.condition_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=5, pady=10)

        self.gt_frame = tk.Frame(root)
        self.gt_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=5, pady=10)

        self.generated_frame = tk.Frame(root)
        self.generated_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=5, pady=10)

        # Panel de métricas
        self.metrics_frame = tk.Frame(root)
        self.metrics_frame.pack(side=tk.LEFT, fill=tk.Y, padx=10, pady=10)

        self.metrics_label = tk.Label(self.metrics_frame, text="Métricas", font=("Arial", 14, "bold"))
        self.metrics_label.pack(pady=5)

        self.metrics_text = tk.Text(self.metrics_frame, height=12, width=35, state="disabled")
        self.metrics_text.pack()

        self.metrics_button = tk.Button(self.metrics_frame, text="Calcular métricas", command=self.calculate_metrics)
        self.metrics_button.pack(pady=10)

        # Botón extra para exportar CSV
        self.export_button = tk.Button(self.metrics_frame, text="Exportar métricas por clase (CSV)", command=self.exportar_metricas_todas_las_clases)
        self.export_button.pack(pady=5)

        # Estado raíz y clases
        self.root_path = None
        self.classes = []

        self.condition_images = []
        self.gt_images = []
        self.generated_images = []

        self.current_index = 0

        # Setup columnas
        self.setup_column(self.condition_frame, "Condition", is_condition=True)
        self.setup_column(self.gt_frame, "Ground Truth", is_gt=True)
        self.setup_column(self.generated_frame, "Generated Images", is_generated=True)

        # Navegación
        self.nav_frame = tk.Frame(self.condition_frame)
        self.nav_frame.pack(pady=5)
        self.btn_prev = tk.Button(self.nav_frame, text="Anterior", command=self.prev_image)
        self.btn_prev.pack(side=tk.LEFT, padx=5)
        self.btn_next = tk.Button(self.nav_frame, text="Siguiente", command=self.next_image)
        self.btn_next.pack(side=tk.LEFT, padx=5)

    def setup_column(self, frame, label_text, is_condition=False, is_gt=False, is_generated=False):
        label = tk.Label(frame, text=label_text, font=("Arial", 14, "bold"))
        label.pack()

        canvas = tk.Canvas(frame, width=280, height=280)
        canvas.pack(pady=10)

        filename_label = tk.Label(frame, text="", font=("Arial", 10))
        filename_label.pack()

        if is_condition:
            self.condition_canvas = canvas
            self.condition_filename_label = filename_label
        elif is_gt:
            self.gt_canvas = canvas
            self.gt_filename_label = filename_label
        elif is_generated:
            self.generated_canvas = canvas
            self.generated_filename_label = filename_label

    def select_root(self):
        path = filedialog.askdirectory(title="Selecciona la carpeta raíz")
        if path:
            self.root_path = path
            condition_path = os.path.join(path, "Condition")
            if os.path.exists(condition_path):
                self.classes = self.get_subfolders(condition_path)
            else:
                self.classes = []

            self.class_combobox['values'] = self.classes
            if self.classes:
                self.class_combobox.current(0)
                self.load_images()
                self.update_image_count()
            else:
                self.condition_images = []
                self.gt_images = []
                self.generated_images = []
                self.update_images()
                self.update_image_count()

    def get_subfolders(self, path):
        return sorted([name for name in os.listdir(path) if os.path.isdir(os.path.join(path, name))])

    def on_class_change(self, event=None):
        self.current_index = 0
        self.load_images()
        self.update_images()
        self.update_image_count()

    def load_images(self):
        selected_class = self.class_combobox.get()
        if self.root_path and selected_class:
            self.condition_images = self.get_image_paths(os.path.join(self.root_path, "Condition", selected_class))
            self.gt_images = self.get_image_paths(os.path.join(self.root_path, "Ground Truth", selected_class))
            self.generated_images = self.get_image_paths(os.path.join(self.root_path, "Generated Images", selected_class))
        else:
            self.condition_images = []
            self.gt_images = []
            self.generated_images = []

    def get_image_paths(self, base_path):
        image_paths = []
        if not os.path.exists(base_path):
            return []
        for root, dirs, files in os.walk(base_path):
            for file in sorted(files):
                if file.lower().endswith((".png", ".jpg", ".jpeg")):
                    image_paths.append(os.path.join(root, file))
        return image_paths

    def update_images(self):
        if self.condition_images and self.gt_images and self.generated_images:
            max_len = min(len(self.condition_images), len(self.gt_images), len(self.generated_images))
            if max_len == 0:
                self.clear_canvases()
                return

            if self.current_index < 0:
                self.current_index = 0
            if self.current_index >= max_len:
                self.current_index = max_len - 1

            self.show_image(self.condition_canvas, self.condition_images[self.current_index])
            self.show_image(self.gt_canvas, self.gt_images[self.current_index])
            self.show_image(self.generated_canvas, self.generated_images[self.current_index])

            self.condition_filename_label.config(text=os.path.basename(self.condition_images[self.current_index]))
            self.gt_filename_label.config(text=os.path.basename(self.gt_images[self.current_index]))
            self.generated_filename_label.config(text=os.path.basename(self.generated_images[self.current_index]))
        else:
            self.clear_canvases()

    def clear_canvases(self):
        for canvas in [self.condition_canvas, self.gt_canvas, self.generated_canvas]:
            canvas.delete("all")
            canvas.image = None
        self.condition_filename_label.config(text="")
        self.gt_filename_label.config(text="")
        self.generated_filename_label.config(text="")

    def show_image(self, canvas, path):
        image = Image.open(path)
        image = image.resize((280, 280), resample=0)
        photo = ImageTk.PhotoImage(image)
        canvas.image = photo
        canvas.delete("all")
        canvas.create_image(0, 0, anchor=tk.NW, image=photo)

    def next_image(self):
        if self.condition_images and self.gt_images and self.generated_images:
            max_len = min(len(self.condition_images), len(self.gt_images), len(self.generated_images))
            if self.current_index < max_len - 1:
                self.current_index += 1
                self.update_images()

    def prev_image(self):
        if self.condition_images and self.gt_images and self.generated_images:
            if self.current_index > 0:
                self.current_index -= 1
                self.update_images()

    def update_image_count(self):
        count_cond = len(self.condition_images)
        count_gt = len(self.gt_images)
        count_gen = len(self.generated_images)
        self.image_count_label.config(
            text=f"Condition: {count_cond} | Ground Truth: {count_gt} | Generated: {count_gen}"
        )

    def calculate_metrics(self):
        if not self.gt_images or not self.generated_images:
            return

        min_len = min(len(self.gt_images), len(self.generated_images))
        ssim_vals, psnr_vals, mse_vals = [], [], []

        for i in range(min_len):
            img1 = Image.open(self.gt_images[i]).convert("L")
            img2 = Image.open(self.generated_images[i]).convert("L")

            if img1.size != img2.size:
                continue

            arr1 = np.array(img1, dtype=np.float32) / 255.0
            arr2 = np.array(img2, dtype=np.float32) / 255.0

            ssim_vals.append(ssim(arr1, arr2, data_range=1.0))
            psnr_vals.append(peak_signal_noise_ratio(arr1, arr2, data_range=1.0))
            mse_vals.append(mean_squared_error(arr1, arr2))

        self.metrics_text.config(state="normal")
        self.metrics_text.delete("1.0", tk.END)
        self.metrics_text.insert(tk.END, f"Cantidad evaluada: {min_len}\n\n")
        self.metrics_text.insert(tk.END, f"SSIM promedio: {np.mean(ssim_vals):.4f}\n")
        self.metrics_text.insert(tk.END, f"PSNR promedio: {np.mean(psnr_vals):.2f} dB\n")
        self.metrics_text.insert(tk.END, f"MSE promedio:  {np.mean(mse_vals):.4f}\n")
        self.metrics_text.config(state="disabled")

    def exportar_metricas_todas_las_clases(self):
        if self.root_path:
            calculate_metrics_all_classes(self.root_path)
            messagebox.showinfo("Exportación completa", "Métricas por clase guardadas correctamente.")
        else:
            print("No se ha seleccionado carpeta raíz.")

def calculate_metrics_all_classes(root_path, output_csv="resultados_metricas.csv"):
    gt_root = os.path.join(root_path, "Ground Truth")
    gen_root = os.path.join(root_path, "Generated Images")

    if not os.path.exists(gt_root) or not os.path.exists(gen_root):
        print("No se encontraron las carpetas requeridas.")
        return

    clases = sorted(os.listdir(gt_root))
    resultados = []

    for clase in clases:
        gt_dir = os.path.join(gt_root, clase)
        gen_dir = os.path.join(gen_root, clase)

        if not os.path.isdir(gt_dir) or not os.path.isdir(gen_dir):
            continue

        gt_images = sorted([os.path.join(gt_dir, f) for f in os.listdir(gt_dir)
                            if f.lower().endswith((".png", ".jpg", ".jpeg"))])
        gen_images = sorted([os.path.join(gen_dir, f) for f in os.listdir(gen_dir)
                             if f.lower().endswith((".png", ".jpg", ".jpeg"))])

        min_len = min(len(gt_images), len(gen_images))
        ssim_vals, psnr_vals, mse_vals = [], [], []
        skipped = 0

        for i in range(min_len):
            img1 = Image.open(gt_images[i]).convert("L")
            img2 = Image.open(gen_images[i]).convert("L")

            if img1.size != img2.size:
                skipped += 1
                continue

            arr1 = np.array(img1, dtype=np.float32) / 255.0
            arr2 = np.array(img2, dtype=np.float32) / 255.0

            ssim_vals.append(ssim(arr1, arr2, data_range=1.0))
            psnr_vals.append(peak_signal_noise_ratio(arr1, arr2, data_range=1.0))
            mse_vals.append(mean_squared_error(arr1, arr2))

        if ssim_vals:
            resultados.append({
                "Clase": clase,
                "SSIM": np.mean(ssim_vals),
                "PSNR": np.mean(psnr_vals),
                "MSE": np.mean(mse_vals),
                "Cantidad": len(ssim_vals),
                "Omitidas": skipped
            })

    output_path = os.path.join(root_path, output_csv)
    with open(output_path, "w", newline='') as f:
        writer = csv.DictWriter(f, fieldnames=["Clase", "SSIM", "PSNR", "MSE", "Cantidad", "Omitidas"])
        writer.writeheader()
        writer.writerows(resultados)

    print(f"Métricas por clase guardadas en: {output_path}")

if __name__ == "__main__":
    root = tk.Tk()
    app = EvaluadorDDPM3ColsSingleCombo(root)
    root.mainloop()
