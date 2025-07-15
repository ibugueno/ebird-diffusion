import os
import glob
import random
from torch.utils.data import Dataset
from PIL import Image
import torchvision.transforms as transforms

class PairedImageDataset(Dataset):
    def __init__(self, main_paths, condition_paths, im_size=(28, 28), subset_ratio=1.0):
        """
        main_paths: Lista de rutas a las carpetas de imágenes principales (cada carpeta es una clase).
        condition_paths: Lista de rutas a las carpetas de imágenes de condición (misma estructura de clases).
        im_size: Tamaño al que se redimensionarán las imágenes.
        subset_ratio: Porcentaje de datos a utilizar (0.0 - 1.0) por clase.
        """
        self.main_images, self.condition_images = self.load_images(main_paths, condition_paths, subset_ratio)
        self.im_size = im_size
        
        assert len(self.main_images) == len(self.condition_images), "Las carpetas deben contener los mismos archivos"
        
        print(f"Dataset reducido al {subset_ratio*100}%: {len(self.main_images)} muestras emparejadas seleccionadas.")
        
        # Imprimir los archivos seleccionados para verificación
        for main_img, cond_img in zip(self.main_images, self.condition_images):
            print(f"Main: {main_img} <-> Condition: {cond_img}")

    def load_images(self, main_paths, condition_paths, subset_ratio):
        """
        Cargar imágenes de múltiples carpetas, aplicando la reducción por clase y asegurando el emparejamiento por nombre.
        """
        main_images = {}
        condition_images = {}
        
        for main_path, condition_path in zip(main_paths, condition_paths):
            main_files = sorted(glob.glob(os.path.join(main_path, '**', '*'), recursive=True))
            main_files = {os.path.basename(f): f for f in main_files if os.path.isfile(f)}
            
            condition_files = sorted(glob.glob(os.path.join(condition_path, '**', '*'), recursive=True))
            condition_files = {os.path.basename(f): f for f in condition_files if os.path.isfile(f)}
            
            common_files = sorted(set(main_files.keys()) & set(condition_files.keys()))
            
            if not common_files:
                continue
            
            num_samples = int(len(common_files) * subset_ratio)
            selected_files = random.sample(common_files, num_samples)
            
            main_images.update({name: main_files[name] for name in selected_files})
            condition_images.update({name: condition_files[name] for name in selected_files})
            
            print(f"Clase {os.path.basename(main_path)}: {num_samples}/{len(common_files)} imágenes emparejadas seleccionadas.")
        
        return list(main_images.values()), list(condition_images.values())


    def __len__(self):
        return len(self.main_images)

    def __getitem__(self, index):
        main_im = Image.open(self.main_images[index]).convert('L')
        condition_im = Image.open(self.condition_images[index]).convert('L')

        main_im = main_im.resize(self.im_size, Image.Resampling.LANCZOS)
        condition_im = condition_im.resize(self.im_size, Image.Resampling.LANCZOS)

        main_tensor = transforms.ToTensor()(main_im)
        condition_tensor = transforms.ToTensor()(condition_im)

        main_tensor = (2 * main_tensor) - 1
        condition_tensor = (2 * condition_tensor) - 1

        return main_tensor, condition_tensor
