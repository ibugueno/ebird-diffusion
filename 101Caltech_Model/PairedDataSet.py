import glob
import os
from PIL import Image
import torchvision.transforms as transforms
from torch.utils.data import Dataset

class PairedImageDataset(Dataset):
    def __init__(self, main_paths, condition_paths, im_size=(28, 28)):
        """
        main_paths: Lista de rutas a las carpetas de imágenes principales.
        condition_paths: Lista de rutas a las carpetas de imágenes de condición.
        im_size: Tamaño al que se redimensionarán las imágenes.
        """
        self.main_images = self.load_images(main_paths)
        self.condition_images = self.load_images(condition_paths)
        self.im_size = im_size

        # Verifica que ambas listas de imágenes tengan el mismo número de elementos
        assert len(self.main_images) == len(self.condition_images), "Las carpetas deben tener el mismo número de imágenes"

    def load_images(self, paths):
        """
        Cargar todas las imágenes de múltiples carpetas y subcarpetas.
        """
        ims = []
        for path in paths:
            # Recorrer todas las subcarpetas y cargar las imágenes
            for fname in glob.glob(os.path.join(path, '**', '*'), recursive=True):
                if os.path.isfile(fname):  # Asegurarse de que sea un archivo
                    ims.append(fname)
        return sorted(ims)  # Ordenar para que las imágenes se correspondan

    def __len__(self):
        return len(self.main_images)

    def __getitem__(self, index):
        # Cargar la imagen principal y de condición
        main_im = Image.open(self.main_images[index]).convert('L')  # Escala de grises
        condition_im = Image.open(self.condition_images[index]).convert('L')  # Escala de grises

        # Redimensionar ambas imágenes
        main_im = main_im.resize(self.im_size, Image.Resampling.LANCZOS)
        condition_im = condition_im.resize(self.im_size, Image.Resampling.LANCZOS)

        # Convertir a tensor
        main_tensor = transforms.ToTensor()(main_im)
        condition_tensor = transforms.ToTensor()(condition_im)

        # Convertir el rango a [-1, 1]
        main_tensor = (2 * main_tensor) - 1
        condition_tensor = (2 * condition_tensor) - 1

        # Devolver ambos tensores
        return main_tensor, condition_tensor
