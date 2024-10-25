import glob
import os
import torchvision
from PIL import Image
from tqdm import tqdm
from torch.utils.data.dataloader import DataLoader
from torch.utils.data.dataset import Dataset
import torchvision.transforms as transforms


class SingleImageDataset(Dataset):
    def __init__(self, im_paths, im_size=(28, 28)):
        """
        im_paths: Lista de rutas de carpetas donde se encuentran las imágenes.
        im_size: Tamaño al que se redimensionarán las imágenes.
        """
        self.im_paths = im_paths if isinstance(im_paths, list) else [im_paths]
        self.im_size = im_size
        self.images = self.load_images(self.im_paths)

    def load_images(self, im_paths):
        ims = []
        # Recorrer cada carpeta de la lista de rutas
        for im_path in im_paths:
            for fname in glob.glob(os.path.join(im_path, '*')):
                ims.append(fname)
        return ims

    def __len__(self):
        return len(self.images)

    def __getitem__(self, index):
        # Cargar la imagen y convertir a escala de grises
        im = Image.open(self.images[index]).convert('L')  
        
        # Redimensionar la imagen
        im = im.resize(self.im_size, Image.Resampling.LANCZOS)
        
        # Convertir a tensor
        im_tensor = transforms.ToTensor()(im)
        
        # Convertir el rango a [-1, 1]
        im_tensor = (2 * im_tensor) - 1
        
        return im_tensor
