from pathlib import Path
import numpy as np

import dlib
import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms
from torch.utils.data import DataLoader


from utils import load_and_preprocess_image


def collate_fn(batch):
    imgs = [item['image'] for item in batch if item['image'] is not None]
    targets = [item['label'] for item in batch if item['image'] is not None]
    filenames = [item['filename'] for item in batch if item['image'] is not None]
    imgs = torch.stack(imgs)
    targets = torch.stack(targets)
    return {'image': imgs, 'label': targets, 'filename': filenames}


def get_transforms():
    pre_trained_mean, pre_trained_std = [0.485, 0.456, 0.406], [0.229, 0.224, 0.225]
    
    train_transforms = transforms.Compose([
        transforms.RandomHorizontalFlip(),
        #     transforms.RandomVerticalFlip(),
        transforms.RandomAffine(degrees=40, scale=(.9, 1.1), shear=0),
        #     transforms.RandomRotation(degrees=40),
        transforms.RandomPerspective(distortion_scale=0.2),
        transforms.ColorJitter(brightness=0.5, contrast=0.5, saturation=0.5),
        transforms.ToTensor(),
        transforms.RandomErasing(scale=(0.02, 0.16), ratio=(0.3, 1.6)),
        transforms.Normalize(mean=pre_trained_mean, std=pre_trained_std),
    ])
    
    val_transforms = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=pre_trained_mean, std=pre_trained_std)
    ])
    return train_transforms, val_transforms


class FFDataset(Dataset):
    def __init__(self, filenames, filepath, transform, output_image_size=224, recompute=False):
        self.filenames = filenames
        self.transform = transform
        self.image_size = output_image_size
        self.recompute = recompute
        
        self.cached_path = Path(filepath)
        self.cached_path.mkdir(exist_ok=True)
        self.face_detector = dlib.get_frontal_face_detector()
    
    def __len__(self):
        return len(self.filenames)
    
    def __getitem__(self, idx: int):
        filename = self.filenames[idx]
        image_id = filename.stem
        filename = str(filename)
        label = 1 if 'fake' in filename.split('/') else 0
        
        preprocessed_filename = self.cached_path / f'processed_{image_id}.npy'
        
        if preprocessed_filename.is_file() and not self.recompute:
            image = np.load(preprocessed_filename)
        else:
            image = load_and_preprocess_image(filename, self.image_size, self.face_detector)
            if image is None:
                image = []
            np.save(preprocessed_filename, image)
        
        if len(image) == 0:
            return {'image': None, 'label': None ,'filename': filename}
        
        image = Image.fromarray(image)
        image = self.transform(image)
        label = torch.tensor(label)
        
        return {'image': image, 'label': label, 'filename': filename}
    

def create_dataloaders(params):
    train_transforms, val_transforms = get_transforms()
    train_dl = _create_dataloader(f'/datasets/{params["train_data"]}_deepfake', mode='train', batch_size=params['batch_size'],
                                  transformations=train_transforms)
    val_base_dl = _create_dataloader(f'/datasets/base_deepfake/val', mode='val', batch_size=params['batch_size'], transformations=val_transforms)
    val_augment_dl = _create_dataloader(f'/datasets/augment_deepfake/val', mode='val', batch_size=params['batch_size'], transformations=val_transforms)
    display_file_paths = [f'/datasets/{i}_deepfake/val' for i in ['base', 'augment']]
    display_dl_iter = iter(_create_dataloader(display_file_paths, mode='val', batch_size=32, transformations=val_transforms))
    
    return train_dl, val_base_dl, val_augment_dl, display_dl_iter


def _create_dataloader(file_paths, mode, batch_size, transformations, num_workers=80):
    if not isinstance(file_paths, list):
        file_paths = [file_paths]
    
    filenames = []
    for file_path in file_paths:
        data_path = Path(file_path)
    
        real_frame_filenames = _find_filenames(data_path / 'real/frames/', '*.png')
        fake_frame_filenames = _find_filenames(data_path / 'fake/frames/', '*.png')
        
        filenames += real_frame_filenames
        filenames += fake_frame_filenames
        
    # filenames = real_frame_filenames + fake_frame_filenames
    assert len(filenames) != 0, f'filenames are empty {filenames}'
    np.random.shuffle(filenames)
    
    ds = FFDataset(filenames, filepath=f'/datasets/precomputed/', transform=transformations, recompute=False)
    dl = DataLoader(ds, batch_size=batch_size, num_workers=num_workers, shuffle=True, collate_fn=collate_fn)
    
    print(f"{mode} data: {len(ds)}")
    
    return dl


def _find_filenames(file_dir_path, file_pattern): return list(file_dir_path.glob(file_pattern))

