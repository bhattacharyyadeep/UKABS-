import os
import glob
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import nibabel as nib
from sklearn.model_selection import train_test_split
from tqdm import tqdm
import yaml
from collections import OrderedDict
import pandas as pd
from tensorboardX import SummaryWriter
from albumentations import Compose, Resize
from torch.optim import lr_scheduler
import archs
from torch.nn.modules.loss import CrossEntropyLoss
from met_loss import DiceLoss, calculate_metric_percase
import warnings
import matplotlib.pyplot as plt

# Suppress FutureWarning from timm (temporary; update timm for permanent fix)
warnings.filterwarnings("ignore", category=FutureWarning)

class ACDCDataset(Dataset):
    def __init__(self, image_paths, mask_paths, input_h=256, input_w=256):
        self.data = []
        self.input_h = input_h
        self.input_w = input_w
        self.transform = Compose([Resize(input_h, input_w)])
        
        for img_path, mask_path in zip(image_paths, mask_paths):
            # Load 3D images and masks (excluding 4D)
            if '4d' not in img_path.lower():
                img_vol = nib.load(img_path).get_fdata()
                mask_vol = nib.load(mask_path).get_fdata().astype(np.uint8)
                
                # Ensure the image and mask have the same depth
                depth = min(img_vol.shape[2], mask_vol.shape[2])
                
                # Iterate over slices (z-axis)
                for z in range(depth):
                    img_slice = img_vol[:, :, z]
                    mask_slice = mask_vol[:, :, z]
                    self.data.append((img_slice, mask_slice))

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        img, mask = self.data[idx]

        # Normalize image
        img = (img - img.mean()) / (img.std() + 1e-8)
        img = np.expand_dims(img, axis=-1)  # [H, W, 1]

        augmented = self.transform(image=img, mask=mask)
        img = augmented['image'].transpose(2, 0, 1)  # [C=1, H, W]
        mask = augmented['mask']

        img = torch.tensor(img, dtype=torch.float32)
        mask = torch.tensor(mask, dtype=torch.long)
        return img, mask

def get_dataset_files(data_dir):
    if not os.path.exists(data_dir):
        raise FileNotFoundError(f"Dataset directory {data_dir} does not exist. Please verify the path or download the ACDC dataset.")
    img_files = []
    mask_files = []
    patient_folders = sorted([f for f in os.listdir(data_dir) if f.startswith('patient') and os.path.isdir(os.path.join(data_dir, f))])
    t=0
    
    if not patient_folders:
        print(f"Warning: No patient folders found in {data_dir}")
        return [], []
    
    for patient in tqdm(patient_folders, desc="Scanning patient folders"):
        patient_dir = os.path.join(data_dir, patient)
        for frame in ['frame01', 'frame12']:
            img_path = os.path.join(patient_dir, f'{patient}_{frame}.nii.gz')
            mask_path = os.path.join(patient_dir, f'{patient}_{frame}_gt.nii.gz')
            
            if os.path.exists(img_path) and os.path.exists(mask_path):
                img_files.append(img_path)
                mask_files.append(mask_path)
            else:
                #print(f"Warning: Missing files for {patient}_{frame}: img={os.path.exists(img_path)}, mask={os.path.exists(mask_path)}")
                t=+1
                
    print(f"Found {len(img_files)} image-mask pairs")
    return img_files, mask_files

def train_one_epoch(loader, model, criterion_ce, criterion_dice, optimizer):
    model.train()
    avg_loss = 0
    avg_dice = 0
    class_dice = {1: 0, 2: 0, 3: 0}  # RV, Myo, LV
    count = 0
    for img, mask in tqdm(loader):
        img, mask = img.cuda(), mask.cuda()
        optimizer.zero_grad()
        output = model(img)
        loss_ce = criterion_ce(output, mask)
        loss_dice = criterion_dice(output, mask, softmax=True)
        loss = 0.5 * loss_ce + 0.5 * loss_dice
        
        # Calculate Dice for each class
        pred = torch.argmax(torch.softmax(output, dim=1), dim=1)
        dice_scores = {}
        for i in range(1, 4):  # Classes RV (1), Myo (2), LV (3)
            class_dice_i, _ = calculate_metric_percase((pred == i).cpu().numpy(), (mask == i).cpu().numpy())
            dice_scores[i] = class_dice_i
            class_dice[i] += class_dice_i
        avg_dice += sum(dice_scores.values()) / 3  # Average over the 3 classes
        
        loss.backward()
        optimizer.step()
        avg_loss += loss.item()
        count += 1
    avg_loss /= count
    avg_dice /= count
    class_dice = {k: v / count for k, v in class_dice.items()}
    return avg_loss, avg_dice, class_dice

def validate(loader, model, criterion_ce, criterion_dice):
    model.eval()
    avg_loss = 0
    avg_dice = 0
    avg_hd95 = 0
    class_dice = {1: 0, 2: 0, 3: 0}  # RV, Myo, LV
    count = 0
    with torch.no_grad():
        for img, mask in tqdm(loader):
            img, mask = img.cuda(), mask.cuda()
            output = model(img)
            loss_ce = criterion_ce(output, mask)
            loss_dice = criterion_dice(output, mask, softmax=True)
            loss = 0.5 * loss_ce + 0.5 * loss_dice
            
            # Calculate metrics per class
            pred = torch.argmax(torch.softmax(output, dim=1), dim=1)
            dice_scores = {}
            hd95_scores = {}
            for i in range(1, 4):  # Classes RV (1), Myo (2), LV (3)
                class_dice_i, class_hd95 = calculate_metric_percase((pred == i).cpu().numpy(), (mask == i).cpu().numpy())
                dice_scores[i] = class_dice_i
                hd95_scores[i] = class_hd95
                class_dice[i] += class_dice_i
                avg_hd95 += class_hd95
            avg_dice += sum(dice_scores.values()) / 3
            avg_loss += loss.item()
            count += 1
    avg_loss /= count
    avg_dice /= count
    avg_hd95 /= count
    class_dice = {k: v / count for k, v in class_dice.items()}
    return avg_loss, avg_dice, avg_hd95, class_dice

def plot_dice_scores(output_dir):
    log_df = pd.read_csv(os.path.join(output_dir, 'log.csv'))
    epochs = log_df['epoch']
    plt.figure(figsize=(10, 6))
    plt.plot(epochs, log_df['train_dice_rv'], label='Train Dice RV', color='blue')
    plt.plot(epochs, log_df['train_dice_myo'], label='Train Dice Myo', color='green')
    plt.plot(epochs, log_df['train_dice_lv'], label='Train Dice LV', color='red')
    plt.plot(epochs, log_df['val_dice_rv'], label='Val Dice RV', linestyle='--', color='blue')
    plt.plot(epochs, log_df['val_dice_myo'], label='Val Dice Myo', linestyle='--', color='green')
    plt.plot(epochs, log_df['val_dice_lv'], label='Val Dice LV', linestyle='--', color='red')
    plt.xlabel('Epoch')
    plt.ylabel('Dice Score')
    plt.title('Class-Wise Dice Scores Over Epochs')
    plt.legend()
    plt.grid(True)
    plt.savefig(os.path.join(output_dir, 'dice_scores.png'))
    plt.close()

def main():
    data_root = '/home/deep/Documents/ACDC/'  # Replace with the actual path to your ACDC dataset
    config = {
        'epochs': 400,
        'batch_size': 2,
        'input_h': 256,
        'input_w': 256,
        'num_classes': 4,  # Background, RV, Myo, LV
        'input_channels': 1,
        'deep_supervision': False,
        'lr': 1e-3,
        'data_dir': data_root,
        'output_dir': 'acdc_dualKAN3',
        'no_kan': False
    }

    os.makedirs(config['output_dir'], exist_ok=True)
    writer = SummaryWriter(config['output_dir'])

    with open(os.path.join(config['output_dir'], 'config.yml'), 'w') as f:
        yaml.dump(config, f)

    criterion_ce = CrossEntropyLoss().cuda()
    criterion_dice = DiceLoss(config['num_classes']).cuda()
    
    model = archs.UKABS(
        num_classes=config['num_classes'],
        input_channels=config['input_channels'],
        deep_supervision=config['deep_supervision'],
        embed_dims=[256, 320, 512],
        no_kan=config['no_kan']
    ).cuda()

    optimizer = optim.Adam(model.parameters(), lr=config['lr'])
    scheduler = lr_scheduler.CosineAnnealingLR(optimizer, T_max=config['epochs'])

    image_paths, mask_paths = get_dataset_files(config['data_dir'])
    if not image_paths:
        raise FileNotFoundError(f"No image-mask pairs found in {config['data_dir']}. Please verify the dataset path and structure.")
    train_img_paths, val_img_paths, train_mask_paths, val_mask_paths = train_test_split(
        image_paths, mask_paths, test_size=0.2, random_state=42
    )

    train_dataset = ACDCDataset(train_img_paths, train_mask_paths, config['input_h'], config['input_w'])
    val_dataset = ACDCDataset(val_img_paths, val_mask_paths, config['input_h'], config['input_w'])

    train_loader = DataLoader(train_dataset, batch_size=config['batch_size'], shuffle=True, num_workers=4)
    val_loader = DataLoader(val_dataset, batch_size=config['batch_size'], shuffle=False, num_workers=4)

    log = OrderedDict([
        ('epoch', []),
        ('train_loss', []),
        ('train_dice', []),
        ('train_dice_rv', []),
        ('train_dice_myo', []),
        ('train_dice_lv', []),
        ('val_loss', []),
        ('val_dice', []),
        ('val_dice_rv', []),
        ('val_dice_myo', []),
        ('val_dice_lv', []),
        ('val_hd95', []),
    ])

    best_dice = 0
    for epoch in range(config['epochs']):
        train_loss, train_dice, train_class_dice = train_one_epoch(train_loader, model, criterion_ce, criterion_dice, optimizer)
        val_loss, val_dice, val_hd95, val_class_dice = validate(val_loader, model, criterion_ce, criterion_dice)
        scheduler.step()

        print(f"Epoch {epoch+1}/{config['epochs']}: "
              f"Train Loss = {train_loss:.4f}, Train Dice = {train_dice:.4f}, "
              f"RV = {train_class_dice[1]:.4f}, Myo = {train_class_dice[2]:.4f}, LV = {train_class_dice[3]:.4f}, "
              f"Val Loss = {val_loss:.4f}, Val Dice = {val_dice:.4f}, "
              f"RV = {val_class_dice[1]:.4f}, Myo = {val_class_dice[2]:.4f}, LV = {val_class_dice[3]:.4f}, "
              f"Val HD95 = {val_hd95:.2f}")

        log['epoch'].append(epoch)
        log['train_loss'].append(train_loss)
        log['train_dice'].append(train_dice)
        log['train_dice_rv'].append(train_class_dice[1])
        log['train_dice_myo'].append(train_class_dice[2])
        log['train_dice_lv'].append(train_class_dice[3])
        log['val_loss'].append(val_loss)
        log['val_dice'].append(val_dice)
        log['val_dice_rv'].append(val_class_dice[1])
        log['val_dice_myo'].append(val_class_dice[2])
        log['val_dice_lv'].append(val_class_dice[3])
        log['val_hd95'].append(val_hd95)

        pd.DataFrame(log).to_csv(os.path.join(config['output_dir'], 'log.csv'), index=False)
        writer.add_scalar('Loss/train', train_loss, epoch)
        writer.add_scalar('Dice/train', train_dice, epoch)
        writer.add_scalar('Dice/train_rv', train_class_dice[1], epoch)
        writer.add_scalar('Dice/train_myo', train_class_dice[2], epoch)
        writer.add_scalar('Dice/train_lv', train_class_dice[3], epoch)
        writer.add_scalar('Loss/val', val_loss, epoch)
        writer.add_scalar('Dice/val', val_dice, epoch)
        writer.add_scalar('Dice/val_rv', val_class_dice[1], epoch)
        writer.add_scalar('Dice/val_myo', val_class_dice[2], epoch)
        writer.add_scalar('Dice/val_lv', val_class_dice[3], epoch)
        writer.add_scalar('HD95/val', val_hd95, epoch)

        if val_dice > best_dice:
            torch.save(model.state_dict(), os.path.join(config['output_dir'], 'best_model.pth'))
            best_dice = val_dice
            print("=> Saved best model")

    plot_dice_scores(config['output_dir'])

if __name__ == '__main__':
    main()
