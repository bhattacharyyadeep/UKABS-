import os
import numpy as np
import torch
import nibabel as nib
from torch.utils.data import DataLoader
from sklearn.model_selection import train_test_split
from tqdm import tqdm
import matplotlib.pyplot as plt
from albumentations import Compose, Resize
from torch.nn.modules.loss import CrossEntropyLoss
from met_loss import DiceLoss, calculate_metric_percase
import archs

class ACDCDataset:
    def __init__(self, image_paths, mask_paths, input_h=256, input_w=256):
        self.data = []
        self.input_h = input_h
        self.input_w = input_w
        self.transform = Compose([Resize(input_h, input_w)])
        
        for img_path, mask_path in zip(image_paths, mask_paths):
            if '4d' not in img_path.lower():
                img_vol = nib.load(img_path).get_fdata()
                mask_vol = nib.load(mask_path).get_fdata().astype(np.uint8)
                depth = min(img_vol.shape[2], mask_vol.shape[2])
                for z in range(depth):
                    img_slice = img_vol[:, :, z]
                    mask_slice = mask_vol[:, :, z]
                    self.data.append((img_slice, mask_slice))

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        img, mask = self.data[idx]
        img = (img - img.mean()) / (img.std() + 1e-8)
        img = np.expand_dims(img, axis=-1)
        augmented = self.transform(image=img, mask=mask)
        img = augmented['image'].transpose(2, 0, 1)
        mask = augmented['mask']
        return torch.tensor(img, dtype=torch.float32), torch.tensor(mask, dtype=torch.long)

def get_dataset_files(data_dir):
    if not os.path.exists(data_dir):
        raise FileNotFoundError(f"Dataset directory {data_dir} does not exist.")
    img_files = []
    mask_files = []
    patient_folders = sorted([f for f in os.listdir(data_dir) if f.startswith('patient') and os.path.isdir(os.path.join(data_dir, f))])
    
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
    
    print(f"Found {len(img_files)} image-mask pairs")
    return img_files, mask_files

def visualize_slice(img, mask, pred, output_dir, idx):
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    axes[0].imshow(img[0, :, :], cmap='gray')
    axes[0].set_title('Input Image')
    axes[0].axis('off')
    
    axes[1].imshow(mask, cmap='viridis')
    axes[1].set_title('Ground Truth Mask')
    axes[1].axis('off')
    
    axes[2].imshow(pred, cmap='viridis')
    axes[2].set_title('Predicted Mask')
    axes[2].axis('off')
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f'vis_slice_{idx:04d}.png'))
    plt.close()

def validate_and_visualize(loader, model, criterion_ce, criterion_dice, output_dir):
    model.eval()
    avg_loss = 0
    avg_dice = 0
    avg_hd95 = 0
    class_dice = {1: 0, 2: 0, 3: 0}  # RV, Myo, LV
    count = 0
    
    os.makedirs(os.path.join(output_dir, 'visualizations'), exist_ok=True)
    
    with torch.no_grad():
        for idx, (img, mask) in enumerate(tqdm(loader, desc="Validating and Visualizing")):
            img, mask = img.cuda(), mask.cuda()
            output = model(img)
            loss_ce = criterion_ce(output, mask)
            loss_dice = criterion_dice(output, mask, softmax=True)
            loss = 0.5 * loss_ce + 0.5 * loss_dice
            
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
            
            # Visualize each slice
            for b in range(img.size(0)):  # Iterate over batch
                img_np = img[b].cpu().numpy()  # [C=1, H, W]
                mask_np = mask[b].cpu().numpy()  # [H, W]
                pred_np = pred[b].cpu().numpy()  # [H, W]
                visualize_slice(img_np, mask_np, pred_np, os.path.join(output_dir, 'visualizations'), idx * loader.batch_size + b)
    
    avg_loss /= count
    avg_dice /= count
    avg_hd95 /= count
    class_dice = {k: v / count for k, v in class_dice.items()}
    return avg_loss, avg_dice, avg_hd95, class_dice

def main():
    data_root = '/home/deep/Documents/ACDC/'  # Replace with actual path to ACDC dataset
    output_dir = 'ACDC_'  # Match training output directory
    config = {
        'batch_size': 2,
        'input_h': 256,
        'input_w': 256,
        'num_classes': 4,  # Background, RV, Myo, LV
        'input_channels': 1,
        'deep_supervision': False,
        'no_kan': False
    }

    # Load dataset files
    image_paths, mask_paths = get_dataset_files(data_root)
    if not image_paths:
        raise FileNotFoundError(f"No image-mask pairs found in {data_root}.")
    
    # Split dataset
    _, val_img_paths, _, val_mask_paths = train_test_split(
        image_paths, mask_paths, test_size=0.2, random_state=42
    )

    # Create validation dataset and loader
    val_dataset = ACDCDataset(val_img_paths, val_mask_paths, config['input_h'], config['input_w'])
    val_loader = DataLoader(val_dataset, batch_size=config['batch_size'], shuffle=False, num_workers=4)

    # Initialize model
    model = archs.UKABS(
        num_classes=config['num_classes'],
        input_channels=config['input_channels'],
        deep_supervision=config['deep_supervision'],
        embed_dims=[256, 320, 512],
        no_kan=config['no_kan']
    ).cuda()

    # Load best model checkpoint
    checkpoint_path = '/home/deep/Documents/UKAN_repo/U-KAN/Seg_UKAN/acdc_outputs3/best_model.pth'
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"Checkpoint {checkpoint_path} not found.")
    model.load_state_dict(torch.load(checkpoint_path))
    print(f"Loaded model from {checkpoint_path}")

    # Initialize loss functions
    criterion_ce = CrossEntropyLoss().cuda()
    criterion_dice = DiceLoss(config['num_classes']).cuda()

    # Run validation and generate visualizations
    val_loss, val_dice, val_hd95, val_class_dice = validate_and_visualize(
        val_loader, model, criterion_ce, criterion_dice, output_dir
    )

    # Print results
    print(f"Validation Results: "
          f"Loss = {val_loss:.4f}, Dice = {val_dice:.4f}, "
          f"RV = {val_class_dice[1]:.4f}, Myo = {val_class_dice[2]:.4f}, LV = {val_class_dice[3]:.4f}, "
          f"HD95 = {val_hd95:.2f}")

if __name__ == '__main__':
    main()
