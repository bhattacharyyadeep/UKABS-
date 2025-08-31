import cv2
import numpy as np
import os
from xml.etree import ElementTree as ET

class Dataset:
    def __init__(self, img_ids, img_dir, mask_dir, img_ext, mask_ext, num_classes, transform=None):
        self.img_ids = img_ids
        self.img_dir = img_dir
        self.mask_dir = mask_dir
        self.img_ext = img_ext
        self.mask_ext = mask_ext
        self.num_classes = num_classes
        self.transform = transform

    def __len__(self):
        return len(self.img_ids)

    def __getitem__(self, idx):
        img_id = self.img_ids[idx]
        img_path = os.path.join(self.img_dir, f"{img_id}{self.img_ext}")

        # Load image
        img = cv2.imread(img_path)
        if img is None:
            raise FileNotFoundError(f"Image not found: {img_path}")
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        # Load mask
        mask_path = os.path.join(self.mask_dir, f"{img_id}{self.mask_ext}")
        if self.mask_ext == '.xml':  # Handle MoNuSeg XML masks
            if not os.path.exists(mask_path):
                raise FileNotFoundError(f"Mask not found: {mask_path}")
            mask = self._parse_monuseg_xml(mask_path, img.shape[:2])
        else:
            # Handle image-based masks
            mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
            if mask is None:
                raise FileNotFoundError(f"Mask not found: {mask_path}")
            mask = (mask > 0).astype(np.uint8)

        # Ensure binary mask for num_classes=1
        if self.num_classes == 1:
            mask = (mask > 0).astype(np.uint8)
        else:
            raise NotImplementedError("Multi-class not supported yet")

        # Apply transformations
        if self.transform is not None:
            augmented = self.transform(image=img, mask=mask)
            img = augmented['image']
            mask = augmented['mask']

        # Convert to tensors
        img = img.transpose((2, 0, 1)).astype(np.float32)  # (C, H, W)
        mask = mask[np.newaxis, :, :].astype(np.float32)  # (1, H, W)

        return img, mask, img_id

    def _parse_monuseg_xml(self, xml_path, img_shape):
        """Parse MoNuSeg XML to create a binary mask."""
        mask = np.zeros(img_shape, dtype=np.uint8)
        tree = ET.parse(xml_path)
        root = tree.getroot()

        # Iterate over regions (nuclei)
        for region in root.findall('.//Region'):
            vertices = []
            for vertex in region.findall('.//Vertex'):
                x = float(vertex.get('X'))
                y = float(vertex.get('Y'))
                vertices.append([x, y])
            if vertices:
                vertices = np.array(vertices, dtype=np.int32)
                cv2.fillPoly(mask, [vertices], 1)  # Fill nucleus with 1

        return mask
