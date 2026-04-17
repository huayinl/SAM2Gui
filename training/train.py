"""
# finetune both mask decoder and prompt encoder:
python train.py --finetune_mode mask_decoder
# finetune only mask decoder:
python train.py --finetune_mode mask_decoder
# finetune only prompt encoder:
python train.py --finetune_mode prompt_encoder
# finetune all parameters:
python train.py --finetune_mode all

checkpoint_path = r"D:\huayin\SAM2Gui\checkpoints\sam2.1_hiera_large.pt"
train_data_dir = r"D:\huayin\SAM2Gui\data\AH2\train_data_dir"
val_data_dir = r"D:\huayin\SAM2Gui\data\AH2\val_data_dir"
output_dir = "finetuned_checkpoints"
"""

import os
import torch
import argparse
from torch.utils.data import DataLoader, Dataset
import cv2
from segment_anything_2.sam2.build_sam import build_sam2
from segment_anything_2.sam2.sam2_image_predictor import SAM2ImagePredictor



# Paths

# Default paths
default_checkpoint_path = r"D:\huayin\SAM2Gui\checkpoints\sam2.1_hiera_large.pt"
default_train_data_dir = r"D:\huayin\SAM2Gui\data\AH2\train_data_dir"
default_val_data_dir = r"D:\huayin\SAM2Gui\data\AH2\val_data_dir"
default_output_dir = "finetuned_checkpoints"


# Parameter-efficient finetuning mode as CLI argument
# Options: "all", "mask_decoder", "prompt_encoder", or comma-separated list of substrings

parser = argparse.ArgumentParser(description="SAM2 Finetuning Script")
parser.add_argument(
    "--finetune_mode",
    type=str,
    default="mask_decoder",
    help="Which parameters to finetune: 'all', 'mask_decoder', 'prompt_encoder', or comma-separated substrings (e.g. 'mask_decoder,prompt_encoder')"
)
parser.add_argument(
    "--checkpoint_path",
    type=str,
    default=default_checkpoint_path,
    help="Path to the model checkpoint (.pt file)"
)
parser.add_argument(
    "--train_data_dir",
    type=str,
    default=default_train_data_dir,
    help="Path to the training data directory (should contain images/ and masks/)"
)
parser.add_argument(
    "--val_data_dir",
    type=str,
    default=default_val_data_dir,
    help="Path to the validation data directory (should contain images/ and masks/)"
)
parser.add_argument(
    "--output_dir",
    type=str,
    default=default_output_dir,
    help="Directory to save finetuned checkpoints"
)
args = parser.parse_args()
if "," in args.finetune_mode:
    finetune_mode = [s.strip() for s in args.finetune_mode.split(",")]
else:
    finetune_mode = args.finetune_mode

checkpoint_path = args.checkpoint_path
train_data_dir = args.train_data_dir
val_data_dir = args.val_data_dir
output_dir = args.output_dir
if not os.path.exists(output_dir):
    os.makedirs(output_dir)
    print(f"Created output directory: {output_dir}")

# Hyperparameters
batch_size = 8
num_epochs = 10
learning_rate = 1e-4


# Minimal CustomDataset for image/mask pairs
class CustomDataset(Dataset):
    def __init__(self, root_dir):
        self.images_dir = os.path.join(root_dir, "images")
        self.masks_dir = os.path.join(root_dir, "masks")
        self.image_files = sorted([f for f in os.listdir(self.images_dir) if f.endswith('.png') or f.endswith('.jpg')])
    def __len__(self):
        return len(self.image_files)
    def __getitem__(self, idx):
        img_path = os.path.join(self.images_dir, self.image_files[idx])
        mask_path = os.path.join(self.masks_dir, self.image_files[idx].replace('.png', '_mask.png').replace('.jpg', '_mask.png'))
        image = cv2.imread(img_path)[..., ::-1]  # BGR to RGB
        mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
        image = torch.tensor(image).permute(2, 0, 1).float() / 255.0
        mask = torch.tensor(mask).unsqueeze(0).float() / 255.0
        return image, mask


print(f"Loading training dataset from: {train_data_dir}")
train_dataset = CustomDataset(train_data_dir)
print(f"Loading validation dataset from: {val_data_dir}")
val_dataset = CustomDataset(val_data_dir)
print(f"Number of training samples: {len(train_dataset)}")
print(f"Number of validation samples: {len(val_dataset)}")
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=batch_size)



# Model (use official SAM2 build)

print("Building SAM2 model...")
model_cfg = "sam2_hiera_l.yaml"  # Adjust path if needed
model = build_sam2(model_cfg, checkpoint_path, device="cuda" if torch.cuda.is_available() else "cpu")
print("Model built and checkpoint loaded.")

# Parameter-efficient finetuning: freeze all except selected modules
if finetune_mode != "all":
    print(f"Freezing all parameters except: {finetune_mode}")
    for name, param in model.named_parameters():
        param.requires_grad = False
        if finetune_mode == "mask_decoder" and "mask_decoder" in name:
            print(f"Unfreezing {name}")
            param.requires_grad = True
        elif finetune_mode == "prompt_encoder" and "prompt_encoder" in name:
            print(f"Unfreezing {name}")
            param.requires_grad = True
        elif isinstance(finetune_mode, list):
            for key in finetune_mode:
                if key in name:
                    print(f"Unfreezing {name}")
                    param.requires_grad = True

# Optimizer: only update trainable parameters
optimizer = torch.optim.Adam(
    filter(lambda p: p.requires_grad, model.parameters()), lr=learning_rate
)

# Training loop
print("Starting training...")
for epoch in range(num_epochs):
    print(f"Epoch {epoch+1}/{num_epochs}...")
    model.train()
    running_loss = 0.0
    for i, batch in enumerate(train_loader):
        inputs, targets = batch
        inputs, targets = inputs.to(model.device), targets.to(model.device)
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = model.compute_loss(outputs, targets)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()
        if (i+1) % 10 == 0 or (i+1) == len(train_loader):
            print(f"  Batch {i+1}/{len(train_loader)} - Loss: {loss.item():.4f}")
    print(f"Epoch {epoch+1} training loss: {running_loss / len(train_loader):.4f}")

    # Validation (optional)
    model.eval()
    with torch.no_grad():
        val_loss = 0
        for j, batch in enumerate(val_loader):
            inputs, targets = batch
            inputs, targets = inputs.to(model.device), targets.to(model.device)
            outputs = model(inputs)
            val_loss += model.compute_loss(outputs, targets).item()
        print(f"Epoch {epoch+1} validation loss: {val_loss / len(val_loader):.4f}")

    # Save checkpoint
    os.makedirs(output_dir, exist_ok=True)
    ckpt_path = os.path.join(output_dir, f"sam2_finetuned_epoch{epoch+1}.pt")
    torch.save(model.state_dict(), ckpt_path)
    print(f"Saved checkpoint: {ckpt_path}")

print("Finetuning complete.")