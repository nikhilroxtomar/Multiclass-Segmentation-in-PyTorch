import os
import cv2
import torch
import numpy as np
from tqdm import tqdm
from glob import glob

import torch.nn as nn
from model import build_unet
from utils import create_dir, seeding
from train import load_data


def add_dropout_to_decoder(model, p=0.5):
    """
    Inject Dropout2d after each ReLU in decoder conv_blocks.
    Applies to model.d1, d2, d3, d4.
    """
    for decoder_name in ["d1", "d2", "d3", "d4"]:
        decoder_block = getattr(model, decoder_name)
        conv_block = decoder_block.conv.conv

        new_layers = []
        for layer in conv_block:
            new_layers.append(layer)
            if isinstance(layer, nn.ReLU):
                new_layers.append(nn.Dropout2d(p))

        decoder_block.conv.conv = nn.Sequential(*new_layers)

def index_to_rgb_mask(mask, colormap):
    h, w = mask.shape
    rgb_mask = np.zeros((h, w, 3), dtype=np.uint8)
    for class_id, rgb in enumerate(colormap):
        rgb_mask[mask == class_id] = rgb
    return rgb_mask


def mc_dropout_inference(model, image_tensor, T=20):
    """
    Performs T stochastic forward passes with MC Dropout.
    Returns mean and std deviation over softmax predictions.
    """
    model.train()
    preds = []

    with torch.no_grad():
        for _ in range(T):
            output = model(image_tensor)
            probs = torch.softmax(output, dim=1)
            preds.append(probs.cpu().numpy())

    preds = np.stack(preds, axis=0)  # [T, C, H, W]
    mean_prob = np.mean(preds, axis=0)[0]  # [C, H, W]
    std_prob = np.std(preds, axis=0)[0]    # [C, H, W]

    return mean_prob, std_prob

def run_uncertainty(model, test_x, size, colormap, save_dir, T=20):
    os.makedirs(save_dir, exist_ok=True)

    for path in tqdm(test_x, total=len(test_x)):
        name = path.split("/")[-1].split(".")[0]

        image = cv2.imread(path, cv2.IMREAD_COLOR)
        image = cv2.resize(image, size)
        orig_image = image.copy()

        input_tensor = np.transpose(image, (2, 0, 1)) / 255.0  # [C, H, W]
        input_tensor = np.expand_dims(input_tensor, axis=0).astype(np.float32)
        input_tensor = torch.from_numpy(input_tensor).to(device)

        mean_prob, std_prob = mc_dropout_inference(model, input_tensor, T=T)

        pred_mask = np.argmax(mean_prob, axis=0)  # [H, W]
        uncertainty_map = np.mean(std_prob, axis=0)  # [H, W]

        # Normalize uncertainty heatmap to 0–255
        norm_unc = (uncertainty_map * 255 / np.max(uncertainty_map)).astype(np.uint8)
        heatmap = cv2.applyColorMap(norm_unc, cv2.COLORMAP_JET)

        rgb_pred = index_to_rgb_mask(pred_mask, colormap)

        # Create joint visualization
        line = np.ones((size[1], 10, 3), dtype=np.uint8) * 255
        cat = np.concatenate([orig_image, line, rgb_pred, line, heatmap], axis=1)

        cv2.imwrite(f"{save_dir}/joint/{name}.jpg", cat)
        cv2.imwrite(f"{save_dir}/uncertainty/{name}.jpg", heatmap)

if __name__ == "__main__":
    seeding(42)

    # Config
    image_w, image_h = 256, 256
    size = (image_w, image_h)
    dataset_path = "./Weeds-Dataset/weed_augmented"

    colormap = [
        [0, 0, 0],      # Background
        [0, 0, 128],    # Weed class 1
        [0, 128, 0]     # Weed class 2
    ]
    num_classes = len(colormap)

    # Load model
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = build_unet(num_classes=num_classes)
    model = model.to(device)

    # Load pre-trained weights
    checkpoint_path = "files/checkpoint.pth"
    model.load_state_dict(torch.load(checkpoint_path, map_location=device)["model_state_dict"])

    # Inject dropout into decoder
    add_dropout_to_decoder(model, p=0.5)

    # Load test data
    (train_x, train_y), (valid_x, valid_y), (test_x, test_y) = load_data(dataset_path)

    # Run uncertainty estimation
    save_path = "results_uncertainty"
    for item in ["joint", "uncertainty"]:
        create_dir(os.path.join(save_path, item))
    run_uncertainty(model, test_x, size, colormap, save_path, T=20)