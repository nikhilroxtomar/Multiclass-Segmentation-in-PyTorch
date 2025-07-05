import os, time
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
from operator import add
import numpy as np
from glob import glob
import cv2
from tqdm import tqdm
import torch
import torch.nn.functional as F
from model import build_unet
from utils import create_dir, seeding
from train import load_data
from sklearn.metrics import accuracy_score, f1_score, jaccard_score, precision_score, recall_score


def index_to_rgb_mask(mask, colormap):
    height, width = mask.shape
    rgb_mask = np.zeros((height, width, 3), dtype=np.uint8)

    for class_id, rgb in enumerate(colormap):
        rgb_mask[mask == class_id] = rgb

    return rgb_mask


def TTA(model, image_tensor):
    model.eval()
    predictions = []

    with torch.no_grad():
        # Original
        pred_orig = model(image_tensor)
        pred_orig = torch.softmax(pred_orig, dim=1)
        predictions.append(pred_orig)

        # Horizontal Flip
        img_hflip = torch.flip(image_tensor, dims=[3])
        pred_hflip = model(img_hflip)
        pred_hflip = torch.softmax(pred_hflip, dim=1)
        pred_hflip = torch.flip(pred_hflip, dims=[3])
        predictions.append(pred_hflip)

        # Vertical Flip
        img_vflip = torch.flip(image_tensor, dims=[2])
        pred_vflip = model(img_vflip)
        pred_vflip = torch.softmax(pred_vflip, dim=1)
        pred_vflip = torch.flip(pred_vflip, dims=[2])
        predictions.append(pred_vflip)

        # Horizontal + Vertical Flip
        img_hvflip = torch.flip(image_tensor, dims=[2, 3])
        pred_hvflip = model(img_hvflip)
        pred_hvflip = torch.softmax(pred_hvflip, dim=1)
        pred_hvflip = torch.flip(pred_hvflip, dims=[2, 3])
        predictions.append(pred_hvflip)

        # Average the softmax outputs
        mean_pred = torch.stack(predictions, dim=0).mean(dim=0)
        final_pred = torch.argmax(mean_pred, dim=1)
        final_pred = final_pred.squeeze(0).cpu().numpy().astype(np.uint8)

    return final_pred


def evaluate(model, save_path, test_x, test_y, size, colormap, classes):
    time_taken = []
    score = []

    for i, (x, y) in tqdm(enumerate(zip(test_x, test_y)), total=len(test_x)):
        name = y.split("/")[-1].split(".")[0]
        
        """ Image """
        image = cv2.imread(x, cv2.IMREAD_COLOR)
        image = cv2.resize(image, size)
        save_img = image
        image = np.transpose(image, (2, 0, 1))
        image = image/255.0 ## [C, H, W]
        image = np.expand_dims(image, axis=0) ## [1, C, H, W]
        image = image.astype(np.float32)
        image = torch.from_numpy(image)
        image = image.to(device)

        """ Mask """
        mask = cv2.imread(y, cv2.IMREAD_COLOR)
        mask = cv2.resize(mask, size)
        save_mask = mask

        output_mask = []
        for i, color in enumerate(colormap):
            cmap = np.all(np.equal(mask, color), axis=-1)
            output_mask.append(cmap)
        output_mask = np.stack(output_mask, axis=-1)
        output_mask = np.argmax(output_mask, axis=-1)

        with torch.no_grad():
            """ FPS calculation """
            start_time = time.time()

            """ Predict the mask """
            y_pred = TTA(model, image)

            end_time = time.time() - start_time
            time_taken.append(end_time)

        """ Save the image - mask - pred """
        line = np.ones((size[1], 10, 3)) * 255
        cat_images = np.concatenate([save_img, line, save_mask, line, index_to_rgb_mask(y_pred, colormap)], axis=1)
        cv2.imwrite(f"{save_path}/joint/{name}.jpg", cat_images)
        cv2.imwrite(f"{save_path}/mask/{name}.jpg", index_to_rgb_mask(y_pred, colormap))

        """ Calculate the score """
        y_true = output_mask.flatten()
        y_pred = y_pred.flatten()

        labels = [i for i in range(len(colormap))]

        """ Calculating the metrics values """
        f1_value = f1_score(y_true, y_pred, labels=labels, average=None, zero_division=0)
        jac_value = jaccard_score(y_true, y_pred, labels=labels, average=None, zero_division=0)

        score.append([f1_value, jac_value])

    """ Metrics values """
    score = np.array(score)
    score = np.mean(score, axis=0)
    print(score.shape)

    f = open("files/score_tta.csv", "w")
    f.write("Class,F1,Jaccard\n")

    l = ["Class", "F1", "IoU"]
    print(f"{l[0]:15s} {l[1]:10s} {l[2]:10s}")
    print("-"*35)

    for i in range(score.shape[1]):
        class_name = classes[i]
        f1 = score[0, i]
        jac = score[1, i]
        dstr = f"{class_name:15s}: {f1:1.5f} - {jac:1.5f}"
        print(dstr)
        f.write(f"{class_name:15s},{f1:1.5f},{jac:1.5f}\n")

    print("-"*35)
    class_mean = np.mean(score[:,1:], axis=-1)
    class_name = "Mean"
    f1 = class_mean[0]
    jac = class_mean[1]
    dstr = f"{class_name:15s}: {f1:1.5f} - {jac:1.5f}"
    print(dstr)
    f.write(f"{class_name:15s},{f1:1.5f},{jac:1.5f}\n")

    f.close()

    mean_time_taken = np.mean(time_taken)
    mean_fps = 1/mean_time_taken
    print("Mean FPS: ", mean_fps)

if __name__ == "__main__":
    """ Seeding """
    seeding(42)

    """ Hyperparameters """
    image_w = 256
    image_h = 256
    size = (image_w, image_h)
    dataset_path = "./Weeds-Dataset/weed_augmented"
    colormap = [
        [0, 0, 0],      # Background
        [0, 0, 128],    # Class 1
        [0, 128, 0]     # Class 2
    ]
    num_classes = len(colormap)
    classes = ["background", "Weed-1", "Weed-2"]

    """ Load the checkpoint """
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = build_unet(num_classes=num_classes)
    model = model.to(device)
    checkpoint_path = "files/checkpoint.pth"
    model.load_state_dict(torch.load(checkpoint_path, map_location=device)["model_state_dict"])
    model.eval()

    """ Test dataset """
    (train_x, train_y), (valid_x, valid_y), (test_x, test_y) = load_data(dataset_path)

    save_path = f"results_tta"
    for item in ["mask", "joint"]:
        create_dir(f"{save_path}/{item}")
    
    evaluate(model, save_path, test_x, test_y, size, colormap, classes)