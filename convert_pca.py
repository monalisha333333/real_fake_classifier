import os
import cv2
import argparse
import numpy as np
from sklearn.decomposition import PCA

# =========================
# Paths
# =========================
# input_dir = "/home/dnn3/Storage1/pythonCodeArea/Luna/digital_forensic/data-co-spy/in_the_wild/CC3M/1_Real"
# pca_dir = "/home/dnn3/Storage1/pythonCodeArea/Luna/digital_forensic/data-pca/in_the_wild/CC3M/1_Real"

parser = argparse.ArgumentParser()
parser.add_argument("--input_dir", type=str, required=True,
                    help="Directory containing input images")
parser.add_argument("--pca_dir", type=str, required=True,
                    help="Directory to save PCA images")

args = parser.parse_args()

input_dir = args.input_dir
pca_dir = args.pca_dir

os.makedirs(pca_dir, exist_ok=True)

# =========================
# Helper: PCA Major Component
# =========================
def compute_pca_major(img_rgb):
    """
    img_rgb: H x W x 3 float32 [0,1]
    returns: H x W float32 [0,1]
    """
    H, W, C = img_rgb.shape
    flat = img_rgb.reshape(-1, 3)

    pca = PCA(n_components=1)
    pc1 = pca.fit_transform(flat)

    pc1 = pc1.reshape(H, W)
    pc1 = (pc1 - pc1.min()) / (pc1.max() - pc1.min() + 1e-8)
    return pc1

# =========================
# Main Loop
# =========================
for fname in sorted(os.listdir(input_dir)):
    if not fname.lower().endswith((".png", ".jpg", ".jpeg")):
        continue
    
    img_path = os.path.join(input_dir, fname)
    img_bgr = cv2.imread(img_path)

    if img_bgr is None:
        print(f"Skipping unreadable file: {fname}")
        continue

    # Convert
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
    
    # Compute transforms
    pca_img = compute_pca_major(img_rgb)

    # Save (convert to uint8)
    cv2.imwrite(
        os.path.join(pca_dir, fname),
        (pca_img * 255).astype(np.uint8)
    )

    print(f"Processed: {fname}")

print("✅ Done!")
