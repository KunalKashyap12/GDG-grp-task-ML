import os
import sys
sys.path.insert(0, os.path.abspath("."))
import shutil
import tempfile
import numpy as np
import torch
from PIL import Image

from ml_training.utils.augmentations import (
    RandomRotationWithFill,
    FieldBackgroundVariation,
    get_train_transforms,
    get_eval_transforms
)

def test_random_rotation_with_fill():
    print("Testing RandomRotationWithFill...")
    # Solid olive/green image
    solid_color = (80, 140, 70)
    img = Image.new("RGB", (200, 200), solid_color)

    # 1. Mean fill mode
    rot_mean = RandomRotationWithFill(degrees=45, mode="mean")
    out_mean = rot_mean(img)
    corner_pixel = out_mean.getpixel((0, 0))
    print("Mean mode corner pixel:", corner_pixel)
    assert corner_pixel != (0, 0, 0), "Corner pixel should NOT be black!"
    assert abs(corner_pixel[0] - solid_color[0]) <= 5
    assert abs(corner_pixel[1] - solid_color[1]) <= 5
    assert abs(corner_pixel[2] - solid_color[2]) <= 5

    # 2. Reflect mode
    rot_reflect = RandomRotationWithFill(degrees=45, mode="reflect")
    out_reflect = rot_reflect(img)
    corner_ref = out_reflect.getpixel((0, 0))
    print("Reflect mode corner pixel:", corner_ref)
    assert corner_ref != (0, 0, 0), "Corner pixel should NOT be black!"

    print("[PASS] RandomRotationWithFill passed!")

def test_field_background_variation():
    print("\nTesting FieldBackgroundVariation...")
    # Create an image with a central green leaf (~40% coverage) on a neutral white lab background
    w, h = 224, 224
    arr = np.full((h, w, 3), 240, dtype=np.uint8) # White background
    # Add green leaf ellipse in center
    import cv2
    cv2.ellipse(arr, (w // 2, h // 2), (w // 3, h // 4), 30, 0, 360, (45, 150, 40), -1)
    img_leaf = Image.fromarray(arr)

    # 1. Procedural fallback (when backgrounds_dir is missing)
    bg_aug_procedural = FieldBackgroundVariation(p=1.0, backgrounds_dir="non_existent_dir")
    aug_img = bg_aug_procedural(img_leaf)
    aug_arr = np.array(aug_img)
    # Check that corners (which were white 240) were replaced
    corner_orig = arr[0, 0]
    corner_new = aug_arr[0, 0]
    print(f"Original corner: {corner_orig}, Replaced background corner: {corner_new}")
    assert not np.array_equal(corner_orig, corner_new), "White background corner should have been replaced!"
    # Check that leaf center was preserved
    center_orig = arr[h // 2, w // 2]
    center_new = aug_arr[h // 2, w // 2]
    print(f"Original leaf center: {center_orig}, Augmented leaf center: {center_new}")
    assert np.all(np.abs(center_orig.astype(int) - center_new.astype(int)) < 15), "Leaf center should be preserved!"

    # 2. Leaf coverage gating:
    # A) Leaf coverage < 10% (almost pure white) -> Should NOT be replaced
    white_img = Image.new("RGB", (224, 224), (250, 250, 250))
    aug_white = bg_aug_procedural(white_img)
    assert np.array_equal(np.array(white_img), np.array(aug_white)), "Image with <10% leaf coverage should not be modified!"

    # B) Leaf coverage > 90% (almost pure green) -> Should NOT be replaced
    green_img = Image.new("RGB", (224, 224), (40, 160, 40))
    aug_green = bg_aug_procedural(green_img)
    assert np.array_equal(np.array(green_img), np.array(aug_green)), "Image with >90% leaf coverage should not be modified!"

    # 3. Real background replacement from folder
    temp_dir = tempfile.mkdtemp()
    try:
        # Create a mock background image
        mock_bg = Image.new("RGB", (400, 400), (180, 50, 50)) # Distinct red background
        mock_bg.save(os.path.join(temp_dir, "bg1.jpg"))

        bg_aug_folder = FieldBackgroundVariation(p=1.0, backgrounds_dir=temp_dir)
        aug_folder_img = bg_aug_folder(img_leaf)
        aug_folder_arr = np.array(aug_folder_img)
        corner_replaced = aug_folder_arr[0, 0]
        print(f"Folder background corner: {corner_replaced}")
        # Corner should now have high red from mock_bg
        assert corner_replaced[0] > 150, "Corner should be replaced from background folder!"
    finally:
        shutil.rmtree(temp_dir)

    print("[PASS] FieldBackgroundVariation passed!")

def test_pipeline_and_resolutions():
    print("\nTesting resolutions (224, 288, 320)...")
    for size in [224, 288, 320]:
        train_t = get_train_transforms(img_size=size)
        eval_t = get_eval_transforms(img_size=size)

        dummy_img = Image.new("RGB", (400, 300), (60, 140, 50))
        t_out = train_t(dummy_img)
        e_out = eval_t(dummy_img)

        assert isinstance(t_out, torch.Tensor)
        assert isinstance(e_out, torch.Tensor)
        assert t_out.shape == (3, size, size), f"Expected train transform shape (3, {size}, {size}), got {t_out.shape}"
        assert e_out.shape == (3, size, size), f"Expected eval transform shape (3, {size}, {size}), got {e_out.shape}"
        print(f"  - img_size={size}: train shape={t_out.shape}, eval shape={e_out.shape} [OK]")

    print("[PASS] Transforms and resolutions passed!")

if __name__ == "__main__":
    test_random_rotation_with_fill()
    test_field_background_variation()
    test_pipeline_and_resolutions()
    print("\nALL TASK 4 TESTS PASSED SUCCESSFULLY!")
