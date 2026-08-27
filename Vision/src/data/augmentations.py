import albumentations as A
import cv2

def get_field_simulation_pipeline(image_size=(224, 224)):
    """
    Returns an Albumentations pipeline simulating harsh agricultural field conditions.
    """
    return A.Compose([
        # Resize to expected input size
        A.Resize(height=image_size[0], width=image_size[1]),
        
        # Simulate shaky hands in the field
        A.OneOf([
            A.MotionBlur(p=1.0),
            A.AdvancedBlur(p=1.0),
        ], p=0.4),
        
        # Simulate cheap smartphone camera sensors (sensor noise)
        A.OneOf([
            A.GaussNoise(p=1.0),
            A.ISONoise(color_shift=(0.01, 0.05), intensity=(0.1, 0.5), p=1.0),
        ], p=0.4),
        
        # Simulate harsh sunlight, canopy shadows, and varying lighting
        A.OneOf([
            A.RandomBrightnessContrast(brightness_limit=0.3, contrast_limit=0.3, p=1.0),
            A.RandomShadow(p=1.0),
            A.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4, hue=0.1, p=1.0),
        ], p=0.6)
    ])

def apply_augmentation(image_numpy, pipeline):
    """
    Applies the albumentations pipeline to a numpy image (HWC, BGR or RGB).
    """
    augmented = pipeline(image=image_numpy)
    return augmented['image']
