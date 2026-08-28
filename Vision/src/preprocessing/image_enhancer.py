"""
Phase 1: Hardware-Resilient Enhancer
Corrects artifacts from budget Android smartphone cameras.
Must execute under 25ms to prevent thermal throttling on Edge NPUs.
"""

import cv2
import numpy as np
import time

class ImageEnhancer:
    def __init__(self, blur_threshold=45.0):
        self.blur_threshold = blur_threshold

    def check_quality(self, frame) -> bool:
        """
        Calculates the Laplacian variance to determine if the image is too blurry.
        Returns True if quality is acceptable (score >= threshold), False otherwise.
        """
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        blur_score = cv2.Laplacian(gray, cv2.CV_64F).var()
        # Even if it's blurry, we might not want to block entirely, but we log the quality
        self.last_blur_score = blur_score
        return blur_score >= self.blur_threshold

    def auto_white_balance(self, frame):
        """
        Applies Gray World assumption to correct severe color casts from cheap sensors.
        """
        result = frame.copy()
        avg_b = np.mean(result[:, :, 0])
        avg_g = np.mean(result[:, :, 1])
        avg_r = np.mean(result[:, :, 2])
        
        avg_gray = (avg_b + avg_g + avg_r) / 3.0
        
        # Avoid division by zero
        if avg_b == 0 or avg_g == 0 or avg_r == 0:
            return frame
            
        result[:, :, 0] = np.clip(result[:, :, 0] * (avg_gray / avg_b), 0, 255)
        result[:, :, 1] = np.clip(result[:, :, 1] * (avg_gray / avg_g), 0, 255)
        result[:, :, 2] = np.clip(result[:, :, 2] * (avg_gray / avg_r), 0, 255)
        
        return result

    def correct_illumination(self, frame):
        """
        Applies Contrast Limited Adaptive Histogram Equalization (CLAHE) on the LAB L-channel
        to normalize lighting across the image without distorting colors.
        """
        lab = cv2.cvtColor(frame, cv2.COLOR_BGR2LAB)
        l, a, b = cv2.split(lab)
        
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        cl = clahe.apply(l)
        
        limg = cv2.merge((cl, a, b))
        return cv2.cvtColor(limg, cv2.COLOR_LAB2BGR)

    def denoise_and_sharpen(self, frame):
        """
        Uses a fast Bilateral Filter to preserve edges while removing noise,
        followed by an Unsharp Mask for sharpening.
        Optimized via downscaling to meet the 25ms Edge NPU latency budget.
        """
        # Downscale for heavy filtering to meet extreme latency limits
        small_frame = cv2.resize(frame, (320, 320), interpolation=cv2.INTER_LINEAR)
        
        # Fast bilateral filtering
        denoised = cv2.bilateralFilter(small_frame, d=5, sigmaColor=50, sigmaSpace=50)
        
        # Unsharp Mask
        gaussian_blur = cv2.GaussianBlur(denoised, (0, 0), 2.0)
        sharpened = cv2.addWeighted(denoised, 1.5, gaussian_blur, -0.5, 0)
        
        # Upscale back to original size
        return cv2.resize(sharpened, (frame.shape[1], frame.shape[0]), interpolation=cv2.INTER_LINEAR)

    def process(self, frame):
        """
        Runs all hardware-resilient steps sequentially.
        Optimized by executing all heavy operations on a 320x320 scaled buffer.
        """
        start_time = time.perf_counter()
        
        # Immediate downscale for heavy processing
        original_shape = (frame.shape[1], frame.shape[0])
        working_frame = cv2.resize(frame, (320, 320), interpolation=cv2.INTER_LINEAR)
        
        is_clear = self.check_quality(working_frame)
        
        balanced = self.auto_white_balance(working_frame)
        illuminated = self.correct_illumination(balanced)
        
        # Fast bilateral filtering & Unsharp Mask
        denoised = cv2.bilateralFilter(illuminated, d=5, sigmaColor=50, sigmaSpace=50)
        gaussian_blur = cv2.GaussianBlur(denoised, (0, 0), 2.0)
        sharpened = cv2.addWeighted(denoised, 1.5, gaussian_blur, -0.5, 0)
        
        # Upscale
        final_frame = cv2.resize(sharpened, original_shape, interpolation=cv2.INTER_LINEAR)
        
        exec_time = (time.perf_counter() - start_time) * 1000 # in ms
        
        return {
            "enhanced_frame": final_frame,
            "is_clear": is_clear,
            "blur_score": getattr(self, 'last_blur_score', 0.0),
            "execution_time_ms": exec_time
        }
