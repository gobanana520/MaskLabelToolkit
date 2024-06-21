from pathlib import Path
import shutil
import json
import math
import numpy as np
import torch
from scipy.spatial.transform import Rotation as R
import cv2
import matplotlib.pyplot as plt
from matplotlib.backends.backend_agg import FigureCanvasAgg
import concurrent.futures
import time
from typing import List, Tuple, Dict, Any
from tqdm import tqdm


def get_logger(log_level="INFO", log_name="default"):
    import logging

    levels = {
        "debug": logging.DEBUG,
        "info": logging.INFO,
        "warning": logging.WARNING,
        "error": logging.ERROR,
        "critical": logging.CRITICAL,
    }
    log_level = levels.get(log_level.lower(), logging.INFO)

    logger = logging.getLogger(log_name)
    if not logger.handlers:
        logger.setLevel(log_level)
        formatter = logging.Formatter(
            "[%(asctime)s] [%(name)-20s] [%(levelname)s] %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
        )
        ch = logging.StreamHandler()
        ch.setLevel(log_level)
        ch.setFormatter(formatter)
        logger.addHandler(ch)
    return logger


def make_clean_folder(folder_path):
    folder = Path(folder_path)
    if folder.exists():
        shutil.rmtree(folder)
    folder.mkdir(parents=True)


def read_data_from_json(json_path):
    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    return data


def write_data_to_json(json_path, data):
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=4, ensure_ascii=False, sort_keys=False)


def apply_transformation(points, trans_mat):
    homo_points = np.stack([points, np.ones((points.shape[0], 1))])
    homo_points = homo_points.dot(trans_mat.T)
    return homo_points[:, :3]


def rvt_to_quat(rvt):
    """Convert rotation vector and translation vector to quaternion and translation vector.

    Args:
        rvt (np.ndarray): Rotation vector and translation vector, shape (6,) or (N, 6). [rvx, rvy, rvz, tx, ty, tz]

    Returns:
        np.ndarray: Quaternion and translation vector, shape (7,) or (N, 7), [qx, qy, qz, qw, tx, ty, tz].

    Raises:
        ValueError: If the input does not have the expected shape or dimensions.
    """
    if not isinstance(rvt, np.ndarray) or rvt.shape[-1] != 6:
        raise ValueError("Input must be a numpy array with last dimension size 6.")

    if rvt.ndim == 2:
        rv = rvt[:, :3]
        t = rvt[:, 3:]
    elif rvt.ndim == 1:
        rv = rvt[:3]
        t = rvt[3:]
    else:
        raise ValueError(
            "Input must be either 1D or 2D with a last dimension size of 6."
        )

    r = R.from_rotvec(rv)
    q = r.as_quat()  # this will be (N, 4) if rv is (N, 3), otherwise (4,)
    if q.ndim == 1:
        return np.concatenate((q, t))  # 1D case
    return np.concatenate((q, t), axis=-1)  # 2D case


def rvt_to_mat(rvt):
    """Convert rotation vector and translation vector to pose matrix.

    Args:
        rvt (np.ndarray): Rotation vector and translation vector, shape (6,) or (N, 6). [rvx, rvy, rvz, tx, ty, tz]
    Returns:
        np.ndarray: Pose matrix, shape (4, 4) or (N, 4, 4).

    Raises:
        ValueError: If the input does not have the expected shape or dimensions.
    """
    if not isinstance(rvt, np.ndarray) or rvt.shape[-1] != 6:
        raise ValueError("Input must be a numpy array with last dimension size 6.")

    if rvt.ndim == 1:
        p = np.eye(4)
        rv = rvt[:3]
        t = rvt[3:]
        r = R.from_rotvec(rv)
        p[:3, :3] = r.as_matrix()
        p[:3, 3] = t
        return p.astype(np.float32)
    elif rvt.ndim == 2:
        p = np.eye(4).reshape((1, 4, 4)).repeat(len(rvt), axis=0)
        rv = rvt[:, :3]
        t = rvt[:, 3:]
        r = R.from_rotvec(rv)
        for i in range(len(rvt)):
            p[i, :3, :3] = r[i].as_matrix()
            p[i, :3, 3] = t[i]
        return p.astype(np.float32)
    else:
        raise ValueError(
            "Input must be either 1D or 2D with a last dimension size of 6."
        )


def mat_to_rvt(mat_4x4):
    """Convert pose matrix to rotation vector and translation vector.

    Args:
        mat_4x4 (np.ndarray): Pose matrix, shape (4, 4) or (N, 4, 4).
    Returns:
        np.ndarray: Rotation vector and translation vector, shape (6,) or (N, 6).

    Raises:
        ValueError: If the input does not have the expected shape or dimensions.
    """
    if not isinstance(mat_4x4, np.ndarray) or mat_4x4.shape[-2:] != (4, 4):
        raise ValueError("Input must be a numpy array with shape (4, 4) or (N, 4, 4).")

    if mat_4x4.ndim == 2:  # Single matrix input
        r = R.from_matrix(mat_4x4[:3, :3])
        rv = r.as_rotvec()
        t = mat_4x4[:3, 3]
        return np.concatenate([rv, t], dtype=np.float32)
    elif mat_4x4.ndim == 3:  # Batch of matrices
        rv = np.empty((len(mat_4x4), 3), dtype=np.float32)
        t = mat_4x4[:, :3, 3]
        for i, mat in enumerate(mat_4x4):
            r = R.from_matrix(mat[:3, :3])
            rv[i] = r.as_rotvec()
        return np.concatenate([rv, t], axis=1, dtype=np.float32)
    else:
        raise ValueError("Input dimension is not valid. Must be 2D or 3D.")


def mat_to_quat(mat_4x4):
    """Convert pose matrix to quaternion and translation vector.

    Args:
        mat_4x4 (np.ndarray): Pose matrix, shape (4, 4) or (N, 4, 4).
    Returns:
        np.ndarray: Quaternion and translation vector, shape (7,) or (N, 7).

    Raises:
        ValueError: If the input does not have the expected shape or dimensions.
    """
    if not isinstance(mat_4x4, np.ndarray) or mat_4x4.shape[-2:] != (4, 4):
        raise ValueError("Input must be a numpy array with shape (4, 4) or (N, 4, 4).")

    if mat_4x4.ndim == 2:  # Single matrix
        r = R.from_matrix(mat_4x4[:3, :3])
        q = r.as_quat()
        t = mat_4x4[:3, 3]
    elif mat_4x4.ndim == 3:  # Batch of matrices
        r = R.from_matrix(mat_4x4[:, :3, :3])
        q = r.as_quat()
        t = mat_4x4[:, :3, 3]
    else:
        raise ValueError("Input dimension is not valid. Must be 2D or 3D.")

    return np.concatenate([q, t], axis=-1).astype(np.float32)


def quat_to_mat(quat):
    """Convert quaternion and translation vector to pose matrix.

    Args:
        quat (np.ndarray): Quaternion and translation vector, shape (7,) or (N, 7).
    Returns:
        np.ndarray: Pose matrix, shape (4, 4) or (N, 4, 4).

    Raises:
        ValueError: If the input does not have the last dimension size of 7.
    """
    if quat.shape[-1] != 7:
        raise ValueError("Input must have the last dimension size of 7.")

    batch_mode = quat.ndim == 2
    q = quat[..., :4]
    t = quat[..., 4:]

    if batch_mode:
        p = np.eye(4).reshape(1, 4, 4).repeat(len(quat), axis=0)
    else:
        p = np.eye(4)

    r = R.from_quat(q)
    p[..., :3, :3] = r.as_matrix()
    p[..., :3, 3] = t

    return p.astype(np.float32)


def quat_to_rvt(quat):
    """Convert quaternion and translation vector to rotation vector and translation vector.

    Args:
        quat (np.ndarray): Quaternion and translation vector, shape (7,) or (N, 7).
    Returns:
        np.ndarray: Rotation vector and translation vector, shape (6,) or (N, 6).

    Raises:
        ValueError: If the input does not have the last dimension size of 7.
    """
    if quat.shape[-1] != 7:
        raise ValueError("Input must have the last dimension size of 7.")

    batch_mode = quat.ndim == 2
    q = quat[..., :4]
    t = quat[..., 4:]

    r = R.from_quat(q)
    rv = r.as_rotvec()

    if batch_mode:
        return np.concatenate(
            [rv, t], axis=-1, dtype=np.float32
        )  # Ensure that the right axis is used for batch processing
    else:
        return np.concatenate([rv, t], dtype=np.float32)  # No axis needed for 1D arrays


def read_rgb_image(image_path, idx=None):
    image = cv2.imread(str(image_path))
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    return image if idx is None else (image, idx)


def write_rgb_image(image_path, image):
    cv2.imwrite(str(image_path), image[:, :, ::-1])


def read_depth_image(image_path, idx=None):
    image = cv2.imread(str(image_path), cv2.IMREAD_ANYDEPTH)
    return image if idx is None else (image, idx)


def write_depth_image(image_path, image):
    cv2.imwrite(str(image_path), image)


def read_mask_image(image_path, idx=None):
    image = cv2.imread(str(image_path), cv2.IMREAD_UNCHANGED)
    return image if idx is None else (image, idx)


def write_mask_image(image_path, image):
    cv2.imwrite(str(image_path), image)


def create_video_from_rgb_images(video_path, images, fps=30):
    height, width, _ = images[0].shape
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    video = cv2.VideoWriter(str(video_path), fourcc, fps, (width, height))
    for image in images:
        video.write(image)
    video.release()


def erode_mask(mask, kernel_size=3, interations=1):
    if kernel_size <= 1:
        return mask
    kernel = np.ones((kernel_size, kernel_size), np.uint8)
    m_dtype = mask.dtype
    mask = mask.astype(np.uint8)
    mask = cv2.erode(mask, kernel, iterations=interations)
    return mask.astype(m_dtype)


def dilate_mask(mask, kernel_size=3, interations=1):
    if kernel_size <= 1:
        return mask
    kernel = np.ones((kernel_size, kernel_size), np.uint8)
    m_dtype = mask.dtype
    mask = mask.astype(np.uint8)
    mask = cv2.dilate(mask, kernel, iterations=interations)
    return mask.astype(m_dtype)


def adjust_xyxy_bbox(bbox, width, height, margin=3):
    """
    Adjust bounding box coordinates with margins and boundary conditions.

    Args:
        bbox (list of int or float): Bounding box coordinates [x_min, y_min, x_max, y_max].
        width (int): Width of the image or mask. Must be greater than 0.
        height (int): Height of the image or mask. Must be greater than 0.
        margin (int): Margin to be added to the bounding box. Must be non-negative.

    Returns:
        np.ndarray: Adjusted bounding box as a numpy array.

    Raises:
        ValueError: If inputs are not within the expected ranges or types.
    """
    if len(bbox) != 4:
        raise ValueError("Bounding box must contain exactly four coordinates.")
    if not all(isinstance(x, (int, float)) for x in bbox):
        raise ValueError("Bounding box coordinates must be integers or floats.")
    if (
        not isinstance(width, int)
        or not isinstance(height, int)
        or not isinstance(margin, int)
    ):
        raise ValueError("Width, height, and margin must be integers.")
    if width <= 0 or height <= 0:
        raise ValueError("Width and height must be positive integers.")
    if margin < 0:
        raise ValueError("Margin must be a non-negative integer.")

    # Convert bbox to integers if necessary
    x_min, y_min, x_max, y_max = map(int, bbox)

    x_min = max(0, x_min - margin)
    y_min = max(0, y_min - margin)
    x_max = min(width - 1, x_max + margin)
    y_max = min(height - 1, y_max + margin)
    return np.array([x_min, y_min, x_max, y_max], dtype=np.int64)


def get_bbox_from_landmarks(landmarks, width, height, margin=3):
    """
    Calculate a bounding box from a set of landmarks, adding an optional margin.

    Args:
        landmarks (np.ndarray): Landmarks array, shape (num_points, 2), with points marked as [-1, -1] being invalid.
        width (int): Width of the image or frame from which landmarks were extracted.
        height (int): Height of the image or frame.
        margin (int): Margin to add around the calculated bounding box. Default is 3.

    Returns:
        np.ndarray: Bounding box coordinates as [x_min, y_min, x_max, y_max].

    Raises:
        ValueError: If landmarks are not a 2D numpy array with two columns, or if width, height, or margin are non-positive.
    """
    if (
        not isinstance(landmarks, np.ndarray)
        or landmarks.ndim != 2
        or landmarks.shape[1] != 2
    ):
        raise ValueError(
            "Landmarks must be a 2D numpy array with shape (num_points, 2)."
        )
    if not all(isinstance(i, int) and i > 0 for i in [width, height, margin]):
        raise ValueError("Width, height, and margin must be positive integers.")

    valid_marks = landmarks[~np.any(landmarks == -1, axis=1)]
    if valid_marks.size == 0:
        raise ValueError(
            "No valid landmarks found; all landmarks are marked as invalid."
        )

    x, y, w, h = cv2.boundingRect(valid_marks)
    bbox = np.array(
        [x - margin, y - margin, x + w + margin, y + h + margin], dtype=np.int64
    )
    bbox[:2] = np.maximum(0, bbox[:2])  # Adjust x_min and y_min
    bbox[2] = min(width - 1, bbox[2])  # Adjust x_max
    bbox[3] = min(height - 1, bbox[3])  # Adjust y_max

    return bbox


def get_bbox_from_mask(mask, margin=3):
    """
    Calculate a bounding box from a binary mask with an optional margin.

    Args:
        mask (np.ndarray): Binary mask, shape (height, width), where non-zero values indicate areas of interest.
        margin (int): Margin to add around the calculated bounding box. Must be non-negative.

    Returns:
        np.ndarray: Adjusted bounding box coordinates as [x_min, y_min, x_max, y_max].

    Raises:
        ValueError: If the mask is not a 2D array or contains no non-zero values, or if the margin is negative.
    """
    if not isinstance(mask, np.ndarray) or mask.ndim != 2:
        raise ValueError("Mask must be a 2D numpy array.")
    if margin < 0:
        raise ValueError("Margin must be non-negative.")
    if not np.any(mask):
        raise ValueError(
            "Mask contains no non-zero values; cannot determine bounding rectangle."
        )

    height, width = mask.shape
    mask_uint8 = mask.astype(
        np.uint8
    )  # Ensure mask is in appropriate format for cv2.boundingRect
    x, y, w, h = cv2.boundingRect(mask_uint8)
    bbox = [x, y, x + w, y + h]
    bbox[0] = max(0, bbox[0] - margin)
    bbox[1] = max(0, bbox[1] - margin)
    bbox[2] = min(width - 1, bbox[2] + margin)
    bbox[3] = min(height - 1, bbox[3] + margin)

    return np.array(bbox, dtype=np.int64)


def xyxy_to_cxcywh(bbox):
    """
    Convert bounding box coordinates from top-left and bottom-right (XYXY) format
    to center-x, center-y, width, and height (CXCYWH) format.

    Args:
        bbox (np.ndarray): Bounding box array in XYXY format. Should be of shape (4,) for a single box
                            or (N, 4) for multiple boxes, where N is the number of boxes.

    Returns:
        np.ndarray: Converted bounding box array in CXCYWH format, with the same shape as the input.

    Raises:
        ValueError: If the input is not 1D or 2D with the last dimension size of 4.
    """
    bbox = np.asarray(bbox)
    if bbox.ndim not in [1, 2] or bbox.shape[-1] != 4:
        raise ValueError(
            "Input array must be 1D or 2D with the last dimension size of 4."
        )

    # Calculate the center coordinates, width, and height
    cx = (bbox[..., 0] + bbox[..., 2]) / 2
    cy = (bbox[..., 1] + bbox[..., 3]) / 2
    w = bbox[..., 2] - bbox[..., 0]
    h = bbox[..., 3] - bbox[..., 1]

    return np.stack([cx, cy, w, h], axis=-1)


def display_images(
    images,
    names=None,
    cmap="gray",
    figsize=(19.2, 10.8),
    dpi=100,
    max_cols=4,
    facecolor="white",
    save_path=None,
    return_array=False,
    idx=None,
):
    num_images = len(images)
    num_cols = min(num_images, max_cols)
    num_rows = (
        num_images + num_cols - 1
    ) // num_cols  # More efficient ceiling division
    if names is None:
        names = [f"fig_{i}" for i in range(num_images)]

    fig, axs = plt.subplots(
        num_rows, num_cols, figsize=figsize, dpi=dpi, facecolor=facecolor
    )

    # Flatten axs for easy iteration and handle cases with single subplot
    if num_images == 1:
        axs = [axs]
    else:
        axs = axs.flat

    try:
        for i, (image, name) in enumerate(zip(images, names)):
            ax = axs[i]
            if image.ndim == 3 and image.shape[2] == 3:  # RGB images
                ax.imshow(image)
            else:  # Depth or grayscale images
                ax.imshow(image, cmap=cmap)
            ax.set_title(name)
            ax.axis("off")

        # Hide any unused axes
        for j in range(i + 1, len(axs)):
            axs[j].axis("off")

        plt.tight_layout()

        img_array = None
        if return_array:
            canvas = FigureCanvasAgg(fig)
            canvas.draw()
            width, height = fig.get_size_inches() * fig.get_dpi()
            img_array = np.frombuffer(canvas.tostring_rgb(), dtype="uint8").reshape(
                int(height), int(width), 3
            )

        if save_path:
            plt.savefig(str(save_path))

        if not save_path and not return_array:
            plt.show()

    finally:
        plt.close(fig)

    if return_array:
        return img_array if idx is None else (img_array, idx)


def draw_mask_overlay(rgb, mask, alpha=0.5, mask_color=(0, 255, 0), reduce_bg=False):
    """Draw a mask overlay on an RGB image.

    Args:
        rgb (np.ndarray): RGB image, shape (height, width, 3).
        mask (np.ndarray): Binary mask, shape (height, width).
        alpha (float): Transparency of the mask overlay.
        mask_color (tuple): RGB color of the mask overlay.
        reduce_bg (bool): If True, reduce the background intensity of the RGB image.

    Returns:
        np.ndarray: RGB image with mask overlay.
    """
    # Create an overlay based on whether to reduce the background
    overlay = np.zeros_like(rgb) if reduce_bg else rgb.copy()

    # Apply the mask color to the overlay where the mask is true
    overlay[mask.astype(bool)] = mask_color

    # Blend the overlay with the original image
    overlay = cv2.addWeighted(overlay, alpha, rgb, 1 - alpha, 0)

    return overlay
