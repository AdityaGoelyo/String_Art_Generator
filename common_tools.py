from PIL import Image, ImageDraw
import numpy as np
import matplotlib.pyplot as plt
import io
from skimage.draw import line_aa
from tqdm import tqdm
import csv
import json

def get_image(path_or_bytes, size: int, scale: float = 1.0):
    """
    Load and preprocess an input image into a square grayscale numpy array.

    This function performs the following steps:
    1. Opens the image either from a file path or from raw byte data.
    2. Converts the image to grayscale (PIL mode 'L').
    3. Crops the image to the largest possible centered square.
    4. Resizes the square image to (size x size) using Lanczos filtering.
    5. Converts the image to a float32 numpy array and normalizes pixel values
       into the range [0, scale], where:
           0.0   = black
           scale = white   (normally scale = 1.0)

    Parameters:
        path_or_bytes : str or bytes
            Path to the image file or raw image bytes.
        size : int
            Output resolution in pixels (size x size).
        scale : float
            Maximum brightness value after normalization.

    Returns:
        arr : np.ndarray (float32, shape = [size, size])
            Preprocessed grayscale image.
    """

    # Open image from disk or from memory
    if isinstance(path_or_bytes, bytes):
        img = Image.open(io.BytesIO(path_or_bytes))
    else:
        img = Image.open(path_or_bytes)

    # Convert to grayscale (0 = black, 255 = white)
    img = img.convert('L')

    # Crop the image to a centered square
    w, h = img.size
    min_edge = min(w, h)
    left = (w - min_edge) // 2
    top = (h - min_edge) // 2
    img = img.crop((left, top, left + min_edge, top + min_edge))

    # Resize to fixed resolution using high-quality Lanczos filter
    img = img.resize((size, size), Image.LANCZOS)

    # Convert to float array and normalize to [0, scale]
    arr = np.array(img).astype(np.float32)
    arr = arr / 255.0 * scale

    return arr

def get_circular_mask_and_masked_image(image: np.ndarray, scale = 1) -> tuple[np.ndarray, np.ndarray]:
    """
    Generate a circular mask for a square image and apply it.

    Pixels inside the circle are kept unchanged.
    Pixels outside the circle are set to `scale` (white), effectively removing
    them from the optimization region.

    The circle is centered in the image and has radius = image_size // 2.

    Parameters:
        image : np.ndarray
            Square grayscale image (2D).
        scale : float
            Value assigned to pixels outside the circular region.

    Returns:
        mask : np.ndarray (bool)
            True for pixels inside the circular region.
        masked_image : np.ndarray
            Image with outside-circle pixels set to white.
    """

    # Image must be square for a centered circular mask
    h, w = image.shape
    assert h == w, "Image must be square"

    # Radius and center of the circle
    radius = h // 2
    center = (radius, radius)

    # Create coordinate grid
    y, x = np.ogrid[:h, :w]

    # Squared distance of each pixel from the center
    dist_from_center = (x - center[0])**2 + (y - center[1])**2

    # Mask selects pixels inside the circle
    mask = dist_from_center <= radius**2

    # Apply mask: outside pixels set to white
    masked_image = np.copy(image)
    masked_image[~mask] = scale

    return mask, masked_image

def find_coordinates_of_all_nails(num_nails: int, canvas_size: int) -> dict[int, np.ndarray]:
    """
    Compute evenly spaced nail coordinates placed on the circumference of a circle.

    Nails are placed uniformly in angle around a circle that fits inside the
    square canvas. Each nail is represented by integer pixel coordinates.

    Parameters:
        num_nails : int
            Total number of nails around the circle.
        canvas_size : int
            Width / height of the square canvas in pixels.

    Returns:
        nails : dict[int, np.ndarray]
            Mapping from nail index -> (x, y) pixel coordinate.
    """

    # Radius chosen so the circle fits exactly inside the canvas
    radius = (canvas_size - 1) / 2

    nails = {}
    angle = 2 * np.pi / num_nails   # angular spacing between nails

    for i in range(num_nails):
        # Convert polar (angle, radius) to Cartesian (x, y)
        x = int(round(radius * np.cos(i * angle) + radius))
        y = int(round(radius * np.sin(i * angle) + radius))

        nails[i] = np.array([x, y])

    return nails

def bresenham_imprints(image: np.ndarray, x0: int, y0: int, x1: int, y1: int, darkness: float) -> np.ndarray:
    """
    Draw a discrete straight line between two points using Bresenham's algorithm.

    This implementation:
    - Handles all line orientations by transposing steep lines.
    - Uses integer arithmetic only (fast, no floating point).
    - Overwrites pixel values along the line with the given darkness.

    Parameters:
        image : np.ndarray
            2D grayscale canvas to draw on.
        x0, y0 : int
            Start point coordinates.
        x1, y1 : int
            End point coordinates.
        darkness : float
            Pixel value assigned to line pixels.

    Returns:
        img : np.ndarray
            Copy of the input image with the line drawn.
    """

    img = image.copy()

    # Check if the line is steep (slope magnitude > 1)
    steep = abs(y1 - y0) > abs(x1 - x0)

    # Transpose coordinates if steep so we always step in x
    if steep:
        x0, y0 = y0, x0
        x1, y1 = y1, x1

    # Ensure left-to-right drawing
    if x0 > x1:
        x0, x1 = x1, x0
        y0, y1 = y1, y0

    dx = x1 - x0
    dy = abs(y1 - y0)

    # Error accumulator (half of dx initially)
    error = dx // 2

    # Direction to step in y
    ystep = 1 if y0 < y1 else -1

    y = y0
    for x in range(x0, x1 + 1):

        # Plot either (x, y) or (y, x) depending on transposition
        if steep:
            if 0 <= x < img.shape[0] and 0 <= y < img.shape[1]:
                img[x, y] = darkness
        else:
            if 0 <= y < img.shape[0] and 0 <= x < img.shape[1]:
                img[y, x] = darkness

        # Update error and step in y when threshold crossed
        error -= dy
        if error < 0:
            y += ystep
            error += dx

    return img

def xiaolin_wu_imprints(canvas, x0, y0, x1, y1, darkness):
    """
    Applies an anti-aliased line imprint onto the canvas using Xiaolin Wu's method
    via skimage.draw.line_aa.

    Args:
        canvas (ndarray): 2D numpy array of float32 or float64 values in range [0, 1]
        x0, y0, x1, y1 (int): coordinates of the line
        darkness (float): how much to subtract from pixel brightness (0 = no change, 1 = full black)

    Returns:
        ndarray: new canvas with the line imprinted
    """
    img = canvas.copy()
    rr, cc, val = line_aa(y0, x0, y1, x1)  # Note: skimage takes (row, col)
    # Subtract brightness with respect to darkness and clip
    img[rr, cc] = img[rr, cc] - val*darkness
    img = np.ones(img.shape) - img
    
    #ones = np.ones(img.shape)
    #img = ones - img

    return img

def generate_all_imprints_sparse(mask, nails, darkness, algo):
    """
    Precompute sparse line imprints for every pair of nails.

    For each unordered nail pair (i, j):
    - Draw a line using the selected algorithm.
    - Keep only pixels inside the circular mask.
    - Store the affected pixel indices and their corresponding brightness values.

    This creates a sparse dictionary representation so that each greedy iteration
    only touches the pixels affected by a candidate string.

    Parameters:
        mask : np.ndarray (bool)
            Circular mask defining the valid drawing region.
        nails : dict[int, np.ndarray]
            Nail index -> (x, y) coordinates.
        darkness : float
            Darkness contribution of each line.
        algo : str
            Line algorithm ("bresenham" or "wu").

    Returns:
        imprints : dict
            (i, j) -> (indices, values)
            where indices are mask-relative flattened indices and values are
            the pixel contributions of that line.
    """

    imprints = {}
    side = mask.shape[0]
    n = len(nails)

    # Precompute flat indices of all valid (inside-circle) pixels
    mask_flat_indices = np.where(mask.flatten())[0]

    total_pairs = n * (n - 1) // 2
    progress = tqdm(total=total_pairs, desc="Generating imprints", unit="pair")

    for i in range(n):
        for j in range(i + 1, n):

            # Create fresh blank canvas for this line
            if algo == "bresenham":
                canvas = np.zeros((side, side), dtype=np.float32)
                canvas = bresenham_imprints(canvas, *nails[i], *nails[j], darkness)

            elif algo == "wu":
                canvas = np.ones((side, side), dtype=np.float32)
                canvas = xiaolin_wu_imprints(canvas, *nails[i], *nails[j], darkness)

            else:
                raise ValueError("Invalid algo. Choose 'bresenham' or 'wu'.")

            # Detect which pixels were modified by this line
            diff = canvas < 1.0
            masked_diff = diff & mask

            if np.any(masked_diff):
                # Extract only masked pixels in flattened order
                flat_line = canvas[mask]

                # Select pixels that were actually affected
                affected_mask = flat_line > 0
                indices = np.where(affected_mask)[0]
                values = flat_line[indices]

                imprints[(i, j)] = (indices, values)

            progress.update(1)

    progress.close()
    return imprints

def plot_image(img, nails, num_nails, ax=None, title=None):
    created = False
    if ax is None:
        fig, ax = plt.subplots(figsize=(6, 6))
        created = True

    ax.imshow(img, cmap='gray', vmin=0, vmax=1)

    xs = [nails[i][0] for i in range(num_nails)]
    ys = [nails[i][1] for i in range(num_nails)]
    ax.scatter(xs, ys, color='black', s=5)

    if title:
        ax.set_title(title)

    ax.axis('off')

    if created:
        plt.show()

def mse(true, pred):
    return np.mean((true - pred) ** 2)

def save_imprints_to_csv(imprints, filename):
    """
    Save imprints dictionary to CSV.

    Parameters:
        imprints (dict): {(i, j): (indices_array, values_array)}
        filename (str): Path to CSV file
    """
    with open(filename, mode="w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["nail_i", "nail_j", "indices", "values"])  # Header

        for (i, j), (indices, values) in imprints.items():
            # Serialize arrays as JSON strings so they can be read back exactly
            indices_str = json.dumps(indices.tolist())
            values_str = json.dumps(values.tolist())
            writer.writerow([i, j, indices_str, values_str])

def save_string_art_with_nails(image, nails, num_nails, filename, scale=1.0, nail_radius=2, quality=95):
    """
    Save reconstructed string-art image as a JPEG with nails drawn on top.

    Parameters:
        image : np.ndarray
            2D array with pixel values in [0, scale].
        nails : dict
            Nail index -> (x, y) pixel coordinates.
        num_nails : int
            Number of nails.
        filename : str
            Output file path (.jpg or .jpeg).
        scale : float
            Maximum brightness scale used in the image (usually 1.0).
        nail_radius : int
            Radius (in pixels) of each nail dot.
        quality : int
            JPEG quality (1–100).
    """

    # Normalize to [0, 1] using known scale
    img = image.astype(np.float32)
    img = np.clip(img / scale, 0, 1)

    # Convert to 8-bit grayscale
    img_uint8 = (img * 255).astype(np.uint8)

    # Create PIL image
    pil_img = Image.fromarray(img_uint8, mode="L")
    draw = ImageDraw.Draw(pil_img)

    # Draw nails as black dots
    for i in range(num_nails):
        x, y = nails[i]
        r = nail_radius
        draw.ellipse((x - r, y - r, x + r, y + r), fill=0)  # black nails

    # Save JPEG
    pil_img.save(filename, format="JPEG", quality=quality)
