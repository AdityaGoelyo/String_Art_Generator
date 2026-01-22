import common_tools as ct
import numpy as np
import matplotlib.pyplot as plt
import time


def greedy_search(flat_img, pred, imprints, nails):
    """
    Perform one greedy optimization step.

    For every possible string (nail pair), estimate how much the global MSE
    would change if this string were drawn next. Select the line that gives
    the best improvement and update the current prediction accordingly.

    Parameters:
        flat_img : 1D numpy array
            Target grayscale image restricted to circular mask (flattened).
        pred : 1D numpy array
            Current reconstruction (flattened, same shape as flat_img).
        imprints : dict
            Precomputed sparse line imprints:
            (i, j) -> (indices, values) where this line affects the canvas.
        nails : dict
            Nail index -> (x, y) coordinates (only used for indexing length).

    Returns:
        best_delta : float
            Estimated change in MSE for the selected line.
        best_line_coords : tuple
            (start_nail, end_nail) chosen for this iteration.
        updated_pred : 1D numpy array
            Updated reconstruction after subtracting this line.
    """

    best_line_coords = None
    best_delta = 0
    n = len(nails)                 # number of nails (not directly used here)
    m = len(flat_img)             # total number of pixels inside the mask

    # Try every possible precomputed line
    for (i, j), (indices, values) in imprints.items():

        # True pixel values at positions affected by this line
        masked_true = flat_img[indices]

        # Current predicted pixel values at those same positions
        masked_pred = pred[indices]

        # Closed-form ΔMSE estimate for subtracting this line
        delta = np.dot(values + 2*masked_pred - 2*masked_true, values) / m

        # Keep the line that gives the largest improvement
        if delta > best_delta:
            best_delta = delta
            best_line_coords = (i, j)

            # Apply the candidate line to a copy of the current prediction
            updated_pred = pred.copy()
            updated_pred[indices] = np.clip(masked_pred - values, 0, 1)

    return best_delta, best_line_coords, updated_pred


def main(
        image_path,
        thread_thickness,
        canvas_size,
        num_nails,
        darkness,
        num_threads,
        line_algo):
    
    """
    Full pipeline for greedy string-art reconstruction.

    Steps:
    1. Load and preprocess the input image.
    2. Apply circular mask and place nails on the boundary.
    3. Precompute sparse imprints for all possible strings.
    4. Optionally invert the target image.
    5. Run greedy optimization for a fixed number of threads.
    6. Display intermediate and final reconstructions.

    Parameters:
        image_path : str
            Path to the input image.
        thread_thickness : float
            Physical thickness of one string (controls resolution).
        canvas_size : float
            Diameter of the circular canvas (in same units as thickness).
        num_nails : int
            Number of nails placed around the circle.
        darkness : float
            Darkness contribution of one string.
        num_threads : int
            Maximum number of greedy iterations.
        line_algo : str
            Line drawing algorithm ("wu" or "bresenham").
    """

    # Compute pixel resolution from physical canvas size and string thickness
    num_pixels = int(canvas_size / thread_thickness)

    # Load and preprocess the image, create circular mask, and place nails
    img = ct.get_image(image_path, num_pixels, 1)
    mask, img = ct.get_circular_mask_and_masked_image(img)
    nails = ct.find_coordinates_of_all_nails(num_nails, num_pixels)

    # Flatten only the circular region (optimization is done only there)
    flat_img = img[mask]

    # Start from a completely white canvas
    pred = np.ones_like(flat_img)

    # Precompute sparse imprints for all possible nail pairs
    sparse_imprints = ct.generate_all_imprints_sparse(mask, nails, darkness, line_algo)

    # Optional: save imprints to disk (commented out by default)
    # ct.save_imprints_to_csv(sparse_imprints, "/imprints.csv")

    print("successfully loaded image, imprint, nails and blank canvas")

    # Show original and inverted target images side-by-side for user choice
    fig, axes = plt.subplots(1, 2, figsize=(12, 6))
    ct.plot_image(img, nails, num_nails, ax=axes[0], title="Original")
    ct.plot_image(np.ones(img.shape) - img, nails, num_nails, ax=axes[1], title="Inverted")
    plt.tight_layout()
    plt.show()

    # Ask user whether to optimize for inverted image
    use_inverse = input("use_inverse_img?(y or n): ")
    if use_inverse == "y":
        img = np.ones(img.shape) - img

    # Main greedy optimization loop
    start_time = time.time()
    threaded_lines = {}   # stores which line was chosen at each iteration
    delta_mse = {}        # stores ΔMSE values for each iteration

    for i in range(num_threads):

        # Perform one greedy step
        best_delta_i, best_line_coords_i, pred = greedy_search(
            flat_img, pred, sparse_imprints, nails
        )

        print(f"iteration {i}, delta_mse {best_delta_i}, line drawn {best_line_coords_i}")

        threaded_lines[i] = best_line_coords_i
        delta_mse[i] = best_delta_i

        # Every 200 iterations, visualize current reconstruction and ask whether to continue
        if i % 200 == 0:
            adjusted_pred = np.ones((num_pixels, num_pixels))
            adjusted_pred[mask] = pred

            ct.plot_image(adjusted_pred, nails, num_nails)

            _ = input("continue?(y or n): ")
            if _ == "n":
                break

    # Reconstruct final 2D image from flattened prediction
    adjusted_pred = np.ones((num_pixels, num_pixels))
    adjusted_pred[mask] = pred

    stop_time = time.time()
    runtime = stop_time - start_time

    print(f"ran for {runtime} seconds")

    # Show final reconstruction
    ct.plot_image(adjusted_pred, nails, num_nails)

    ct.save_string_art_with_nails(
    adjusted_pred,
    nails,
    num_nails,
    "final_string_art.jpg",
    scale=darkness,
    nail_radius=thread_thickness*2
    )

main(
    image_path="/home/aesha/projects/string art generator/papa.jpeg",
    thread_thickness=1,
    canvas_size=610,
    num_nails=200,
    darkness=1,
    num_threads=1000,
    line_algo="wu", # use 'wu' or 'bresenham', wu is recommended
)
