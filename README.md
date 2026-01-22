# String Art Generator

A Python project that generates **string art on a circular board** by simulating threads drawn between nails to approximate a grayscale image.
I made this readme, and the comments in the programs using GPT. But the code is writeen by myself:)

The program places nails evenly around a circle, precomputes all possible strings between nail pairs, and greedily selects the strings that best reconstruct the target image using a fast ΔMSE update rule.

---

## Features

- Circular board with configurable size and number of nails  
- Anti-aliased line drawing using Xiaolin Wu’s algorithm and a faster method using Bresenham's line drawing algorithm
- Greedy optimization with closed-form MSE updates (fast sparse pursuit)  
- Optional inversion of the target image  
- Intermediate visualization during optimization  
- Export final string-art image as JPEG (with nails visible)  
- Optional export of the selected string sequence  

---

## Installation

Clone the repository and install dependencies:

```bash
git clone https://github.com/yourusername/String_Art_Generator.git
cd String_Art_Generator
pip install -r requirements.txt
```

### Requirements

* numpy
* matplotlib
* pillow
* scikit-image
* tqdm

---

## Usage

Run the main script:

```bash
python main.py
```

At the bottom of `main.py`, modify the parameters:

```python
main(
    image_path="your_image.jpg",
    thread_thickness=1,
    canvas_size=610,
    num_nails=200,
    darkness=1,
    num_threads=1000,
    line_algo="wu"
)
```

---

## Parameters

* `image_path` – Path to the input image
* `thread_thickness` – Thickness of a single string (controls resolution)
* `canvas_size` – Diameter of the circular board
* `num_nails` – Number of nails placed on the circumference
* `darkness` – Darkness contribution of each string
* `num_threads` – Maximum number of greedy iterations
* `line_algo` – Line drawing algorithm:

  * `"wu"` (recommended, anti-aliased)
  * `"bresenham"` (faster, aliased)

---

## Program Flow

1. Load and preprocess the input image (grayscale, square crop, resize)
2. Apply a circular mask to define the board region
3. Place `N` nails evenly around the circumference
4. Precompute sparse imprints for all possible strings
5. Initialize a white canvas
6. Iteratively:

   * Estimate ΔMSE for every candidate string
   * Select the string that best reduces reconstruction error
   * Subtract its contribution from the canvas
7. Display and save the final string-art image

This is a greedy sparse approximation (matching pursuit) over the space of all nail-to-nail strings.

---

## Output

* Final reconstructed string-art image (JPEG or PNG)
* Optional CSV file containing the selected nail pairs (thread sequence)

The saved image includes the reconstructed strings and visible nail positions.

---

## Tips

* Use `"wu"` for smoother and more accurate results
* Higher `num_nails` and `num_threads` improve quality but increase runtime
* High-contrast grayscale portraits work best
* Large boards may take several minutes to converge

---

## Future Additions

* Graphical user interface (GUI)
* Export threading instructions in a machine-readable format
* Automatic generation of the string threading order
* Automatic stopping based on convergence

---

## License

MIT License
