# String Art Generator

This project generates **string art on a circular board** by simulating threads drawn between evenly spaced nails to approximate a grayscale image.

The algorithm places nails on the circumference of a circle, precomputes all possible strings between nail pairs, and then greedily selects strings that best reduce the reconstruction error between a blank canvas and the target image.

The output is a grayscale string-art image and (optionally) the sequence of nail connections used.

---

## Features

- Circular board with configurable size and number of nails  
- Anti-aliased line drawing using Xiaolin Wu’s algorithm and faster line drawing option using Bresenham's algorithm
- Greedy optimization using a fast closed-form ΔMSE update  
- Optional inversion of target image  
- Intermediate visualization during optimization  
- Export final string-art image as JPEG (with nails visible)  

---

## Installation

Clone the repository and install the required packages:

```bash
clone the repo
cd String_Art_Generator
pip install -r requirements.txt
