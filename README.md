# Grad-CAM XAI Example with VGG16

This project demonstrates how to apply the Grad-CAM (Gradient-weighted Class Activation Mapping) technique for visual explanation of image classification using the pre-trained VGG16 model.

## Files

- `main.py`: Main script to run Grad-CAM visualization.
- `dog.jpg`: Sample image for inference (you can replace with your own image).

## How to Run

1. Make sure you have Python 3 and necessary libraries installed.
2. Install required packages:

```bash
pip install tensorflow numpy matplotlib opencv-python
```

3. Place an image named `dog.jpg` or `cat.jpg` in the project folder.
4. Run the script:

```bash
python main.py
```

## Output

The script will show the original image and the Grad-CAM visualization side by side.