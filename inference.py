import cv2
import numpy as np
import torch
import matplotlib.pyplot as plt
from tkinter import Tk, filedialog, messagebox
from model import CNNModel


def load_trained_model(model, model_path):
    if torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")

    model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device)
    model.eval()
    print("Model loaded from", model_path)
    return model

# Define the inference function
def infer_image(model, target_size=(50, 50)):

    # Open file dialog to select image
    Tk().withdraw()  # Hide the root window
    image_path = filedialog.askopenfilename(title="Select an Image", filetypes=[("Image files", "*.png;*.jpg;*.jpeg")])

    if not image_path:
        print("No image selected.")
        return None, None

    try:
        img = cv2.imread(image_path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = cv2.resize(img, target_size)
        img_array = img / 255.0  # Normalize pixel values

        img_batch = np.expand_dims(img_array, axis=0)
        img_batch = np.transpose(img_batch, (0, 3, 1, 2))
        img_tensor = torch.tensor(img_batch, dtype=torch.float32).to(next(model.parameters()).device)
        img_tensor = img_tensor.contiguous()

    except Exception as e:
        messagebox.showerror("Error", f"Error processing {image_path}: {str(e)}")
        return None, None

    with torch.no_grad():
        output = model(img_tensor)
        cancer_probability = torch.sigmoid(output[0][0]).item()

    predicted_class = "Cancer" if cancer_probability >= 0.5 else "Normal"

    plt.imshow(img)
    plt.title(f'Predicted Class: {predicted_class}\nProbability of Cancer: {cancer_probability:.4f}')
    plt.axis('off')
    plt.show()

    messagebox.showinfo("Inference Result", f"Predicted Class: {predicted_class}\nProbability of Cancer: {cancer_probability:.4%}")

    return predicted_class, cancer_probability

# Example usage
if __name__ == "__main__":
    model_path = "cnn_model.pth"
    model = CNNModel()
    model = load_trained_model(model, model_path)

    predicted_class, cancer_probability = infer_image(model)
    if predicted_class is not None:
        print(f'Predicted Class: {predicted_class}, Probability of Cancer: {cancer_probability:.4f}')
