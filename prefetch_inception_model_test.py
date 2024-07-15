import matplotlib.pyplot
from tensorflow.keras.models import load_model
import numpy as np
import matplotlib.pyplot as plt
import json
from tensorflow.keras.preprocessing import image
#loading the model
model = load_model('/home/aisha/inceptionv4.h5')
#Loading and preprocessing the images

def load_and_preprocess_image(img_path, target_size):
    img = image.load_img(img_path, target_size=target_size)
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array /= 255.0
    return img_array


def main():
    img_path = '/home/aisha/deeplearning/Imagenet/test_directory/4.jpeg'
    target_size = (299, 299)
    img_array = load_and_preprocess_image(img_path, target_size)
    # Load the class_indices from the JSON file
    with open('class_indices.json', 'r') as file:
        class_indices = json.load(file)

    # Reverse the class_indices dictionary
    indices_class = {v: k for k, v in class_indices.items()}
    #making prediction
    predictions = model.predict(img_array)
    predicted_class_index = np.argmax(predictions, axis=-1)[0]

    # Get the class name
    predicted_class_name = indices_class[predicted_class_index]
    print(f"Predicted Class: {predicted_class_name}")
    confidence = predictions[0][predicted_class_index] *100
    print("Confidence",confidence)
    #plotiing
    img = image.load_img(img_path, target_size=target_size)
    plt.figure(figsize=(8, 6))
    plt.imshow(img)
    plt.title(f"Predicted: {predicted_class_name}\nConfidence: {confidence:.2f}%")
    plt.axis('off')  # Hide the axes
    plt.show()

if __name__ == '__main__':
        main()

