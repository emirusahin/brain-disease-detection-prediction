import torch
import torchvision.models as models
import torch.nn as nn
from torchvision import transforms
from PIL import Image

# Define the preprocessing transformation
preprocess = transforms.Compose([
    transforms.Grayscale(num_output_channels=1),  # Convert to grayscale
    transforms.Resize(224),                       # Resize the image to 224x224 pixels
    transforms.CenterCrop(224),                   # Crop the center of the image
    transforms.ToTensor(),                        # Convert the image to a PyTorch tensor
    transforms.Normalize(mean=[0.485], std=[0.229]),  # Normalize with the mean and std used in training for grayscale images
])

# Function to load an image, preprocess it, and make a prediction
def predict_image(image_path, model, preprocess):
    # Load the image
    img = Image.open(image_path).convert('L')  # Convert to grayscale ('L' mode in PIL)

    # Preprocess the image
    img_t = preprocess(img)

    # Add a batch dimension (model expects batches, so here we add a batch size of 1)
    batch_t = torch.unsqueeze(img_t, 0)

    # Put the model in evaluation mode and make a prediction
    model.eval()
    with torch.no_grad():
        # Get the prediction
        out = model(batch_t)

        # Convert the prediction to probabilities using softmax
        probabilities = torch.nn.functional.softmax(out, dim=1)

        # Get the max probability and the index of the class
        prob, pred_class = torch.max(probabilities, dim=1)

        # Return the probability and class
        return prob.item(), pred_class.item()

# Load your trained model (assuming it's called resnet18)
# resnet18 = ...

weights = models.ResNet18_Weights.IMAGENET1K_V1
resnet18 = models.resnet18(weights=weights)
num_ftrs = resnet18.fc.in_features # finding how many in_features the last layer has so we can create a new onw with a new out_channel number
resnet18.fc = nn.Linear(num_ftrs, 4)
resnet18.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)
resnet18.load_state_dict(torch.load("model.pth"))

# Predict an image (replace 'path_to_image' with your image's path)
image_path = 'moderate_14.jpg'
probability, class_idx = predict_image(image_path, resnet18, preprocess)

# Print the prediction
print(f'The model predicts class index {class_idx} with probability {probability}')

#%%
