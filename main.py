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

class Main:
    def __init__(self):
        # Load your trained model (assuming it's called resnet18)
        weights = models.ResNet18_Weights.IMAGENET1K_V1
        self.resnet18 = models.resnet18(weights=weights)
        num_ftrs = self.resnet18.fc.in_features  # finding how many in_features the last layer has
        self.resnet18.fc = nn.Linear(num_ftrs, 4)
        self.resnet18.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.resnet18.load_state_dict(torch.load("model.pth"))
        self.resnet18.eval()

    def predict_image(self, image_path):
        try:
            # Load the image
            img = Image.open(image_path).convert('L')  # Convert to grayscale ('L' mode in PIL)

            # Preprocess the image
            img_t = preprocess(img)

            # Add a batch dimension (model expects batches, so here we add a batch size of 1)
            batch_t = torch.unsqueeze(img_t, 0)

            # Put the model in evaluation mode and make a prediction
            with torch.no_grad():
                # Get the prediction
                out = self.resnet18(batch_t)

                # Convert the prediction to probabilities using softmax
                probabilities = torch.nn.functional.softmax(out, dim=1)

                # Get the max probability and the index of the class
                prob, pred_class = torch.max(probabilities, dim=1)

                

                # Return the probability and class
                return prob.item(), pred_class.item()

        except Exception as e:
            return str(e)

# Create an instance of the Main class
main = Main()
