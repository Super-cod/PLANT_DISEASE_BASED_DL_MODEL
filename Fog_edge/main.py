from sys import modules

from matplotlib import transforms
from torchvision import models
import torch
import torch.nn as nn
import os
class PlantDiseaseModel(nn.Module):
    def __init__(self,num_classes):
        super().__init__()
        self.resnet=models.resnet34(pretrained=True)

        for param in self.resnet.parameters():
            param.requires_grad=False

        num_ftrs=self.resnet.fc.in_features
        self.resnet.fc=nn.Linear(num_ftrs,num_classes)
    def forward(self,x):
        return self.resnet(x)

num_classes=38
device=torch.device("cuda" if torch.cuda.is_available() else "cpu")
model=PlantDiseaseModel(num_classes).to(device)
model.load_state_dict(torch.load(r"C:\Users\swaya\Desktop\TIMEPASS\More_Timepass\PLANT_DISEASE_BASED_DL_MODEL\model_updated.pth"))

model.eval()
from torchvision import datasets, transforms
test_transforms = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])
Class={}

folder_name=[name for name in os.listdir(r"C:\Users\swaya\Desktop\TIMEPASS\More_Timepass\PLANT_DISEASE_BASED_DL_MODEL\New Plant Diseases Dataset(Augmented)\train")]

for i in range(len(folder_name)):
    Class[i]=folder_name[i]
print(Class)
from PIL import Image
import torch
img_path = r"C:\Users\swaya\Desktop\TIMEPASS\More_Timepass\PLANT_DISEASE_BASED_DL_MODEL\test.jpg"
image = Image.open(img_path)
image = test_transforms(image).unsqueeze(0).to(device)  # Add batch dimension

output = model(image)

_, predicted = torch.max(output.data, 1)
predicted.item()
print("Predicted class:", Class[predicted.item()])
print("Class labels:", list(Class.values()))

Class[predicted.item()]