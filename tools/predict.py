import torch

from PIL import Image
from torchvision import transforms
import matplotlib.pyplot as plt
import json

import time

cur_time = time.time()
from torchvision.models import MobileNetV2

print(torch.__version__)
data_transform = transforms.Compose(
    [transforms.Resize(256),
     transforms.CenterCrop(224),
     transforms.ToTensor(),
     transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])

# load image
img = Image.open("./test/1.jpg")
plt.imshow(img)
# [N, C, H, W]
img = data_transform(img)
# expand batch dimension
img = torch.unsqueeze(img, dim=0)
print("img:", img)


try:
    json_file = open('./class_indices.json', 'r')
    class_indict = json.load(json_file)
except Exception as e:
    print(e)
    exit(-1)

# create model
model = MobileNetV2(num_classes=26)
# load model weights
model_weight_path = "./weight/mobilenet_v2_transport_0.1.pth"
model.load_state_dict(torch.load(model_weight_path))
# torch.load(model_weight_path)
model.eval()
with torch.no_grad():
    # predict class
    output = torch.squeeze(model(img))
    predict = torch.softmax(output, dim=0)
    predict_cla = torch.argmax(predict).numpy()
print(predict_cla)
predict_cla = int(predict_cla)
print(type(predict_cla))
plt.show()