import models
from efficientnet import EfficientNet
MODELDISPATCHER  = {
     "resnet34" : models.resnet34,
     "efficientnet": EfficientNet.custom_head("efficientnet-b4", n_classes = 5, pretrained = True)
}
