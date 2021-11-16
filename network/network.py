import torch.nn as nn
from torchvision import models
import torch

def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv2d') != -1:
        torch.nn.init.xavier_uniform_(m.weight)
        torch.nn.init.zeros_(m.bias)
    elif classname.find('BatchNorm') != -1:
        m.weight.data.normal_(1.0, 0.01)
        m.bias.data.fill_(0)
    elif classname.find('Linear') != -1:
        m.weight.data.normal_(0.0, 0.01)
        m.bias.data.normal_(0.0, 0.01)
        
        
#-------------------------------------------------------------------------------------------------        
class ResNet18(nn.Module):
    def __init__(self, use_projection=False):
        super(ResNet18, self).__init__()
        
        model_resnet = models.resnet18(pretrained=True)
        self.use_projection = use_projection

        self.conv1 = model_resnet.conv1
        self.bn1 = model_resnet.bn1
        self.relu = model_resnet.relu
        self.maxpool = model_resnet.maxpool
        self.layer1 = model_resnet.layer1
        self.layer2 = model_resnet.layer2
        self.layer3 = model_resnet.layer3
        self.layer4 = model_resnet.layer4
        self.avgpool = model_resnet.avgpool

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)

        x = self.layer4(x)
        x_p = x
        x = self.avgpool(x)
        x = x.view(x.size(0), -1)

        return x, x_p


class RClassifier(nn.Module):
    def __init__(self, input_dim=2048, class_num=47, extract=True, dropout_p=0.5):
        super(RClassifier, self).__init__()
        self.fc1 = nn.Sequential(
            nn.Linear(input_dim, 1000),
            nn.BatchNorm1d(1000, affine=True),
            nn.ReLU(inplace=True),
            nn.Dropout(p=dropout_p)
        )
        self.fc2 = nn.Linear(1000, class_num)
        self.extract = extract
        self.dropout_p = dropout_p

    def forward(self, x):
        emb = self.fc1(x)
        logit = self.fc2(emb)

        if self.extract:
            return emb, logit
        return logit


class SSClassifier(nn.Module):
    def __init__(self, input_dim, projection_dim=100, class_num=24):
        super(SSClassifier, self).__init__()
        self.input_dim = input_dim
        self.projection_dim = projection_dim

        self.conv_1x1 = nn.Sequential(
            nn.Conv2d(self.input_dim, self.projection_dim, [1,1], stride=[1,1]),
            nn.BatchNorm2d(self.projection_dim),
            nn.ReLU(inplace=True)
            )
        self.conv_3x3 = nn.Sequential(
            nn.Conv2d(self.projection_dim, self.projection_dim, [3,3], stride=[2,2]),
            nn.BatchNorm2d(self.projection_dim),
            nn.ReLU(inplace=True)
            )
        self.fc1 = nn.Sequential(
            nn.Linear(self.projection_dim*3*3, self.projection_dim),
            nn.BatchNorm1d(self.projection_dim, affine=True),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.5)
            )
        self.fc2 = nn.Linear(projection_dim, class_num)

    def forward(self, x):
        x = self.conv_1x1(x)
        x = self.conv_3x3(x)
        x = x.view(x.size(0), -1)
        x = self.fc1(x)
        x = self.fc2(x)
        return x
    
#-------------------------------------------------------------------------------------------------

class set_Evaluation:
    def __init__(self, nets):
        self.nets = nets

    def __enter__(self):
        self.prev = torch.is_grad_enabled()
#         torch._C.set_grad_enabled(False)
        for net in self.nets:
            net.eval()

    def __exit__(self, *args):
        torch.set_grad_enabled(self.prev)
        for net in self.nets:
            net.train()
        return False

    def __call__(self, func):
        @functools.wraps(func)
        def decorate_no_grad(*args, **kwargs):
            with self:
                return func(*args, **kwargs)

        return decorate_no_grad