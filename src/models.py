import torch
import torch.nn as nn
import torchvision.models as models
import numpy as np
import cv2

class XRFEncoder(nn.Module):
    def __init__(self, xrf_dim=23, hid=128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(xrf_dim, hid),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hid, hid),
            nn.ReLU()
        )
    def forward(self, x):
        return self.net(x)

class XRFPredictor(nn.Module):
    def __init__(self, hid=128, out_dim=3):
        super().__init__()
        self.head = nn.Sequential(
            nn.Linear(hid, 128),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(128, out_dim),
        )
    def forward(self, xrf_feat):
        return self.head(xrf_feat)

class ImageEncoder(nn.Module):
    def __init__(self, out_dim=8, pretrained=True):
        super().__init__()
        try:
            if pretrained:
                self.backbone = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
            else:
                self.backbone = models.resnet18(weights=None)
        except:
            self.backbone = models.resnet18(weights=None)

        in_dim = self.backbone.fc.in_features
        self.backbone.fc = nn.Identity()
        self.proj = nn.Sequential(
            nn.Linear(in_dim, out_dim),
            nn.BatchNorm1d(out_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
        )
    def forward(self, img):
        feat = self.backbone(img)
        return self.proj(feat)

class ImageResidualHead(nn.Module):
    def __init__(self, in_dim=8, out_dim=3):
        super().__init__()
        self.head = nn.Sequential(
            nn.Linear(in_dim, 128),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(128, out_dim),
        )
        for m in self.head.modules():
            if isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, mean=0.0, std=1e-3)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
    def forward(self, img_feat):
        return self.head(img_feat)

class GateNet(nn.Module):
    def __init__(self, xrf_hid=128, img_hid=8):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(xrf_hid + img_hid, 128),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(128, 3)
        )
        nn.init.zeros_(self.net[-1].weight)
        nn.init.constant_(self.net[-1].bias, -4.0)
    def forward(self, xrf_feat, img_feat):
        g = self.net(torch.cat([xrf_feat, img_feat], dim=1))
        return torch.sigmoid(g)

class GatedResidualFusion(nn.Module):
    def __init__(self, xrf_dim=23):
        super().__init__()
        self.xrf_enc  = XRFEncoder(xrf_dim=xrf_dim, hid=128)
        self.xrf_pred = XRFPredictor(hid=128, out_dim=3)
        self.img_enc  = ImageEncoder(out_dim=8, pretrained=True)
        self.img_delta= ImageResidualHead(in_dim=8, out_dim=3)
        self.gate     = GateNet(xrf_hid=128, img_hid=8)

    def forward(self, image, xrf):
        xrf_feat = self.xrf_enc(xrf)
        pred_xrf = self.xrf_pred(xrf_feat)
        img_feat = self.img_enc(image)
        delta    = self.img_delta(img_feat)
        gate     = self.gate(xrf_feat, img_feat)
        pred     = pred_xrf + gate * delta
        pred     = torch.clamp(pred, 0.0, 1.0)
        return pred, pred_xrf, gate, img_feat

class GradCAM:
    def __init__(self, model, target_layer):
        self.model = model
        self.target_layer = target_layer
        self.gradients = None
        self.activations = None
        self.hook_handles = []
        self.hook_handles.append(self.target_layer.register_forward_hook(self.save_activation))
        self.hook_handles.append(self.target_layer.register_full_backward_hook(self.save_gradient))

    def save_activation(self, module, input, output):
        self.activations = output
    def save_gradient(self, module, grad_input, grad_output):
        self.gradients = grad_output[0]

    def generate_cam(self, combined_features, class_probabilities, class_index, input_shape, retain_graph=False):
        self.model.zero_grad()
        target = (combined_features * torch.randn_like(combined_features)).sum() * class_probabilities[class_index]
        target.backward(retain_graph=retain_graph)

        gradients = self.gradients.cpu().data.numpy()[0]
        activations = self.activations.cpu().data.numpy()[0]
        weights = np.mean(gradients, axis=(1, 2))
        cam = np.zeros(activations.shape[1:], dtype=np.float32)

        for i, w in enumerate(weights):
            cam += w * activations[i]

        cam = np.maximum(cam, 0)
        cam = cv2.resize(cam, (input_shape[3], input_shape[2]))
        cam = cam - np.min(cam)
        cam = cam / (np.max(cam) + 1e-8)
        return cam

    def remove_hooks(self):
        for h in self.hook_handles:
            h.remove()
