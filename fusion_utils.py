import numpy as np, pandas as pd, torch, torch.nn as nn, joblib
from torchvision import models as tvm, transforms as T
from PIL import Image
from matplotlib import cm

CLASSES = ["akiec", "bcc", "bkl", "df", "mel", "nv", "vasc"]
FULL_NAMES = {
    "akiec": "Actinic keratoses / intraepithelial carcinoma",
    "bcc": "Basal cell carcinoma",
    "bkl": "Benign keratosis-like lesion",
    "df": "Dermatofibroma",
    "mel": "Melanoma",
    "nv": "Melanocytic nevus",
    "vasc": "Vascular lesion",
}
SEXES = ["male", "female", "unknown"]
LOCALIZATIONS = ["abdomen", "acral", "back", "chest", "ear", "face", "foot", "genital",
                 "hand", "lower extremity", "neck", "scalp", "trunk", "unknown", "upper extremity"]

EVAL_TF = T.Compose([T.Resize((224, 224)), T.ToTensor(),
                     T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])


class FusionNet(nn.Module):
    def __init__(self, d_clin=19, d_img=512, emb=32, n_cls=7):
        super().__init__()
        self.clin = nn.Sequential(nn.Linear(d_clin, 64), nn.ReLU(), nn.Dropout(0.2),
                                  nn.Linear(64, emb), nn.ReLU())
        self.img_drop = nn.Dropout(0.3)
        self.head = nn.Sequential(nn.Linear(d_img + emb, 128), nn.ReLU(),
                                  nn.Dropout(0.3), nn.Linear(128, n_cls))

    def forward(self, f_img, x_clin):
        z = torch.cat([self.img_drop(f_img), self.clin(x_clin)], 1)
        return self.head(z)


class GradCAM:
    """Forward hook se activations aur unke gradients pakadta hai."""
    def __init__(self, layer):
        self.acts, self.grads = None, None
        layer.register_forward_hook(self._fwd)

    def _fwd(self, module, inp, out):
        self.acts = out
        out.register_hook(lambda g: setattr(self, "grads", g))


def load_models(model_dir, device="cpu"):
    resnet = tvm.resnet18(weights=None)
    resnet.fc = nn.Linear(512, 7)
    resnet.load_state_dict(torch.load(f"{model_dir}/resnet18_best.pth", map_location=device))
    resnet.to(device).eval()
    backbone = nn.Sequential(*list(resnet.children())[:-1])   # 512-d features
    fusion = FusionNet()
    fusion.load_state_dict(torch.load(f"{model_dir}/fusion_best.pth", map_location=device))
    fusion.to(device).eval()
    pre = joblib.load(f"{model_dir}/clinical_preprocessor.joblib")
    cam = GradCAM(resnet.layer4)
    return resnet, backbone, fusion, pre, cam


def clinical_vector(pre, age, sex, loc):
    """age=None matlab unknown (training jaise median se impute hoga)."""
    row = pd.DataFrame([{"age": np.nan if age is None else float(age),
                         "sex": sex, "localization": loc}])
    return pre.transform(row[["age", "sex", "localization"]]).astype("float32")


def image_only_probs(resnet, img, device="cpu"):
    x = EVAL_TF(img.convert("RGB")).unsqueeze(0).to(device)
    with torch.no_grad():
        return torch.softmax(resnet(x), 1)[0].cpu().numpy()


def fusion_predict(img, clin_x, backbone, fusion, cam, device="cpu", target=None):
    """Return: probs (7,), predicted class index, heatmap (7x7, 0-1)."""
    x = EVAL_TF(img.convert("RGB")).unsqueeze(0).to(device)
    c = torch.tensor(clin_x, dtype=torch.float32).to(device)
    with torch.enable_grad():
        feats = backbone(x).flatten(1)
        logits = fusion(feats, c)
        probs = torch.softmax(logits, 1)[0]
        cls = int(probs.argmax()) if target is None else target
        backbone.zero_grad(); fusion.zero_grad()
        logits[0, cls].backward()
    w = cam.grads.mean(dim=(2, 3), keepdim=True)
    heat = torch.relu((w * cam.acts).sum(1))[0].detach().cpu().numpy()
    heat = heat / (heat.max() + 1e-8)
    return probs.detach().cpu().numpy(), cls, heat


def overlay(img, heat, alpha=0.4, size=224):
    base = np.array(img.convert("RGB").resize((size, size))) / 255.0
    h = np.array(Image.fromarray(np.uint8(255 * heat)).resize((size, size), Image.BILINEAR)) / 255.0
    colored = cm.jet(h)[..., :3]
    return np.uint8(255 * ((1 - alpha) * base + alpha * colored))
