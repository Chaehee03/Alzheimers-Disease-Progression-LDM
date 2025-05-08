"""
lpips3d_medicalnet.py
─────────────────────
MedicalNet‑ResNet50 3D 백본을 이용한 LPIPS‑3D 구현 예시.
학습 전용(NetLinLayer 파라미터만 학습) & 추론용 모듈 모두 포함.
"""

import torch, torch.nn as nn
from torch.hub import load_state_dict_from_url

# ──────────────────────────────────────────────────────────
# 1. MedicalNet ResNet‑50 3D 백본 정의 ─────────────────────
#    (MedicalNet repo 의 generate_model 함수 일부 발췌)
# ──────────────────────────────────────────────────────────
def conv3x3x3(in_planes, out_planes, stride=1):
    return nn.Conv3d(in_planes, out_planes, kernel_size=3,
                     stride=stride, padding=1, bias=False)

class BasicBlock3D(nn.Module):
    expansion = 1
    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super().__init__()
        self.conv1 = conv3x3x3(inplanes, planes, stride)
        self.bn1   = nn.BatchNorm3d(planes)
        self.relu  = nn.ReLU(inplace=True)
        self.conv2 = conv3x3x3(planes, planes)
        self.bn2   = nn.BatchNorm3d(planes)
        self.downsample = downsample
    def forward(self, x):
        identity = x
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        if self.downsample is not None:
            identity = self.downsample(x)
        out = self.relu(out + identity)
        return out

def make_layer(block, inplanes, planes, blocks, stride=1):
    downsample = None
    if stride != 1 or inplanes != planes * block.expansion:
        downsample = nn.Sequential(
            nn.Conv3d(inplanes, planes*block.expansion,
                      kernel_size=1, stride=stride, bias=False),
            nn.BatchNorm3d(planes*block.expansion))
    layers = [block(inplanes, planes, stride, downsample)]
    for _ in range(1, blocks):
        layers.append(block(planes*block.expansion, planes))
    return nn.Sequential(*layers)

class ResNet50_3D(nn.Module):
    """MedicalNet‑ResNet50 (conv1 1‑channel)."""
    def __init__(self, n_input_channels=1):
        super().__init__()
        self.inplanes = 64
        self.conv1 = nn.Conv3d(n_input_channels, 64, kernel_size=7,
                               stride=2, padding=3, bias=False)
        self.bn1   = nn.BatchNorm3d(64)
        self.relu  = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool3d(kernel_size=3, stride=2, padding=1)
        self.layer1 = make_layer(BasicBlock3D, 64,  64,  3)
        self.layer2 = make_layer(BasicBlock3D, 64, 128, 4, stride=2)
        self.layer3 = make_layer(BasicBlock3D, 128,256, 6, stride=2)
        self.layer4 = make_layer(BasicBlock3D, 256,512, 3, stride=2)
    def forward(self, x):
        x = self.maxpool(self.relu(self.bn1(self.conv1(x))))
        f1 = self.layer1(x)  # 64  channels
        f2 = self.layer2(f1) # 128 channels
        f3 = self.layer3(f2) # 256 channels
        f4 = self.layer4(f3) # 512 channels
        return [f1, f2, f3, f4]           # 4 스테이지만 반환
# ──────────────────────────────────────────────────────────

# MedicalNet ResNet‑50 가중치 URL (1‑channel, 23 datasets)
MEDICALNET_URL = ("https://huggingface.co/TencentMedicalNet/"
                  "MedicalNet-Resnet50/resolve/main/resnet_50_23dataset.pth")

def load_medicalnet_backbone():
    """MedicalNet ResNet‑50 3D 백본 반환 (freeze)."""
    model = ResNet50_3D(n_input_channels=1)
    state = load_state_dict_from_url(MEDICALNET_URL, progress=True)
    model.load_state_dict(state, strict=False)  # conv1 weight 일치(1채널)
    for p in model.parameters(): p.requires_grad = False
    model.eval()
    return model

# ──────────────────────────────────────────────────────────
# 2. LPIPS‑3D 모듈 (NetLinLayer + spatial avg) ─────────────
# ──────────────────────────────────────────────────────────
def spatial_avg_3d(t, keepdim=True):
    return t.mean([2, 3, 4], keepdim=keepdim)

class NetLinLayer3D(nn.Module):
    def __init__(self, chn_in, use_dropout=True):
        super().__init__()
        layers = [nn.Dropout()] if use_dropout else []
        layers += [nn.Conv3d(chn_in, 1, kernel_size=1, bias=False)]
        self.model = nn.Sequential(*layers)
    def forward(self, x): return self.model(x)

class LPIPS3D(nn.Module):
    """
    LPIPS‑3D perceptual distance for 1‑channel MRI volumes.
    - backbone: MedicalNet ResNet50 3D (freeze)
    - lins:    학습 대상 (random init → fine‑tune)
    """
    def __init__(self, use_dropout=True):
        super().__init__()
        self.backbone = load_medicalnet_backbone()
        # 채널 수는 backbone 각 stage out_channels
        chns = [64, 128, 256, 512]
        self.lins = nn.ModuleList(
            [NetLinLayer3D(c, use_dropout) for c in chns])
    @torch.no_grad()
    def extract_feats(self, x):
        return self.backbone(x)
    def forward(self, x, y, normalize=True):
        if normalize:
            x = (x - x.mean()) / (x.std() + 1e-5)
            y = (y - y.mean()) / (y.std() + 1e-5)
        fx, fy = self.extract_feats(x), self.extract_feats(y)
        diffs  = [(fx[i] - fy[i]) ** 2 for i in range(len(fx))]
        res    = [spatial_avg_3d(self.lins[i](d)) for i, d in enumerate(diffs)]
        val    = torch.zeros_like(res[0])
        for r in res: val += r
        return val   # (B,1,1,1,1)

# ──────────────────────────────────────────────────────────
# 3. Linear‑head 학습 예시 (distillation/ ranking / MSE) ───
# ──────────────────────────────────────────────────────────
if __name__ == "__main__":
    import torch.nn.functional as F
    # toy example: 두 random volume 의 MSE 를 target distance 로 사용
    model = LPIPS3D().cuda()
    opt   = torch.optim.Adam(model.lins.parameters(), lr=1e-4)

    for step in range(100):
        vol1 = torch.randn(2, 1, 96, 96, 96, device="cuda")
        vol2 = vol1 + 0.05 * torch.randn_like(vol1)  # 약간의 노이즈
        with torch.no_grad():
            target = F.mse_loss(vol1, vol2, reduction="none").mean([1,2,3,4])
        pred = model(vol1, vol2).squeeze()   # (B,)
        loss = F.mse_loss(pred, target)
        loss.backward(); opt.step(); opt.zero_grad()
        if step % 20 == 0:
            print(f"step {step:3d} | loss {loss.item():.4f}")
