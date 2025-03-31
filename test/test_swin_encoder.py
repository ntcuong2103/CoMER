from comer.model.swin_encoder import SwinEncoder
import torch
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = SwinEncoder().to(device)
img = torch.randn(2, 1, 512, 256).to(device)
img_mask = torch.zeros(2, 512, 256).long().to(device)
feats, mask = model(img, img_mask)
pass
