from comer.model.comer_swin import CoMER
import torch

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = CoMER(
    d_model=512,
    nhead=8,
    num_decoder_layers=6,
    dim_feedforward=2048,
    dropout=0.1,
    dc=4,
    cross_coverage=True,
    self_coverage=True,
).to(device)
img = torch.randn(2, 1, 256, 256).to(device)
img_mask = torch.zeros(2, 256, 256).long().to(device)
tgt = torch.zeros(4, 256).long().to(device)
out = model(img, img_mask, tgt)
print(out.shape)  # Expecting [2, 256, vocab_size]