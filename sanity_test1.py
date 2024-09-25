import sys
import torch
from torch import nn
from layers import *

class Model(nn.Module):
    def __init__(self):
        super().__init__();
        self.seq = nn.Sequential(
            tdLayer(nn.Linear(2, 4), nn.BatchNorm1d(4)),
            LIFSpike(),
            tdLayer(nn.Linear(4, 2))
        )

    def forward(self, x):
        x = self.seq(x)
        out = torch.sum(x, dim=2) / steps
        return out, None, None
    
separate = int(sys.argv[1])
torch.manual_seed(0)

img1 = torch.tensor([[1, 0]], dtype=torch.float32)
img2 = torch.tensor([[0, 1]], dtype=torch.float32)
batched_img = torch.cat((img1, img2))
imgs = []

label1 = torch.tensor([[0, 1]], dtype=torch.float32)
label2 = torch.tensor([[1, 0]], dtype=torch.float32)
batched_label = torch.cat((label1, label2)) 
labels = []

print(batched_img.shape, batched_label.shape)

if(separate == 1):
    print("separate mode")
    #imgs = [img1, img2]
    #labels = [label1, label2]
    imgs = [batched_img, batched_img]
    labels = [batched_label, batched_label]
else:
    print("batched mode")
    imgs = [batched_img]
    labels = [batched_label]

model = Model()

loss_fn = nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(model.parameters(), lr=0.1, momentum=0.9, weight_decay=1e-4)

epochs = 100
for epoch in range(epochs):
    running_loss = 0
    for i in range(len(imgs)):
        img, label = imgs[i], labels[i]
        img, _ = torch.broadcast_tensors(img, torch.zeros((steps,) + img.shape))
        img = img.permute(1, 2, 0)
        print(img.shape)

        optimizer.zero_grad()
        output, mem_out, k_logits = model(img)
        loss = loss_fn(output, label)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
    
    print("running loss:", running_loss)
    
