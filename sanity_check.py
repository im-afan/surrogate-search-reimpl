import sys
import torch
from torch import nn
from model_vgg import *
from layers import *

class TestModel(nn.Module):
    def __init__(self, spk=False):
        super().__init__()
        self.spk = spk
        if(not spk):
            self.seq = nn.Sequential(
                tdLayer(nn.Linear(3*32*32, 10))
            )
        else:
            self.layer1 = nn.Sequential(
                tdLayer(nn.Linear(3*32*32, 512)),
                LIFSpike()
            )
            self.seq = nn.Sequential(
                tdLayer(nn.Linear(512, 10))
            )

    def forward(self, x):
        x = x.view(x.shape[0], -1, x.shape[4])
        if(not self.spk):
            x = self.seq(x)
        else:
            x = self.layer1(x)
            print(x.shape)
            if(x.shape[0] > 1):
                print("x.mean():", x[0].sum(), x[1].sum())
            else:
                print("x.mean():", x[0].sum())
            x = self.seq(x)

        out = torch.sum(x, dim=2) / steps
        return out, None, None

separate = int(sys.argv[1])
torch.manual_seed(0)

img1 = torch.zeros((1, 3, 32, 32))
img2 = torch.ones((1, 3, 32, 32))
batched_img = torch.cat((img1, img2))
imgs = []

label1 = torch.zeros((1, 10))
label2 = torch.zeros((1, 10))
label1[0][0] = 1
label2[0][1] = 1
batched_label = torch.cat((label1, label2)) 
labels = []

if(separate == 1):
    print("separate mode")
    imgs = [img1, img2]
    labels = [label1, label2]
else:
    print("batched mode")
    imgs = [batched_img]
    labels = [batched_label]

model = vgg11_bn()
for param in model.features.parameters():
    param.requires_grad = False
#model = TestModel(spk=True)

loss_fn = nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(model.parameters(), lr=0.01, momentum=0.9, weight_decay=1e-4)

epochs = 100
for epoch in range(epochs):
    running_loss = 0
    for i in range(len(imgs)):
        img, label = imgs[i], labels[i]
        print(img[0][0][0][0])
        img, _ = torch.broadcast_tensors(img, torch.zeros((steps,) + img.shape))
        img = img.permute(1, 2, 3, 4, 0)

        optimizer.zero_grad()
        output, mem_out, k_logits = model(img)
        print(output, label)
        loss = loss_fn(output, label)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
    
    print("running loss:", running_loss)