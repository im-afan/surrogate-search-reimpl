import sys
import torch
from torch import nn
from model_vgg import *

torch.set_printoptions(sci_mode=False)

separate = int(sys.argv[1])
torch.manual_seed(0)

img1 = torch.ones((1, 3, 32, 32), dtype=torch.float32) * 2
img2 = torch.ones((1, 3, 32, 32), dtype=torch.float32) * 1
img3 = torch.ones((1, 3, 32, 32), dtype=torch.float32) * -1
img4 = torch.ones((1, 3, 32, 32), dtype=torch.float32) * -2
imgs = []

label1 = torch.zeros((1, 10))
label2 = torch.zeros((1, 10))
label3 = torch.zeros((1, 10))
label4 = torch.zeros((1, 10))
label1[0][0] = 1
label2[0][1] = 1
label3[0][2] = 1
label4[0][3] = 1
labels = []

if(separate):
    print("separate mode")
    batched_img1 = torch.cat((img1, img2))
    batched_img2 = torch.cat((img3, img4))
    batched_label1 = torch.cat((label1, label2)) 
    batched_label2 = torch.cat((label3, label4)) 
    imgs = [batched_img1, batched_img2]
    labels = [batched_label1, batched_label2]
else:
    print("batched mode")
    batched_img = torch.cat((img1, img2, img3, img4))
    batched_label = torch.cat((label1, label2, label3, label4)) 
    imgs = [batched_img]
    labels = [batched_label]

model = vgg11_bn()

loss_fn = nn.CrossEntropyLoss()
#optimizer = torch.optim.SGD(model.parameters(), lr=0.01, momentum=0.9, weight_decay=1e-4)
optimizer = torch.optim.SGD(model.parameters(), lr=0.001, momentum=0.5, weight_decay=1e-4)
#optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

epochs = 100
for epoch in range(epochs):
    running_loss = 0
    for i in range(len(imgs)):
        img, label = imgs[i], labels[i]
        img, _ = torch.broadcast_tensors(img, torch.zeros((steps,) + img.shape))
        img = img.permute(1, 2, 3, 4, 0)

        #print(img.shape, label.shape)

        optimizer.zero_grad()
        output, mem_out, k_logits = model(img)
        loss = loss_fn(output, label)
        loss.backward()
        optimizer.step()

        print(F.softmax(output))

        running_loss += loss.item()
    
    print("running loss:", running_loss)
    
