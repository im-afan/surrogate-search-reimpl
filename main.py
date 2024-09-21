from __future__ import print_function
import numpy as np
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import os
import time
from torchvision import datasets, transforms
from model import *
from model_vgg import *
from tensorboardX import SummaryWriter

torch.autograd.set_detect_anomaly(True)
best_acc=0
best_epoch=0
model_save_path = 'checkpoint/' + 'resnet19'

def set_surrogate(model: nn.Module, k: torch.Tensor):
    for name, child_module in model.named_children():
        if(isinstance(child_module, LIFSpike_loss_kt)):
            child_module.t = k
        set_surrogate(child_module, k)

def sample_surrogate(logits: torch.Tensor):
    dist = torch.distributions.Categorical(logits=logits)
    k = dist.sample()
    return k, dist.log_prob(k)

def test(args, model, device, test_loader, epoch, writer):
    model.eval()
    test_loss = 0
    correct = 0
    isEval = False
    global best_acc
    global best_epoch
    global model_save_name
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            data, _ = torch.broadcast_tensors(data, torch.zeros((steps,) + data.shape))
            data = data.permute(1, 2, 3, 4, 0)
            output, _, _ = model(data)
            test_loss += F.cross_entropy(output, target, reduction='sum').item() # sum up batch loss
            pred = output.argmax(dim=1, keepdim=True)  # get the index of the max log-probability
            correct += pred.eq(target.view_as(pred)).sum().item()
        if best_acc < 100. * correct / len(test_loader.dataset):
            best_acc = 100. * correct / len(test_loader.dataset)
            best_epoch = epoch
            torch.save(model.state_dict(), model_save_path + '_{:.4f}.pth'.format(best_acc))
            print('Save Best Model {}'.format(best_acc))

        print('best_acc is: {}'.format(best_acc))
        print('Iters: {}'.format(best_epoch))
        writer.add_scalar('Test Acc /epoch', 100. * correct / len(test_loader.dataset), epoch)   
    test_loss /= len(test_loader.dataset)
    writer.add_scalar('Test Loss /epoch', test_loss, epoch)
    writer.add_scalar('Test Acc /epoch', 100. * correct / len(test_loader.dataset), epoch)
    for i, (name, param) in enumerate(model.named_parameters()):
        if '_s' in name:
            writer.add_histogram(name, param, epoch)
    print('Test set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
        test_loss, correct, len(test_loader.dataset),
        100. * correct / len(test_loader.dataset)))


def train(args, model, device, train_loader, test_loader, epoch, writer, optimizer, scheduler, loss_fn, dist_optimizer, dist_params):
    for epoch_num in range(epoch):
        cnt = 0
        running_loss, running_loss_dist, running_k = 0, 0, 0
        prev_loss, prev_k, prev_log_prob, prev_entropy = None, None, None, None
        dist_loss = torch.zeros(1)
        for batch_idx, (data, target) in enumerate(train_loader):
            data, target = data.to(device), target.to(device)
            target = F.one_hot(target, num_classes=10).to(torch.float32)
            data, _ = torch.broadcast_tensors(data, torch.zeros((steps,) + data.shape))
            data = data.permute(1, 2, 3, 4, 0)

            optimizer.zero_grad()
            dist_optimizer.zero_grad()
            output, mem_out, k_logits = model(data)

            dist = torch.distributions.Categorical(logits=k_logits)
            k = dist.sample().detach().to(torch.float32)
            set_surrogate(model, k)

            #print(loss_fn(output, target), dist_loss)
            model_loss = loss_fn(output, target)
            train_loss = model_loss# + dist_loss 
            train_loss.backward()
            optimizer.step()

            running_loss += model_loss.detach().item() 
            running_loss_dist += dist_loss.detach().item()
            running_k += (k.sum() / k.shape[0]).item()
            cnt += 1

            if(prev_loss is not None):
                loss_chg = model_loss.detach() - prev_loss.detach()
                #print("log prob and entropy: ", torch.mean(prev_log_prob).item(), torch.mean(prev_entropy).item())
                #print(torch.exp(k_logits))
                #print(k_logits)
                dist_loss = loss_chg * torch.mean(prev_log_prob) - args.k_entropy * torch.mean(prev_entropy) # maximum entropy -> minimum -entropy
                dist_loss.backward()
                #nn.utils.clip_grad_norm_(dist_params, 0.1)
                dist_optimizer.step()

            prev_loss = model_loss 
            prev_k = k.detach()
            prev_log_prob = dist.log_prob(k)
            prev_entropy = dist.entropy()

            if(batch_idx % args.log_interval == 0):
                running_loss /= cnt
                running_loss_dist /= cnt
                running_k /= cnt
                writer.add_scalar("train/loss", running_loss)
                writer.add_scalar("train/dist_loss", running_loss_dist)
                writer.add_scalar("train/k", running_k)
                print("loss:", running_loss)
                print("dist_loss:", running_loss_dist)
                print("k:", running_k)
                running_loss = 0
                running_loss_dist = 0
                running_k = 0
                cnt = 0

        test(args, model, device, test_loader, epoch, writer) 
        scheduler.step()

def main():
    # Training settings
    parser = argparse.ArgumentParser(description='PyTorch MNIST Example')
    parser.add_argument('--arch', type=str, default='vgg16', choices=['resnet19', 'vgg11', 'vgg13', 'vgg16', 'vgg19'])
    parser.add_argument('--batch-size', type=int, default=64, metavar='N',
                        help='input batch size for training (default: 64)')
    parser.add_argument('--test-batch-size', type=int, default=300, metavar='N',
                        help='input batch size for testing (default: 1000)')
    parser.add_argument('--epochs', type=int, default=1000, metavar='N',
                        help='number of epochs to train (default: 10)')
    parser.add_argument('--lr', type=float, default=1e-2, metavar='LR',
                        help='learning rate (default: 0.01)')
    parser.add_argument('--momentum', type=float, default=0.9, metavar='M',
                        help='SGD momentum (default: 0.5)')
    parser.add_argument('--no-cuda', action='store_true', default=False,
                        help='disables CUDA training')
    parser.add_argument('--seed', type=int, default=666, metavar='S',
                        help='random seed (default: 1)')
    parser.add_argument('--log-interval', type=int, default=50, metavar='N',
                        help='how many batches to wait before logging training status')
    parser.add_argument('--k-entropy', default=1, type=float,
                        help='weight of entropy')
    parser.add_argument('--save-model', action='store_true', default=True,
                        help='For Saving the current Model')
    args = parser.parse_args()

    use_cuda = not args.no_cuda and torch.cuda.is_available()
    torch.manual_seed(args.seed)

    device = torch.device("cuda:0" if use_cuda else "cpu")
    kwargs = {'num_workers': 0, 'pin_memory': True} if use_cuda else {}
    writer = SummaryWriter('./summaries/Cifarnet')
    
    train_loader = torch.utils.data.DataLoader(
        datasets.CIFAR10('./data', train=True, download=True,
                         transform=transforms.Compose([
                             transforms.RandomHorizontalFlip(),
                             transforms.RandomCrop(32, padding=4),
                             transforms.ToTensor(),
                             transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
                         ])),
        batch_size=args.batch_size, shuffle=True, **kwargs)
    test_loader = torch.utils.data.DataLoader(
        datasets.CIFAR10('./data', train=False, 
                        transform=transforms.Compose([
                            transforms.ToTensor(),
                            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
                        ])),
        batch_size=args.test_batch_size, shuffle=True, **kwargs)

    if(args.arch == "resnet19"):
        model = resnet19()
    elif(args.arch == "vgg11"):
        model = vgg11_bn()
    else:
        model = vgg16_bn()
    model.to(device)
    print('success')

    model_params = []
    dist_params = []
    for name, param in model.named_parameters():
        if(name.startswith("surrogate_pred")):
            print("surrogate", name)
            dist_params.append(param)
        else:
            print("model", name)
            model_params.append(param)

    optimizer = optim.SGD(model_params, lr=args.lr, momentum=args.momentum, weight_decay=1e-4)
    dist_optimizer = optim.Adam(dist_params, lr=args.lr)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, eta_min=0, T_max=args.epochs)
    dist_scheduler = optim.lr_scheduler.CosineAnnealingLR(dist_optimizer, eta_min=0, T_max=args.epochs)
    loss_fn = nn.CrossEntropyLoss()
    train(args, model, device, train_loader, test_loader, args.epochs, writer, optimizer, scheduler, loss_fn, dist_optimizer, dist_params)
    writer.close()

    
if __name__ == '__main__':
    main()
