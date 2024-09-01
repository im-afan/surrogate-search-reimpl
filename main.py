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
from tensorboardX import SummaryWriter

best_acc=0
best_epoch=0
model_save_path = 'checkpoint/' + 'resnet19'

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
            output, dloss = model(data)
            test_loss += F.cross_entropy(output, target, reduction='sum').item() + dloss.mean() * args.distrloss # sum up batch loss
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


def train(args, model, device, train_loader, test_loader, epoch, writer, optimizer, scheduler, loss_fn):
    for epoch_num in range(epoch):
        #set_kt(model, torch.tensor([5]).float(), torch.tensor([10]))
        running_loss = 0
        for batch_idx, (data, target) in enumerate(train_loader):
            data, target = data.to(device), target.to(device)
            target = F.one_hot(target, num_classes=10).to(torch.float32)
            data, _ = torch.broadcast_tensors(data, torch.zeros((steps,) + data.shape))
            data = data.permute(1, 2, 3, 4, 0)
            output, dloss = model(data)

            #print(output, target)
            train_loss = loss_fn(output, target) + dloss.mean() * args.distrloss # sum up batch loss
            train_loss.backward()
            optimizer.step()

            running_loss += train_loss.detach().item() 

            if(batch_idx % args.log_interval == 0):
                writer.add_scalar("train/loss", running_loss)
                print("loss:", running_loss)
                running_loss = 0

        test(args, model, device, test_loader, epoch, writer) 
        scheduler.step()

def main():
    # Training settings
    parser = argparse.ArgumentParser(description='PyTorch MNIST Example')
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
    parser.add_argument('--distrloss', default=2.0, type=float,
                        help='weight of distrloss')
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

    model = resnet19()
    #checkpoint = torch.load('./best.pth', map_location='cpu')
    #model.load_state_dict({k.replace('module.', ''):v for k, v in checkpoint.items()}, strict = False)
    model.to(device)
    print('success')
    optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=args.momentum, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, eta_min=0, T_max=args.epochs)
    loss_fn = nn.CrossEntropyLoss()
    #test(args, model, device, test_loader, 0, writer)
    train(args, model, device, train_loader, test_loader, args.epochs, writer, optimizer, scheduler, loss_fn)
    writer.close()

    
if __name__ == '__main__':
    main()
