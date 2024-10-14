import argparse
import os
import torch.optim as optim
from models import *
import dataloader
import torch
import torch.nn as nn
import random
import numpy as np
from torch.utils.tensorboard import SummaryWriter

torch.autograd.set_detect_anomaly(True)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

parser = argparse.ArgumentParser(description='TEBN')
parser.add_argument('-w', '--workers', default=10, type=int, metavar='N', help='number of workers')
parser.add_argument('--epochs', default=200, type=int, metavar='N', help='number of training epochs')
parser.add_argument('--start_epoch', default=0, type=int, metavar='N', help='start epoch number for resume models')
parser.add_argument('-b', '--batch_size', default=64, type=int, metavar='N', help='number of batch size')
parser.add_argument('--seed', default=1000, type=int, help='seed')
parser.add_argument('-T', '--time', default=4, type=int, metavar='N', help='inference time-step')
parser.add_argument('-out_dir', default='./logs/', type=str, help='log dir')
parser.add_argument('-resume', default='./TEBN_VGG9.pth', type=str, help='resume from checkpoint')
parser.add_argument('-method', default='TEBN', type=str, help='BN method')
parser.add_argument('-tau', type=float, default=0.25, help='tau value of LIF neuron')
parser.add_argument('-k_dist', type=float, default=1, help='weight of distloss in loss calculation')
parser.add_argument('-static-surrogate', action='store_true', help='toggle using static surrogate instead of dynamic')
parser.add_argument('-loss_chg_discount', default=0.9)
parser.add_argument('-update-dist-freq', default=1000)

args = parser.parse_args()

def train(model, device, train_loader, criterion, optimizer, dist_optimizer, epoch, args):
    running_loss = 0
    running_loss_chg = 0
    model.train()
    total = 0
    correct = 0
    dist_loss = torch.zeros(1).to(device)
    loss_chgs, log_probs = [], []
    for i, (images, labels) in enumerate(train_loader):
        optimizer.zero_grad()
        dist_optimizer.zero_grad()

        labels = labels.to(device)
        images = images.to(device)

        outputs, log_prob = model(images)
        mean_out = outputs.mean(1)
        model_loss = criterion(mean_out, labels).mean()
        running_loss += model_loss.item()
        model_loss.backward()
        optimizer.step()

        if(i <= args.update_dist_freq):
            with torch.no_grad():
                new_outputs, _ = model(images)
                new_outputs = new_outputs.mean(1)
                new_loss = criterion(new_outputs, labels).mean().detach()
            loss_chgs.append(new_loss - model_loss.item())
            log_probs.append(log_prob)
            #print(f"old loss {model_loss.item()} new loss {new_loss.item()}")
            if(not args.static_surrogate and len(log_probs) == 1): 
                dist_loss = torch.zeros(1, device=device)
                reward = 0
                for i in range(len(log_probs)-1, -1, -1):
                    reward += loss_chgs[i] + reward * args.loss_chg_discount
                    dist_loss += args.k_dist * reward * log_probs[i]
                dist_loss.backward()
                dist_optimizer.step()
                del loss_chgs[:]
                del log_probs[:]

        #print(model_loss.item(), dist_loss.item())
        #print("step done")

        total += float(labels.size(0))
        _, predicted = mean_out.cpu().max(1)
        correct += float(predicted.eq(labels.cpu()).sum().item())
    return running_loss, 100 * correct / total


@torch.no_grad()
def test(model, test_loader, device):
    correct = 0
    total = 0
    model.eval()
    for batch_idx, (inputs, targets) in enumerate(test_loader):
        inputs = inputs.to(device)
        outputs, _ = model(inputs)
        mean_out = outputs.mean(1)
        _, predicted = mean_out.cpu().max(1)
        total += float(targets.size(0))
        correct += float(predicted.eq(targets).sum().item())
        if batch_idx % 100 == 0:
            acc = 100. * float(correct) / float(total)
            print(batch_idx, len(test_loader), ' Acc: %.5f' % acc)
    final_acc = 100 * correct / total
    return final_acc


if __name__ == '__main__':
    # set manual seed
    random.seed(args.seed)
    os.environ['PYTHONHASHSEED'] = str(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

    train_dataset, val_dataset = dataloader.Cifar10(download=True)

    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True,
                                               num_workers=args.workers, pin_memory=True)
    test_loader = torch.utils.data.DataLoader(val_dataset, batch_size=args.batch_size,
                                              shuffle=False, num_workers=args.workers, pin_memory=True)

    model = VGG9(tau=args.tau, static=(args.static_surrogate))

    model = torch.nn.DataParallel(model)
    model.to(device)

    model_params, dist_params = [], []
    for name, param in model.named_parameters():
        if("surrogate" not in name):
            model_params.append(param)
        else:
            dist_params.append(param)


    criterion = nn.CrossEntropyLoss().to(device)
    optimizer = optim.SGD(model_params, lr=0.02, weight_decay=5e-4, momentum=0.9)
    #dist_optimizer = optim.SGD(dist_params, lr=0.1, momentum=0.2)
    dist_optimizer = optim.Adam(dist_params, lr=0.01)

    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, eta_min=0, T_max=args.epochs)

    start_epoch = 0

    out_dir = os.path.join(args.out_dir, f'method_{args.method}_tau_{args.tau}_T_{args.time}')

    if args.resume:
        try:
            checkpoint = torch.load(args.resume, map_location='cuda')
            model.load_state_dict(checkpoint['model'])
            optimizer.load_state_dict(checkpoint['optimizer'])
            scheduler.load_state_dict(checkpoint['scheduler'])
            start_epoch = checkpoint['epoch'] + 1
            best_acc = checkpoint['best_acc']
        except:
            pass

    best_acc = 0
    best_epoch = 0

    writer = SummaryWriter(os.path.join(out_dir, 'logs'), purge_step=start_epoch)

    for epoch in range(start_epoch, args.epochs):

        loss, acc = train(model, device, train_loader, criterion, optimizer, dist_optimizer, epoch, args)
        print('Epoch {}/{} train loss={:.5f} train acc={:.3f}'.format(epoch, args.epochs, loss, acc))
        writer.add_scalar('train_loss', loss, epoch)
        writer.add_scalar('train_acc', acc, epoch)
        test_acc = test(model, test_loader, device)
        print('Epoch {}/{} test acc={:.3f}'.format(epoch, args.epochs, test_acc))
        writer.add_scalar('test_acc', test_acc, epoch)
        scheduler.step()

        save_max = False
        if best_acc < test_acc:
            best_acc = test_acc
            save_max = True
            best_epoch = epoch + 1
        print('Best Test acc={:.3f}'.format(best_acc))

        checkpoint = {
            'model': model.state_dict(),
            'optimizer': optimizer.state_dict(),
            'scheduler': scheduler.state_dict(),
            'epoch': epoch,
            'best_acc': best_acc
        }

        if save_max:
            torch.save(checkpoint, os.path.join(out_dir, 'checkpoint_max.pth'))