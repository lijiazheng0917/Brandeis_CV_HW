import numpy as np
import argparse
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import StepLR

from models.sphereface import SphereFace4
from models.loss import AngularSoftmaxWithLoss
from dataset import get_loader

def train_accuracy(output, targets, topk=(1,)):
    """Computes the accuracy over the k top predictions for the specified values of k"""
    maxk = max(topk)
    with torch.no_grad():
        batch_size = targets.size(0)
        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(targets.view(1, -1).expand_as(pred))
        res = []
        for k in topk:
            correct_k = correct[:k].view(-1).float().sum(0, keepdim=True)
            res.append(correct_k / batch_size)
        return res

def eval_acc(threshold, diff,target):
    y_true = []
    y_predict = []
    for d,t in zip(diff,target):
        same = 1 if d > threshold else 0
        y_predict.append(same)
        y_true.append(t.cpu().numpy())
    y_true = np.array(y_true)
    y_predict = np.array(y_predict)
    accuracy = 1.0*np.count_nonzero(y_true==y_predict)/len(y_true)
    return accuracy

def find_best_threshold(thresholds, predicts,target):
    best_threshold = best_acc = 0
    for threshold in thresholds:
        accuracy = eval_acc(threshold, predicts,target)
        if accuracy >= best_acc:
            best_acc = accuracy
            best_threshold = threshold
    return best_threshold

def test_accuracy(predicts,targets):
    acc=[]
    thresholds = np.arange(-1.0, 1.0, 0.005)
    best_thresh = find_best_threshold(thresholds, predicts, targets)
    acc.append(eval_acc(best_thresh,predicts,targets))

    return np.mean(acc)

class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

# train process
def train(args, model, device, train_loader, optimizer, epoch, loss_fn):
    print('Epoch:', epoch)
    print('Training:')
    model.feature = False
    top1 = AverageMeter()
    train_loss = 0
    model.train()

    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        outputs = model(data)
        loss = loss_fn(outputs, target)
        train_loss += loss
        loss.backward()
        optimizer.step()
        # measure accuracy and record loss
        acc = train_accuracy(outputs[0], target)[0].item()
        # pred = torch.max(outputs.data, 1)
        top1.update(acc, data.size(0))

    train_loss /= len(train_loader.dataset)
    print('Training Loss: {:.4f}'.format(train_loss))
    print('Training acc:{:.4f}'.format(top1.val))

# test process
def test(model, device, test_loader, loss_fn):
    print('Testing:')
    model.feature = True
    predicts=torch.tensor([],device=device)
    targets=torch.tensor([],device=device)
    model.eval()
    cos = nn.CosineSimilarity(dim=1, eps=1e-6)
    with torch.no_grad():
        for batch_idx, (img1, img2, target) in enumerate(test_loader):
            f1 = model(img1.to(device))
            f2 = model(img2.to(device))
            target = target.to(device)
            cs = cos(f1,f2)
            predicts = torch.cat((predicts,cs))
            targets = torch.cat((targets,target))
    acc = test_accuracy(predicts,targets)
    print('Testing acc:{:.4f}'.format(acc))
    print('*'*10)

# Using argparse to specify hyperparameters
parser = argparse.ArgumentParser(description='PyTorch Implementation of Sphereface')
parser.add_argument('--dataset', type=str, default='lfw', metavar='N',
                    help='dataset')
parser.add_argument('--bs', type=int, default=64, metavar='N',
                    help='input batch size for training (default: 64)')
parser.add_argument('--test-bs', type=int, default=64, metavar='N',
                    help='input batch size for testing (default: 64)')
parser.add_argument('--epochs', type=int, default=10, metavar='N',
                    help='number of epochs to train (default: 10)')
parser.add_argument('--lr', type=float, default=0.001, metavar='LR',
                    help='learning rate (default: 0.001)')
parser.add_argument('--gamma', type=float, default=0.7, metavar='M',
                    help='Learning rate step gamma (default: 0.7)')
parser.add_argument('--cuda', action='store_true', default=True,
                    help='use cuda or not')
parser.add_argument('--dry-run', action='store_true', default=False,
                    help='quickly check a single pass')
parser.add_argument('--seed', type=int, default=42, metavar='S',
                    help='random seed (default: 42)')
parser.add_argument('--save-model', action='store_true', default=False,
                    help='For Saving the current Model')
args = parser.parse_args()

# set seed, device
torch.manual_seed(args.seed)
use_cuda = args.cuda and torch.cuda.is_available()
device = torch.device("cuda" if use_cuda else "cpu")
print('device:', device)

# generate train and test loader
train_loader, test_loader = get_loader(args.dataset, batch_size=args.bs)

# initialize model, optimizer and loss function
model = SphereFace4().to(device)
optimizer = optim.Adam(model.parameters(), lr=args.lr)
scheduler = StepLR(optimizer, step_size=10, gamma=args.gamma)
loss_fn = AngularSoftmaxWithLoss()

# run for n epochs
print('*'*10)
for epoch in range(1, args.epochs + 1):
    train(args, model, device, train_loader, optimizer, epoch, loss_fn)
    test(model, device, test_loader, loss_fn)
    scheduler.step()

# save model
if args.save_model:
    torch.save(model.state_dict(), "sphereface_lfw.pt")
