from torch.utils.data import DataLoader
from template import TemplateModel, F1Accuracy
from model import Stage2Model
from tensorboardX import SummaryWriter
import torch.optim as optim
import torch
import torch.nn as nn
import argparse
import uuid
import numpy as np
from torchvision import transforms
from dataset import PartsDataset, Stage2Augmentation
from preprogress import Stage2Resize, ToTensor
import os

uuid = str(uuid.uuid1())[0:8]
parser = argparse.ArgumentParser()
parser.add_argument("--cuda", default=0, type=int, help="Choose which GPU")
parser.add_argument("--batch_size", default=20, type=int, help="Batch size to use during training.")
parser.add_argument("--display_freq", default=20, type=int, help="Display frequency")
parser.add_argument("--lr", default=0.001, type=float, help="Learning rate for optimizer")
parser.add_argument("--epochs", default=50, type=int, help="Number of epochs to train")
parser.add_argument("--eval_per_epoch", default=1, type=int, help="eval_per_epoch ")
parser.add_argument("--workers", default=10, type=int, help="dataloader fetch workers")
parser.add_argument("--momentum", default=0.9, type=float, help="momentum ")
parser.add_argument("--weight_decay", default=0.005, type=float, help="weight_decay ")
parser.add_argument("--model_path", default=None, type=str, help="Last trained model")
parser.add_argument("--optimizer", default='Adam', type=str, help="Last trained model")
args = parser.parse_args()
print(args)


class TrainClass(TemplateModel):
    def __init__(self, dataset_class, txt_file, root_dir, transform, num_workers):
        super(TrainClass, self).__init__()
        self.args = args
        self.device = torch.device("cuda:%d" % args.cuda if torch.cuda.is_available() else "cpu")
        self.writer = SummaryWriter('log_three')
        self.model = Stage2Model().to(self.device)
        if args.optimizer == 'Adam':
            self.optimizer = [optim.Adam(self.model.parameters(), self.args.lr)
                              for _ in range(4)]
        else:
            self.optimizer = [optim.SGD(self.model.parameters(), self.args.lr, momentum=self.args.momentum,
                                        weight_decay=self.args.weight_decay)
                              for _ in range(4)]
        self.criterion = [nn.CrossEntropyLoss()
                          for _ in range(4)]
        # self.metric = nn.CrossEntropyLoss()
        self.metric = [F1Accuracy()
                       for _ in range(4)]
        self.train_loader = None
        self.eval_loader = None
        self.ckpt_dir = "checkpoint_%s" % uuid
        self.display_freq = args.display_freq
        self.scheduler = [optim.lr_scheduler.StepLR(self.optimizer[i], step_size=5, gamma=0.5)
                          for i in range(4)]
        self.best_error = [float('Inf'), float('Inf'), float('Inf'), float('Inf')]
        self.best_accu = [float('-Inf'), float('-Inf'), float('-Inf'), float('-Inf')]
        self.load_dataset(dataset_class, txt_file, root_dir, transform, num_workers)

    def load_dataset(self, dataset_class, txt_file, root_dir, transform, num_workers):

        data_after = Stage2Augmentation(dataset=dataset_class,
                                        txt_file=txt_file,
                                        root_dir=root_dir,
                                        resize=(64, 64)
                                        )

        Dataset = data_after.get_dataset()
        # Dataset = {x: dataset_class(txt_file=txt_file[x],
        #                             root_dir=root_dir,
        #                             transform=transform
        #                             )
        #            for x in ['train', 'val']
        #            }
        Loader = {x: DataLoader(Dataset[x], batch_size=args.batch_size,
                                shuffle=True, num_workers=num_workers)
                  for x in ['train', 'val']
                  }
        self.train_loader = Loader['train']
        self.eval_loader = Loader['val']

        return Loader

    def train(self):
        self.model.train()
        self.epoch += 1
        for i, batch in enumerate(self.train_loader):
            self.step += 1
            for k in range(4):
                self.optimizer[k].zero_grad()
            loss, others = self.train_loss(batch)
            for k in range(4):
                loss[k].backward()
                self.optimizer[k].step()
            loss_item = [loss[k].item()
                         for k in range(4)]
            if self.step % self.display_freq == 0:
                self.writer.add_scalar('loss_eye1_%s' % uuid, loss_item[0], self.step)
                self.writer.add_scalar('loss_eye2_%s' % uuid, loss_item[1], self.step)
                self.writer.add_scalar('loss_nose_%s' % uuid, loss_item[2], self.step)
                self.writer.add_scalar('loss_mouth_%s' % uuid, loss_item[3], self.step)
                print('epoch {}\tstep {}\n'
                      'loss_eye1 {:.3}\tloss_eye2 {:.3}\tloss_nose {:.3}\tloss_mouth {:.3}\n'
                      'loss_all_mean {:.3}'.format(
                    self.epoch, self.step, loss_item[0], loss_item[1], loss_item[2], loss_item[3], np.mean(loss_item)
                ))
                if self.train_logger:
                    self.train_logger(self.writer, others)
        torch.cuda.empty_cache()

    def train_loss(self, batch):
        x = batch['image'].to(self.device)
        y = batch['labels']
        # ['eye1', 'eye2', 'nose', 'mouth']
        eye1 = x[:, 0]
        eye2 = x[:, 1]
        nose = x[:, 2]
        mouth = x[:, 3]

        eye1_label = y['eye1'].to(self.device)
        eye2_label = y['eye2'].to(self.device)
        nose_label = y['nose'].to(self.device)
        mouth_label = y['mouth'].to(self.device)
        eye1_pred, eye2_pred, nose_pred, mouth_pred = self.model(eye1, eye2, nose, mouth)
        eye1_loss = self.criterion[0](eye1_pred, eye1_label.argmax(dim=1, keepdim=False))
        eye2_loss = self.criterion[1](eye2_pred, eye2_label.argmax(dim=1, keepdim=False))
        nose_loss = self.criterion[2](nose_pred, nose_label.argmax(dim=1, keepdim=False))
        mouth_loss = self.criterion[3](mouth_pred, mouth_label.argmax(dim=1, keepdim=False))

        loss = [eye1_loss, eye2_loss, nose_loss, mouth_loss]

        return loss, None

    def eval_error(self):
        error = []
        counts = 0
        for i, batch in enumerate(self.eval_loader):
            counts += 1
            x = batch['image'].to(self.device)
            y = batch['labels']
            eye1 = x[:, 0]
            eye2 = x[:, 1]
            nose = x[:, 2]
            mouth = x[:, 3]
            eye1_label = y['eye1'].to(self.device)
            eye2_label = y['eye2'].to(self.device)
            nose_label = y['nose'].to(self.device)
            mouth_label = y['mouth'].to(self.device)
            eye1_pred, eye2_pred, nose_pred, mouth_pred = self.model(eye1, eye2, nose, mouth)
            eye1_error = self.metric[0](eye1_pred, eye1_label.argmax(dim=1, keepdim=False))
            eye2_error = self.metric[1](eye2_pred, eye2_label.argmax(dim=1, keepdim=False))
            nose_error = self.metric[2](nose_pred, nose_label.argmax(dim=1, keepdim=False))
            mouth_error = self.metric[3](mouth_pred, mouth_label.argmax(dim=1, keepdim=False))
            error.append([eye1_error.item(), eye2_error.item(), nose_error.item(), mouth_error.item()])

        error = np.mean(error, axis=0)
        return error, None

    def eval(self):
        self.model.eval()
        with torch.no_grad():
            error, others = self.eval_error()

        if os.path.exists(self.ckpt_dir) is False:
            os.makedirs(self.ckpt_dir)

        name = ['eye1', 'eye2', 'nose', 'mouth']
        for j in range(4):
            if error[j] < self.best_error[j]:
                self.best_error[j] = error[j]
                self.save_state(os.path.join(self.ckpt_dir, 'best_%s.pth.tar' % name[j]), False)
        self.save_state(os.path.join(self.ckpt_dir, '{}.pth.tar'.format(self.epoch)), False)
        self.writer.add_scalar('error_eye1%s' % uuid, error[0], self.epoch)
        self.writer.add_scalar('error_eye2%s' % uuid, error[1], self.epoch)
        self.writer.add_scalar('error_nose%s' % uuid, error[2], self.epoch)
        self.writer.add_scalar('error_mouth%s' % uuid, error[3], self.epoch)
        print('\n==============================')
        print('epoch {} finished\n'
              'error_eye1 {:.3}\terror_eye2 {:.3}\terror_nose {:.3}\terror_mouth {:.3}\n'
              'best_error_eye1 {:.3}\tbest_error_eye2 {:.3}\tbest_error_nose {:.3}\tbest_error_mouth {:.3}\n'
              'best_error_mean {:.3}'
              .format(self.epoch, error[0], error[1], error[2], error[3],
                      self.best_error[0], self.best_error[1], self.best_error[2], self.best_error[3],
                      np.mean(self.best_error)))
        print('==============================\n')
        if self.eval_logger:
            self.eval_logger(self.writer, others)

        torch.cuda.empty_cache()
        return error

    def load_state(self, fname, optim=True, map_location=None):
        path = [os.path.join(fname, 'best_%s.pth.tar' % x)
                for x in ['eye1', 'eye2', 'nose', 'mouth']]
        state = [torch.load(path[i], map_location=map_location)
                 for i in range(4)]

        if isinstance(self.model, torch.nn.DataParallel):

            self.model.module.load_state_dict(state[0]['model'])
            best_eye1 = self.model.module.eye1_model
            self.model.module.load_state_dict(state[1]['model'])
            best_eye2 = self.model.module.eye2_model
            self.model.module.load_state_dict(state[2]['model'])
            best_nose = self.model.module.nose_model
            self.model.module.load_state_dict(state[3]['model'])
            best_mouth = self.model.module.mouth_model
            self.model.module.eye1_model = best_eye1
            self.model.module.eye2_model = best_eye2
            self.model.module.nose_model = best_nose
            self.model.module.mouth_model = best_mouth

        else:
            self.model.load_state_dict(state[0]['model'])
            best_eye1 = self.model.eye1_model
            self.model.load_state_dict(state[1]['model'])
            best_eye2 = self.model.eye2_model
            self.model.load_state_dict(state[2]['model'])
            best_nose = self.model.nose_model
            self.model.load_state_dict(state[3]['model'])
            best_mouth = self.model.mouth_model
            self.model.eye1_model = best_eye1
            self.model.eye2_model = best_eye2
            self.model.nose_model = best_nose
            self.model.mouth_model = best_mouth

            if optim and 'optimizer' in state:
                for i in range(4):
                    self.optimizer[i].load_state_dict(state[i]['optimizer'])
            self.best_error = [state[0]['best_error'][0], state[1]['best_error'][1],
                               state[2]['best_error'][2], state[3]['best_error'][3]]

            print('load model from {}'.format(fname))


class Train_F1_eval(TrainClass):

    def eval(self):
        self.model.eval()
        with torch.no_grad():
            accu, others = self.eval_accu()

        if os.path.exists(self.ckpt_dir) is False:
            os.makedirs(self.ckpt_dir)

        name = ['eye1', 'eye2', 'nose', 'mouth']
        for j in range(4):
            if accu[j] > self.best_accu[j]:
                self.best_accu[j] = accu[j]
                self.save_state(os.path.join(self.ckpt_dir, 'best_%s.pth.tar' % name[j]), False)
        self.save_state(os.path.join(self.ckpt_dir, '{}.pth.tar'.format(self.epoch)), False)
        self.writer.add_scalar('accu_eye1%s' % uuid, accu[0], self.epoch)
        self.writer.add_scalar('accu_eye2%s' % uuid, accu[1], self.epoch)
        self.writer.add_scalar('accu_nose%s' % uuid, accu[2], self.epoch)
        self.writer.add_scalar('accu_mouth%s' % uuid, accu[3], self.epoch)
        print('\n==============================')
        print('epoch {} finished\n'
              'accu_eye1 {:.3}\taccu_eye2 {:.3}\taccu_nose {:.3}\taccu_mouth {:.3}\n'
              'best_accu_eye1 {:.3}\tbest_accu_eye2 {:.3}\tbest_accu_nose {:.3}\tbest_accu_mouth {:.3}\n'
              'best_accu_mean {:.3}'
              .format(self.epoch, accu[0], accu[1], accu[2], accu[3],
                      self.best_accu[0], self.best_accu[1], self.best_accu[2], self.best_accu[3],
                      np.mean(self.best_accu)))
        print('==============================\n')
        if self.eval_logger:
            self.eval_logger(self.writer, others)

        torch.cuda.empty_cache()
        return accu

    def eval_accu(self):
        accu = []
        counts = 0
        for i, batch in enumerate(self.eval_loader):
            counts += 1
            x = batch['image'].to(self.device)
            y = batch['labels']
            eye1 = x[:, 0]
            eye2 = x[:, 1]
            nose = x[:, 2]
            mouth = x[:, 3]
            eye1_label = y['eye1'].to(self.device)
            eye2_label = y['eye2'].to(self.device)
            nose_label = y['nose'].to(self.device)
            mouth_label = y['mouth'].to(self.device)
            eye1_pred, eye2_pred, nose_pred, mouth_pred = self.model(eye1, eye2, nose, mouth)
            eye1_accu = self.metric[0](eye1_pred, eye1_label)
            eye2_accu = self.metric[1](eye2_pred, eye2_label)
            nose_accu = self.metric[2](nose_pred, nose_label)
            mouth_accu = self.metric[3](mouth_pred, mouth_label)
            accu.append([eye1_accu, eye2_accu, nose_accu, mouth_accu])

        accu = np.mean(accu, axis=0)
        return accu, None


class TrainMGPU(Train_F1_eval):
    def __init__(self, dataset_class, txt_file, root_dir, transform, num_workers):
        super(TrainMGPU, self).__init__(dataset_class, txt_file, root_dir, transform, num_workers)
        self.model = nn.DataParallel(Stage2Model(), device_ids=[0, 1, 2, 3, 4])
        self.model = self.model.to(self.device)


def start_train(model_path=None):
    dataset_class = PartsDataset
    txt_file_names = {
        'train': "exemplars.txt",
        'val': "tuning.txt"
    }
    root_dir = "/data1/yinzi/facial_parts"
    transform = transforms.Compose([Stage2Resize((64, 64)),
                                    ToTensor()
                                    ])

    train = Train_F1_eval(dataset_class, txt_file_names, root_dir, transform, num_workers=args.workers)
    if model_path:
        train.load_state(model_path)

    for epoch in range(args.epochs):
        train.train()
        for i in range(4):
            train.scheduler[i].step()
        if (epoch + 1) % args.eval_per_epoch == 0:
            train.eval()

    print('Done!!!')


start_train(model_path=args.model_path)
# start_train(model_path="/home/yinzi/data3/vimg18/python_projects/three_face/checkpoint_ab736fa7/")
