from dataset import Stage1Dataset
from preprogress import Resize, Stage1ToTensor
from torchvision import transforms
from torch.utils.data import DataLoader
from model import Stage1Model
import torch


txt_file_names = {
    'train': "exemplars.txt",
    'val': "tuning.txt"
}
root_dir = "/data1/yinzi/datas"
transform = transforms.Compose([Resize((512, 512)),
                                Stage1ToTensor()
                                ])
Dataset = {x: Stage1Dataset(txt_file=txt_file_names[x],
                            root_dir=root_dir,
                            transform=transform
                            )
           for x in ['train', 'val']
           }
Loader = {x: DataLoader(Dataset[x], batch_size=16,
                        shuffle=True, num_workers=4)
          for x in ['train', 'val']
          }
eval_loader = Loader['val']

device = torch.device("cuda:%d" % 5 if torch.cuda.is_available() else "cpu")
model = Stage1Model().to(device)
for i, batch in enumerate(eval_loader):
    img = batch['image']
    label = batch['labels']
    y = model(img, label)
    print(y)
