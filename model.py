import torch.nn as nn
from icnnmodel import Stage2FaceModel
import torch
import numpy as np
import torch.nn.functional as F
from src import detect_faces
import torchvision.transforms.functional as TF
from PIL import Image

# save np.load
np_load_old = np.load
# modify the default parameters of np.load
np.load = lambda *a, **k: np_load_old(*a, allow_pickle=True, **k)


class Stage1Model(nn.Module):
    def __init__(self):
        super(Stage1Model, self).__init__()
        self.parts_locate = getFacePoints()
        self.model = SelectNet()
        self.parts = None
        self.labels = None
        self.points = None

    def forward(self, img, label):
        self.points = self.parts_locate(img)
        points = self.points.to(img.device)
        self.parts, self.labels = self.model(img, label, points)
        return self.parts, self.labels


class getFacePoints(nn.Module):
    def __init__(self):
        super(getFacePoints, self).__init__()
        self.img = None
        self.points = None

    def forward(self, img):
        self.img = img
        N = img.shape[0]
        out = []
        for i in range(N):
            img_in = TF.to_pil_image(self.img[i].detach().cpu())
            bounding_boxes, landmarks = detect_faces(img_in)
            out.append(np.stack([landmarks[-1][0:5], landmarks[-1][5:10]]).T)
        self.points = np.array(out)
        self.points = np.concatenate([self.points[:, :-2],
                                      np.expand_dims(((self.points[:, -1] + self.points[:, -2]) / 2), axis=1)], axis=1)
        self.points = torch.from_numpy(self.points)
        return self.points


class SelectNet(nn.Module):
    def __init__(self):
        super(SelectNet, self).__init__()
        self.theta = None
        self.points = None
        self.device = None

    def get_theta(self):
        # points in [N, 4, 2]
        points_in = self.points
        # print(points_in)
        N = points_in.shape[0]
        points_in[:, :, 0] = 256 - 6 * points_in[:, :, 0]
        points_in[:, :, 1] = 256 - 6 * points_in[:, :, 1]
        param = torch.zeros((N, 4, 2, 3)).to(self.device)
        param[:, :, 0, 0] = 6
        param[:, :, 0, 2] = points_in[:, :, 0]
        param[:, :, 1, 1] = 6
        param[:, :, 1, 2] = points_in[:, :, 1]
        # Param Shape(N, 4, 2, 3)
        # Every label has a affine param
        ones = torch.tensor([[0., 0., 1.]]).repeat(N, 4, 1, 1).to(self.device)
        param = torch.cat([param, ones], dim=2)
        param = torch.inverse(param)
        # ---               ---
        # Then, convert all the params to thetas
        self.theta = torch.zeros([N, 4, 2, 3]).to(self.device)
        self.theta[:, :, 0, 0] = param[:, :, 0, 0]
        self.theta[:, :, 0, 1] = param[:, :, 0, 1]
        self.theta[:, :, 0, 2] = param[:, :, 0, 2] * 2 / 512 + self.theta[:, :, 0, 0] + self.theta[:, :, 0, 1] - 1
        self.theta[:, :, 1, 0] = param[:, :, 1, 0]
        self.theta[:, :, 1, 1] = param[:, :, 1, 1]
        self.theta[:, :, 1, 2] = param[:, :, 1, 2] * 2 / 512 + self.theta[:, :, 1, 0] + self.theta[:, :, 1, 1] - 1
        # theta Shape(N, 4, 2, 3)
        return self.theta

    def forward(self, face, label, points):
        self.points = points.clone()
        theta = self.get_theta()
        self.device = face.device
        n, l, h, w = face.shape
        samples = []
        labels = []
        for i in range(4):
            grid = F.affine_grid(theta[:, i], [n, l, 64, 64], align_corners=True).to(theta.device)
            samples.append(F.grid_sample(input=face, grid=grid, align_corners=True))
            labels.append(F.grid_sample(input=label, grid=grid, align_corners=True))
        samples = torch.stack(samples, dim=0)
        samples = samples.transpose(1, 0)
        labels = torch.cat(labels, dim=1)
        # print(labels.shape)
        assert samples.shape == (n, 4, 3, 64, 64)
        assert labels.shape == (n, 4, 64, 64)
        # Shape (N, c, 3, 64, 64)
        return samples, labels


class Stage2Model(nn.Module):
    def __init__(self):
        super(Stage2Model, self).__init__()
        self.eye1_model = EyeModel()
        self.eye2_model = EyeModel()
        self.nose_model = NoseModel()
        self.mouth_model = MouthModel()

    def forward(self, eye1, eye2, nose, mouth):
        # Shape(N, 3, 64, 64)
        out_eye1 = self.eye1_model(eye1)
        out_eye2 = self.eye1_model(eye2)
        out_nose = self.nose_model(nose)
        out_mouth = self.mouth_model(mouth)

        return out_eye1, out_eye2, out_nose, out_mouth


class PartsModel(nn.Module):
    def __init__(self):
        super(PartsModel, self).__init__()
        self.model = Stage2FaceModel()

    def forward(self, x):
        # x Shape (N, 3, 64, 64)
        out = self.model(x)
        return out


class EyeModel(PartsModel):
    def __init__(self):
        super(EyeModel, self).__init__()
        self.model.set_label_channels(3)


class NoseModel(PartsModel):
    def __init__(self):
        super(NoseModel, self).__init__()
        self.model.set_label_channels(2)


class MouthModel(PartsModel):
    def __init__(self):
        super(MouthModel, self).__init__()
        self.model.set_label_channels(4)


class TwoStagePipeLine(nn.Module):
    def __init__(self):
        super(TwoStagePipeLine, self).__init__()
        self.stage1_model = Stage1Model()
        self.stage2_model = Stage2Model()

    def forward(self, img, label):
        parts, labels = self.stage1_model(img, label)
        eye1, eye2, nose, mouth = parts[:, 0], parts[:, 1], parts[:, 2], parts[:, 3]
        eye1_pred, eye2_pred, nose_pred, mouth_pred = self.stage2_model(eye1, eye2, nose, mouth)

        return labels, eye1_pred, eye2_pred, nose_pred, mouth_pred
