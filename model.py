import torch.nn as nn
from icnnmodel import Stage2FaceModel


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
