import torch
import torch.nn as nn
import torch.nn.functional as F

class TempoCNN(nn.Module):
    '''
    Reproduced from Schreiber & Müller (2018).
    
    Hendrik Schreiber and Meinard Müller. "A Single-Step Approach to Musical
        Tempo Estimation Using a Convolutional Neural Network." 2018.
        In Proceedings of the 19th International Society for Music Information Retrieval Conference (ISMIR).
    '''
    def __init__(self):
        super(TempoCNN, self).__init__()

        self.sfc = nn.Sequential(  # 3 SFC blocks
            *[SFC(in_channels=_in, out_channels=_out) for (_in, _out) in [(1, 16), (16, 16), (16, 16)]]
        )

        self.mfm = nn.Sequential(  # 4 MFM blocks
            *[MFM(in_channels=_in, mid_channels=_mid, out_channels=_out, avg_kernel_size=_avg)
              for (_in, _mid, _out, _avg) in [(16, 24, 36, 5), (36, 24, 36, 2), (36, 24, 36, 2), (36, 24, 36, 2)]]
        )

        self.tc = TempoClassifier(in_features=64)

        return

    def forward(self, melspec):
        melspec = melspec.unsqueeze(1)  # add channel axis
        
        out = self.sfc(melspec)
        out = self.mfm(out)
        pred_tempo = self.tc(out)
        return pred_tempo
    
    
class SFC(nn.Module):  # short filter conv layer
    def __init__(self, in_channels, out_channels, kernel_size=(5,1)):
        super(SFC, self).__init__()

        self.conv = nn.Sequential(
            nn.BatchNorm2d(in_channels),
            nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size),
            nn.ELU(),
        )
        return

    def forward(self, x):
        return self.conv(x)
    

class MFM(nn.Module): # multi-filter module (mf_mod)
    def __init__(self, in_channels, mid_channels, out_channels, avg_kernel_size, base_kernel_size=32, multipliers=[1, 2, 3, 4, 6, 8]):
        super(MFM, self).__init__()

        self.avg = nn.AvgPool2d(kernel_size=(1, avg_kernel_size))
        self.bn = nn.BatchNorm2d(in_channels)
        self.mf = nn.ModuleList([SFC(in_channels, mid_channels, (m * base_kernel_size, 1)) for m in multipliers])
        self.btn = nn.Conv2d(in_channels=mid_channels, out_channels=out_channels, kernel_size=(1, 1))
        return

    def forward(self, x):
        out = self.avg(x)
        out = self.bn(out)
        out = torch.cat([f(out) for f in self.mf], dim=1)
        return self.btn(out)


class TempoClassifier(nn.Module):
    def __init__(self, in_features):
        super(TempoClassifier, self).__init__()

        self.p = nn.Sequential(
            nn.BatchNorm2d(in_features),
            nn.Dropout2d(0.5)
            )
        self.q = nn.Sequential(
            nn.Linear(in_features=in_features, out_features=64), 
            nn.ELU()
            )
        self.r = nn.Sequential(
            nn.BatchNorm1d(64), 
            nn.Linear(in_features=64, out_features=64), 
            nn.ELU()
            )
        self.s = nn.Sequential(
            nn.BatchNorm1d(64), 
            nn.Linear(in_features=64, out_features=256),
            )
        return

    def forward(self, x):
        out = self.p(x)
        out = self.q(out)
        out = self.r(out)
        logits = self.s(out)  # optimise for logits (log-likelihoods) for numerical stability

        if not self.training:
            return F.softmax(logits, dim=1)  # return probabilities at inference
        
        return logits