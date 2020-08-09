import torch
import torch.nn as nn
import torch.nn.functional as F

__all__ = ["STN"]


class STN(nn.Module):
    """Spatial Transformation Network
    Allows a network to learn a affine transform of an image or filter 
    produced by the network.
    
    https://arxiv.org/pdf/1506.02025.pdf
    """

    def __init__(self, localization=None):
        super().__init__()

        self.fc_inshape = None

        if localization == None:
            self.localization = nn.Sequential(
                nn.Conv2d(1, 8, kernel_size=7),
                nn.MaxPool2d(2, stride=2),
                nn.ReLU(True),
                nn.Conv2d(8, 10, kernel_size=5),
                nn.MaxPool2d(2, stride=2),
                nn.ReLU(True),
            )
        else:
            self.localization = localization

        # Regressor for the 3 * 2 affine matrix
        self.fc_loc = nn.Sequential(
            nn.Linear(40, 32), nn.ReLU(True), nn.Linear(32, 3 * 2)
        )

    def _find_inshape(self, x):
        with torch.no_grad():
            self.localization[0] = nn.Conv2d(x.shape[1], 8, 7).to(x.device)
            shape = self.localization(torch.zeros_like(x, device=x.device)).shape[1:]
            self.fc_inshape = torch.tensor(shape).prod()
            self.fc_loc[0] = nn.Linear(self.fc_inshape, 32).to(x.device)

    def forward(self, x):
        # Setup the in channels etc for the first time if we haven't already
        if self.fc_inshape == None:
            self._find_inshape(x)

        xs = self.localization(x)
        xs = xs.view(x.shape[0], -1)

        # Get the affine tranformation matrix
        theta = self.fc_loc(xs)
        theta = theta.view(-1, 2, 3)

        # Perform the transformation
        grid = F.affine_grid(theta, x.size())
        x = F.grid_sample(x, grid)
        return x
