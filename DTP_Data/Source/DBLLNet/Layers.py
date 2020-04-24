from torch import nn

class Convolutional_Layer(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding = 1, activation=nn.ReLU, batch_norm=False, bias=True):
        super(Convolutional_Layer, self).__init__()
        self.conv = nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size,stride=stride, padding=padding, bias=bias)
        self.batch_norm = nn.BatchNorm2d(out_channels) if batch_norm else None
        self.activation = None if activation is None else activation()

    def forward(self, x):
        out = self.conv(x)
        if self.batch_norm is not None:
            out = self.batch_norm(out)
        if self.activation is not None:
            out = self.activation(out)
        #print("Conv" , out.shape)
        return out

class Fully_Connected_Layer(nn.Module):
    def __init__(self, in_features, out_features, activation=nn.ReLU, batch_norm=False, bias=True):
        super(Fully_Connected_Layer, self).__init__()
        self.FC = nn.Linear(in_features, out_features, bias=bias)
        self.batch_norm = nn.BatchNorm1d(out_features) if batch_norm else None
        self.activation = None if activation is None else activation()

    def forward(self, x):
        out = self.FC(x)
        if self.batch_norm is not None:
            out = self.batch_norm(out)
        if self.activation is not None:
            out = self.activation(out)
        #print("FC", out.shape)
        return out
