class RPS_MNet(nn.Module):
    """
        Model inspired by [Aoe at al., 10.1038/s41598-019-41500-x] integrated with bandpower.
    """
    class Showsize(nn.Module):
        def __init__(self):
            super(RPS_MNet.Showsize, self).__init__()
        def forward(self, x):
            # print(x.shape)
            return x

    def __init__(self, n_times):
        """
        Args:
            n_times (int):
                n_times dimension of the input data.
        """
        super(RPS_MNet, self).__init__()
        # if n_times == 501:  # TODO automatic n_times
        #     self.n_times = 12
        # elif n_times == 601:
        #     self.n_times = 18
        # elif n_times == 701:
        #     self.n_times = 24
        # else:
        #     raise ValueError(
        #         "Network can work only with n_times = 501, 601, 701 "
        #         "(epoch duration of 1., 1.2, 1.4 sec),"
        #         " got instead {}".format(n_times)
        #     )
        self.n_times = n_times
        self.spatial = nn.Sequential(
            nn.Conv2d(1, 32, stride=(1, 1), kernel_size=(272,64), bias=True), #kernel size 204, 64
            nn.ReLU(),
            # nn.BatchNorm2d(32),
            nn.Conv2d(32, 64, stride=(1, 1), kernel_size=(1, 16), bias=True), # kernel size 1,16
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=(1, 2), stride=(1, 2)),
            # nn.BatchNorm2d(64),
        )

        self.temporal = nn.Sequential(
            nn.Conv2d(1, 32, stride=(1, 1), kernel_size=(8, 8), bias=True),
            nn.ReLU(),
            # nn.BatchNorm2d(32),
            nn.Conv2d(32, 32, stride=(1, 1), kernel_size=(8, 8), bias=True),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=(5, 3), stride=(5, 3)),
            # nn.BatchNorm2d(32),
            nn.Conv2d(32, 64, stride=(1, 1), kernel_size=(1, 4), bias=True),
            nn.ReLU(),
            # nn.BatchNorm2d(64),
            nn.Conv2d(64, 64, stride=(1, 1), kernel_size=(1, 4), bias=True), #conv6
            nn.ReLU(),
            # nn.BatchNorm2d(64),
            nn.MaxPool2d(kernel_size=(1, 2), stride=(1, 2)),
            nn.Conv2d(64, 128, stride=(1, 1), kernel_size=(1, 2), bias=True),
            nn.ReLU(),
            nn.Dropout2d(p=0.3),
            # nn.BatchNorm2d(128),
            nn.Conv2d(128, 128, stride=(1, 1), kernel_size=(1, 2), bias=True),
            nn.ReLU(),
            nn.Dropout2d(p=0.3),
            # nn.BatchNorm2d(128),
            nn.MaxPool2d(kernel_size=(1, 2), stride=(1, 2)),
            nn.Conv2d(128, 256, stride=(1, 1), kernel_size=(1, 2), bias=True),
            nn.ReLU(),
            nn.Dropout2d(p=0.3),
            # nn.BatchNorm2d(256),
            nn.Conv2d(256, 256, stride=(1, 1), kernel_size=(1, 2), bias=True), #conv10
            nn.ReLU(),
            nn.Dropout2d(p=0.3),
            # nn.BatchNorm2d(256),
            nn.MaxPool2d(kernel_size=(1, 2), stride=(1, 2)),
        )

        self.attention1 = nn.Sequential(
            ChannelAttention([None, 256, 26, self.n_times]),
        )

        self.attention2 = nn.Sequential(
            SpatialAttention(),
        )

        self.concatenate = Concatenate()

        self.flatten = Flatten_MEG()

        self.ff1 = nn.Sequential(
            # nn.Linear(256 * 26 * self.n_times + 272 * 6, 1024),
            nn.Linear(4 , 1024),
            # nn.BatchNorm1d(num_features=1024),
            nn.ReLU(),
            self.Showsize(),
            nn.Linear(1024, 1024),
            nn.ReLU(),
            nn.BatchNorm1d(num_features=1024),
            # nn.ReLU(),
            nn.Dropout(0.3),
            self.Showsize(),
        )
        self.ff2 = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(2656 , 14),
        )
        self.softmax = nn.Softmax(dim =1)

    def forward(self, x, pb):
        x = self.spatial(x)
        x = torch.transpose(x, 1, 2)
        x = self.temporal(x)
        x = self.attention1(x)
        x = self.attention2(x)
        x = self.ff1(self.flatten(x))
        x = self.concatenate(x, pb)
        x = self.ff2(x)
        x = self.softmax(x)

        return x.squeeze(1)
    
class LSTM1(nn.Module):
    def __init__(self, num_classes, input_size, hidden_size, num_layers, seq_length):
        super(LSTM1, self).__init__()
        self.num_classes = num_classes #number of classes
        self.num_layers = num_layers #number of layers
        self.input_size = input_size #input size
        self.hidden_size = hidden_size #hidden state
        self.seq_length = seq_length #sequence length

        self.lstm = nn.LSTM(input_size=input_size, hidden_size=hidden_size,
                          num_layers=num_layers, batch_first=True) #lstm
        self.fc_1 =  nn.Linear(hidden_size, 128) #fully connected 1
        self.fc = nn.Linear(128, num_classes) #fully connected last layer

        self.relu = nn.ReLU()
    
    def forward(self,x):
        h_0 = Variable(torch.zeros(self.num_layers, x.size(0), self.hidden_size)).to(device) #hidden state
        c_0 = Variable(torch.zeros(self.num_layers, x.size(0), self.hidden_size)).to(device) #internal state
        # Propagate input through LSTM
        output, (hn, cn) = self.lstm(x, (h_0, c_0)) #lstm with input, hidden, and internal state
        hn = hn.view(-1, self.hidden_size) #reshaping the data for Dense layer next
        out = self.relu(hn)
        out = self.fc_1(out) #first Dense
        out = self.relu(out) #relu
        out = self.fc(out) #Final Output
        return out
    
class CNN(nn.Module):
    def __init__(self):
      # super(CNN, self)._init_()
      super(CNN, self).__init__()
      self.n_classes = 14
      n_classes =14
      self.conv1 = nn.Sequential(
          nn.Conv2d(
              in_channels=1,
              out_channels=32,
              kernel_size=3,
              stride=1,
              padding=1,
          ),
          nn.ReLU(),
          nn.MaxPool2d(kernel_size=2,stride=2),
      )
      self.conv2 = nn.Sequential(
          nn.Conv2d(32,64,3,1,1),
          nn.ReLU(),
          nn.MaxPool2d (2,2),
      )
      # self.fc = nn.Linear(64*7*7,128)
      self.fc = nn.Linear(139264, 100)
      self.out = nn.Linear(100,n_classes)
      self.softmax = nn.Softmax()

    def forward(self,x):
      x=self.conv1(x)
      x=self.conv2(x)
      # x=x.view(x.size(0),-1)
      x = torch.flatten(x, 1) # flatten all dimensioxns except batch
      x=self.fc(x)
      x=self.out(x)      
      # output=self.out(x)
      # return output, x
      # x=self.softmax(x)
      return x