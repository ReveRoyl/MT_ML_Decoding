class RPS_MNet_ivan(nn.Module):
    """
        Model inspired by [Aoe at al., 10.1038/s41598-019-41500-x] integrated with bandpower.
    """

    def __init__(self, n_times):
        """
        Args:
            n_times (int):
                n_times dimension of the input data.
        """
        super(RPS_MNet_ivan, self).__init__()
        if n_times == 250:  # TODO automatic n_times
            self.n_times = 10
        elif n_times == 601:
            self.n_times = 18  # to check
        elif n_times == 701:
            self.n_times = 24  # to check
        else:
            raise ValueError(
                "Network can work only with n_times = 250, 601, 701 (epoch duration of 1., 1.2, 1.4 sec),"
                " got instead {}".format(n_times)
            )

    
        self.spatial = nn.Sequential(
                    nn.Conv2d(1, 16, stride=(1, 1), kernel_size=[272, 16],
                              bias=True),
                    nn.ReLU(),
                    nn.Conv2d(16, 32, kernel_size=[1, 16], bias=True),
                    nn.ReLU(),
                    # CBAM([None, 64, 1, 204]),
                    nn.MaxPool2d(kernel_size=[1, 2], stride=(1, 2)),
                    # nn.BatchNorm2d(64),
                )

        self.temporal = nn.Sequential(
                    nn.Conv2d(1, 16, kernel_size=[5, 5], bias=True),
                    nn.ReLU(),
                    nn.Conv2d(16, 16, kernel_size=[5, 5], bias=True),
                    nn.ReLU(),
                    # CBAM([None, 16, 24, 102], reduction_factor=2),
                    nn.MaxPool2d(kernel_size=[2, 3], stride=(2, 3)),
                    # nn.BatchNorm2d(16),
                    ###########################################################
                    nn.Conv2d(16, 32, kernel_size=[4, 4], bias=True),
                    nn.ReLU(),
                    nn.Conv2d(32, 32, kernel_size=[4, 4], bias=True),
                    nn.ReLU(),
                    # CBAM([None, 32, 6, 28], reduction_factor=2),
                    nn.MaxPool2d(kernel_size=[1, 2], stride=(1, 2)),
                    # nn.BatchNorm2d(32),
                    ###########################################################
                    nn.Conv2d(32, 64, kernel_size=[3, 3], bias=True),
                    nn.ReLU(),
                    nn.Conv2d(64, 64, kernel_size=[3, 3], bias=True),
                    nn.ReLU(),
                    # # CBAM([None, 128, 34, 9]),
                    # nn.Dropout2d(p=0.3),
                    # nn.BatchNorm2d(64),
                    ###########################################################
                    # nn.Conv2d(128, 256, kernel_size=[3, 3], bias=True),
                    # nn.ReLU(),
                    # nn.Conv2d(256, 256, kernel_size=[3, 3], bias=False),
                    # nn.ReLU(),
                    # # CBAM([None, 256, 30, self.n_times]),
                    # nn.Dropout2d(p=0.3),
                    # nn.BatchNorm2d(64),
                )

        self.concatenate = Concatenate()

        # self.flatten = Flatten_MEG()

        self.ff = nn.Sequential(nn.Linear(64 * 2 * self.n_times + 204 * 6, 512),
                                nn.BatchNorm1d(num_features=512),
                                nn.ReLU(),
                                nn.Dropout(0.3),
                                nn.Linear(512, 1024),
                                nn.BatchNorm1d(num_features=1024),
                                nn.ReLU(),
                                nn.Dropout(0.3),
                                nn.Linear(1024, 1))
    def forward(self, x, pb):
        x = self.spatial(x)
        x = torch.transpose(x, 1, 2)
        x = self.temporal(x)
        # x = self.attention(x)
        x = self.concatenate(x, pb)
        x = self.ff(x)

        return x.squeeze(1)
    
class RPS_MNet(nn.Module):
    """
        Model inspired by [Aoe at al., 10.1038/s41598-019-41500-x] integrated with bandpower.
    """

    def __init__(self, n_times):
        """
        Args:
            n_times (int):
                n_times dimension of the input data.
        """
        super(RPS_MNet, self).__init__()
        if n_times == 501:  # TODO automatic n_times
            self.n_times = 12
        elif n_times == 601:
            self.n_times = 18
        elif n_times == 701:
            self.n_times = 24
        else:
            raise ValueError(
                "Network can work only with n_times = 501, 601, 701 "
                "(epoch duration of 1., 1.2, 1.4 sec),"
                " got instead {}".format(n_times)
            )

        self.spatial = nn.Sequential(
            nn.Conv2d(1, 32, stride=(1, 1), kernel_size=[272, 64], bias=False),
            nn.ReLU(),
            # nn.BatchNorm2d(32),
            nn.Conv2d(32, 64, kernel_size=[1, 16], bias=False),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=[1, 2], stride=(1, 2)),
            # nn.BatchNorm2d(64),
        )


        self.temporal = nn.Sequential(nn.Conv2d(1, 32, kernel_size=[8, 8], bias=True),
                                      nn.ReLU(),
                                      # nn.BatchNorm2d(32),
                                      nn.Conv2d(32, 32, kernel_size=[8, 8], bias=True),
                                      nn.ReLU(),
                                      nn.MaxPool2d(kernel_size=[1, 3], stride=(1, 2)),
                                      # nn.BatchNorm2d(32),
                                      nn.Conv2d(32, 64, kernel_size=[6, 6], bias=True),
                                      nn.ReLU(),
                                      # nn.BatchNorm2d(64),
                                      nn.Conv2d(64, 64, kernel_size=[6, 6], bias=True),
                                      nn.ReLU(),
                                      # nn.BatchNorm2d(64),
                                      nn.MaxPool2d(kernel_size=[1, 2], stride=(1, 2)),
                                      nn.Conv2d(64, 128, kernel_size=[5, 5], bias=True),
                                      nn.ReLU(),
                                      nn.Dropout2d(p=0.3),
                                      # nn.BatchNorm2d(128),
                                      nn.Conv2d(128, 128, kernel_size=[5, 5], bias=True),
                                      nn.ReLU(),
                                      nn.Dropout2d(p=0.3),
                                      # nn.BatchNorm2d(128),
                                      nn.MaxPool2d(kernel_size=[1, 2], stride=(1, 2)),
                                      nn.Conv2d(128, 256, kernel_size=[4, 4], bias=True),
                                      nn.ReLU(),
                                      nn.Dropout2d(p=0.3),
                                      # nn.BatchNorm2d(256),
                                      nn.Conv2d(256, 256, kernel_size=[4, 4], bias=True),
                                      nn.ReLU(),
                                      nn.Dropout2d(p=0.3),
                                      # nn.BatchNorm2d(256),
                                      )

        self.attention = nn.Sequential(
            ChannelAttention([None, 256, 26, self.n_times]), SpatialAttention()
        )

        self.concatenate = Concatenate()

        # self.flatten = Flatten_MEG()

        self.ff = nn.Sequential(
            nn.Linear(256 * 26 * self.n_times + 204 * 6, 1024),
            nn.BatchNorm1d(num_features=1024),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(1024, 1024),
            nn.BatchNorm1d(num_features=1024),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(1024, 1),
        )

    def forward(self, x, pb):
        x = self.spatial(x)
        x = torch.transpose(x, 1, 2)
        x = self.temporal(x)
        x = self.attention(x)
        x = self.concatenate(x, pb)
        x = self.ff(x)

        return x.squeeze(1)