import torch.nn as nn


class CNN_Feature(nn.Module):

    def __init__(self, in_channels, out_channels):
        """
        Instantiate layers with given params. Same architecture as paper suggested.
        :param d_in:
        :param d_out:
        """
        super(CNN_Feature,self).__init__()
        self.act  = nn.Tanh()
        self.pool = nn.MaxPool2d(2, 2)

        self.conv1 = nn.Conv2d(in_channels, 64, 5)
        self.conv2 = nn.Conv2d(64, 80, 5)
        self.conv3 = nn.Conv2d(80, 160, 5)
        self.conv4 = nn.Conv2d(160, 256, 5)
        self.conv5 = nn.Conv2d(256, 512, 5)
        self.conv6 = nn.Conv2d(512, out_channels, 1)

    def forward(self, x):
        conv1 = self.conv1(x)
        conv1 = self.pool(conv1)
        conv1 = self.act(conv1)

        conv2 = self.act(self.conv2(conv1))

        conv3 = self.conv3(conv2)
        conv3 = self.pool(conv3)
        conv3 = self.act(conv3)

        conv4 = self.act(self.conv4(conv3))

        conv5 = self.act(self.conv5(conv4))

        conv6 = self.act(self.conv6(conv5))

        return conv6


class Judge(nn.Module):

    # Normalize each vector, and calculate cosine similarity, 1 means very similar
    sim_dict = {'dist_sim': nn.PairwiseDistance(p=2), 'cos_sim': nn.CosineSimilarity()}

    def __init__(self, image_channels, out_features, cmp='dist_sim'):
        super(Judge, self).__init__()

        assert cmp in ['dist_sim', 'cos_sim']
        self.feature_extractor = CNN_Feature(image_channels, out_features)
        self.sim = self.sim_dict[cmp]

    def forward(self, sample):
        # print("Sample shape is:", sample.size())
        p0_features = self.feature_extractor.forward(sample[:, 0])
        p1_features = self.feature_extractor.forward(sample[:, 1])
        p0_f = p0_features.clone()
        p1_f = p1_features.clone()
        pred = self.sim(p0_f.squeeze_(), p1_f.squeeze_())
        # print("patch feature shape is", p0_features.size())
        return pred
