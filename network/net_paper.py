import torch.nn as nn
import math
import copy

# # First try. One feature extractor
# class Judge(nn.Module):
#
#     # Normalize each vector, and calculate cosine similarity, 1 means very similar
#     sim_dict = {'dist_sim': nn.PairwiseDistance(p=2), 'cos_sim': nn.CosineSimilarity()}
#
#     def __init__(self, image_channels, out_features, cmp='dist_sim', init_weight=True):
#         super(Judge, self).__init__()
#         assert cmp in ['dist_sim', 'cos_sim']
#         self.feature_extractor = nn.Sequential(
#             nn.Conv2d(image_channels, 64, 5),
#             nn.MaxPool2d(2, 2),
#             nn.Tanh(),
#             nn.Conv2d(64, 80, 5),
#             nn.Tanh(),
#             nn.Conv2d(80, 160, 5),
#             nn.MaxPool2d(2, 2),
#             nn.Tanh(),
#             nn.Conv2d(160, 256, 5),
#             nn.Tanh(),
#             nn.Conv2d(256, 512, 5),
#             nn.Tanh(),
#             nn.Conv2d(512, out_features, 1),
#             nn.Tanh()
#         )
#         self.sim = self.sim_dict[cmp]
#
#         if init_weight:
#             self._initialize_weights()
#
#     def forward(self, sample):
#         # print("Sample shape is:", sample.size())
#         p0_features = self.feature_extractor.forward(sample[:, 0])
#         p1_features = self.feature_extractor.forward(sample[:, 1])
#         p0_f = p0_features.clone()
#         p1_f = p1_features.clone()
#         pred = self.sim(p0_f.squeeze_(), p1_f.squeeze_())
#         # print("patch feature shape is", p0_features.size())
#         return pred
#
#     def _initialize_weights(self):
#         for m in self.modules():
#             print("Module", m)
#             if isinstance(m, nn.Conv2d):
#                 n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
#                 m.weight.data.normal_(0, math.sqrt(2. / n))
#                 if m.bias is not None:
#                     m.bias.data.zero_()
#             elif isinstance(m, nn.BatchNorm2d):
#                 m.weight.data.fill_(1)
#                 m.bias.data.zero_()
#             # elif isinstance(m, nn.Linear):
#             #     m.weight.data.normal_(0, 0.01)
#             #     m.bias.data.zero_()


# First try. One feature extractor
class Judge(nn.Module):

    # Normalize each vector, and calculate cosine similarity, 1 means very similar
    sim_dict = {'dist_sim': nn.PairwiseDistance(p=2), 'cos_sim': nn.CosineSimilarity()}

    def __init__(self, image_channels, out_features, cmp='dist_sim', init_weight=True, two_set_vars=False):
        super(Judge, self).__init__()
        assert cmp in ['dist_sim', 'cos_sim']
        self.feature_extractor_f = nn.Sequential(
            nn.Conv2d(image_channels, 64, 5),
            nn.MaxPool2d(2, 2),
            nn.Tanh(),
            nn.Conv2d(64, 80, 5),
            nn.Tanh(),
            nn.Conv2d(80, 160, 5),
            nn.MaxPool2d(2, 2),
            nn.Tanh(),
            nn.Conv2d(160, 256, 5),
            nn.Tanh(),
            nn.Conv2d(256, 512, 5),
            nn.Tanh(),
            nn.Conv2d(512, out_features, 1),
            nn.Tanh()
        )
        self.sim = self.sim_dict[cmp]
        self.two_set_vars = two_set_vars
        assert init_weight, "Only true option is supported."

        self._initialize_weights()
        if two_set_vars:
            self.feature_extractor_s = copy.deepcopy(self.feature_extractor_f)
        # else:
        #     self.feature_extractor_s = nn.Sequential(
        #         nn.Conv2d(image_channels, 64, 5),
        #         nn.MaxPool2d(2, 2),
        #         nn.Tanh(),
        #         nn.Conv2d(64, 80, 5),
        #         nn.Tanh(),
        #         nn.Conv2d(80, 160, 5),
        #         nn.MaxPool2d(2, 2),
        #         nn.Tanh(),
        #         nn.Conv2d(160, 256, 5),
        #         nn.Tanh(),
        #         nn.Conv2d(256, 512, 5),
        #         nn.Tanh(),
        #         nn.Conv2d(512, out_features, 1),
        #         nn.Tanh()
        #     )

    def forward(self, sample):
        # print("Sample shape is:", sample.size())
        if self.two_set_vars:
            p0_features = self.feature_extractor_f.forward(sample[:, 0])
            p1_features = self.feature_extractor_s.forward(sample[:, 1])
        else:
            p0_features = self.feature_extractor_f.forward(sample[:, 0])
            p1_features = self.feature_extractor_f.forward(sample[:, 1])
        p0_f = p0_features.clone()
        p1_f = p1_features.clone()
        # Squeeze the last dimension, [[f0],[f1]...[f511]]
        pred = self.sim(p0_f.squeeze_(), p1_f.squeeze_())
        # print("patch feature shape is", p0_features.size())
        return pred

    def _initialize_weights(self):
        for m in self.modules():
            print("Module", m)
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            # elif isinstance(m, nn.Linear):
            #     m.weight.data.normal_(0, 0.01)
            #     m.bias.data.zero_()
