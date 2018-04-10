require "nn"

s1 = 2
s2 = 2
s3 = 2
s4 = 2



net1 = nn.Sequential()
net1:add(nn.SpatialConvolution(3, 32, 7, 7))
net1:add(nn.SpatialMaxPooling(s1, s1, s1, s1))
net1:add(nn.TanH())
net1:add(nn.SpatialConvolution(32, 64, 6, 6))
net1:add(nn.SpatialMaxPooling(s2, s2, s2, s2))
net1:add(nn.TanH())
net1:add(nn.SpatialConvolution(64, outChans, 5, 5))
net1:add(nn.SpatialMaxPooling(s3, s3, s3, s3))
net1:add(nn.TanH())