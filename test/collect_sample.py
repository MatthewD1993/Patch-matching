from network.utils import  KITTIPatchesDataset, SintelPatchesDataset, ChairsPatchesDataset

# kitti = KITTIPatchesDataset(56, 160)
# kitti.newData(100)
# kitti.save_data('./data/kitti.npy')
#
# sintel = SintelPatchesDataset(56, 833)
# sintel.newData(100)
# sintel.save_data('./data/sintel.npy')

chairs = ChairsPatchesDataset(56, 1000)
chairs.one_fetch = 100
chairs.newData(100)
chairs.save_data('./data/chair.npy')
