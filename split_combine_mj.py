import torch
import numpy as np


class SplitComb():
    def __init__(self, side_len=[80, 192, 304], margin=60):
        """
        :param side_len: list of input shape, default=[80,192,304] \
        :param margin: sliding stride, default=[60,60,60]
        """
        self.side_len = side_len
        self.margin = margin

    def split_id(self, data):
        """
        :param data: target data to be splitted into sub-volumes, shape = (D, H, W) \
        :return: output list of coordinates for the cropped sub-volumes, start-to-end
        """
        side_len = self.side_len
        margin = self.margin
        
        if type(margin) is not list:
            margin = [margin, margin, margin]
            
        splits = []
        z, h, w = data.shape
        
        nz = int(np.ceil(float(z-margin[0])/side_len[0]))
        nh = int(np.ceil(float(h-margin[1])/side_len[1]))
        nw = int(np.ceil(float(w-margin[2])/side_len[2]))
        
        assert (nz*side_len[0]+margin[0]-z >= 0)
        assert (nh*side_len[1]+margin[1]-h >= 0)
        assert (nw*side_len[2]+margin[2]-w >= 0)
        
        nzhw = [nz, nh, nw]
        self.nzhw = nzhw
        
        pad = [[0, nz * side_len[0] + margin[0] - z],
                [0, nh * side_len[1] + margin[1] - h],
                [0, nw * side_len[2] + margin[2] - w]]
        orgshape = [z, h, w]

        idx = 0
        for iz in range(nz+1):
            for ih in range(nh+1):
                for iw in range(nw+1):
                    sz = iz * side_len[0]  # start
                    ez = iz * side_len[0] + margin[0]  # end
                    sh = ih * side_len[1]
                    eh = ih * side_len[1] + margin[1]
                    sw = iw * side_len[2]
                    ew = iw * side_len[2] + margin[2]
                    if ez > z:
                        sz = z - margin[0]
                        ez = z
                    if eh > h:
                        sh = h - margin[1]
                        eh = h
                    if ew > w:
                        sw = w - margin[2]
                        ew = w
                    idcs = [[sz, ez],[sh, eh], [sw, ew],idx]
                    splits.append(idcs)
                    idx += 1
        splits = np.array(splits,dtype=object)
        # split size
        return splits, nzhw, orgshape
