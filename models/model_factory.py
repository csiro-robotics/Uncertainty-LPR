# Author: Jacek Komorowski
# Warsaw University of Technology

import models.minkloc as minkloc
from torchpack.utils.config import configs 
from models.PointNetVlad import PointNetVlad 
from models.ppt_transformer.pptnet import Network as PPT

def model_factory():
    in_channels = 1

    if 'MinkFPN' in configs.model.name:
        model = minkloc.MinkLoc(configs.model.name, in_channels=in_channels,
                                feature_size=configs.model.feature_size,
                                output_dim=configs.model.output_dim, planes=configs.model.planes,
                                layers=configs.model.layers, num_top_down=configs.model.num_top_down,
                                conv0_kernel_size=configs.model.conv0_kernel_size)
    elif 'PointNetVlad' in configs.model.name:
        model = PointNetVlad(
            num_points = configs.data.num_points,
            global_feat = True,
            feature_transform = True,
            max_pool = False,
            output_dim = configs.model.output_dim
        
        )
    elif 'PPT' in configs.model.name:
        model = PPT()
    else:
        raise NotImplementedError('Model not implemented: {}'.format(configs.model.name))

    return model
