# Author: Jacek Komorowski
# Warsaw University of Technology

import models.minkloc as minkloc
from torchpack.utils.config import configs 
from models.PointNetVlad import PointNetVlad 
from models.transloc3d.model import TransLoc3D

def model_factory(uncertainty_method):
    in_channels = 1

    if 'MinkFPNstun' in configs.model.name and uncertainty_method=="stun_student":
        model = minkloc.MinkLocStudent(configs.model.name, in_channels=in_channels,
                                feature_size=configs.model.feature_size,
                                output_dim=configs.model.output_dim, planes=configs.model.planes,
                                layers=configs.model.layers, num_top_down=configs.model.num_top_down,
                                conv0_kernel_size=configs.model.conv0_kernel_size)
    elif 'MinkFPNstun' in configs.model.name and uncertainty_method=="stun_teacher":
        model = minkloc.MinkLoc(configs.model.name, in_channels=in_channels,
                                feature_size=configs.model.feature_size,
                                output_dim=configs.model.output_dim, planes=configs.model.planes,
                                layers=configs.model.layers, num_top_down=configs.model.num_top_down,
                                conv0_kernel_size=configs.model.conv0_kernel_size)
    elif 'MinkFPNpfe' in configs.model.name:
        model = minkloc.MinkLocPFE(configs.model.name, in_channels=in_channels,
                                feature_size=configs.model.feature_size,
                                output_dim=configs.model.output_dim, planes=configs.model.planes,
                                layers=configs.model.layers, num_top_down=configs.model.num_top_down,
                                conv0_kernel_size=configs.model.conv0_kernel_size)
    elif 'MinkFPNDropout' in configs.model.name:
        model = minkloc.MinkLocDropout(configs.model.name, in_channels=in_channels,
                                feature_size=configs.model.feature_size,
                                output_dim=configs.model.output_dim, planes=configs.model.planes,
                                layers=configs.model.layers, num_top_down=configs.model.num_top_down,
                                conv0_kernel_size=configs.model.conv0_kernel_size, dropout_rate=configs.model.dropout_rate, dropout_location=configs.model.dropout_location)
    elif 'MinkFPN' in configs.model.name:
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
    elif 'TransLoc3D' in configs.model.name:
        model = TransLoc3D()
    else:
        raise NotImplementedError('Model not implemented: {}'.format(configs.model.name))

    return model
