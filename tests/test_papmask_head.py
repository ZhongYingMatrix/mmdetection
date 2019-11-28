import torch
from mmdet.models import builder

if __name__ == "__main__":
    # import pdb
    # pdb.set_trace()

    feats = (torch.randn((4, 256, int(96/2**i), int(160/2**i))).to('cuda:0') for i in range(5))
    mask_head=dict(
        type='PAPMask_Head',
        num_classes=81,
        in_channels=256,
        coefficient_channels=16,
        stacked_convs=4,
        feat_channels=256,
        strides=[8, 16, 32, 64, 128],
        use_dcn=True,
        loss_cls=dict(
            type='FocalLoss',
            use_sigmoid=True,
            gamma=2.0,
            alpha=0.25,
            loss_weight=1.0),
        #loss_bbox=dict(type='IoULoss', loss_weight=1.0),
        loss_polarcontour=dict(type='IoULoss'),
        loss_centerness=dict(
            type='CrossEntropyLoss', use_sigmoid=True, loss_weight=1.0),
        loss_mask=dict(
            type='CrossEntropyLoss', use_mask=True, loss_weight=1.0),
        conv_cfg=None,
        norm_cfg=dict(type='GN', num_groups=32, requires_grad=True),    
            )
    
    test_head = builder.build_head(mask_head)
    test_head.to('cuda:0')
    res = test_head(feats)
    import pdb
    pdb.set_trace()