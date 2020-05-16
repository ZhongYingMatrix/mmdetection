_base_ = './otss_r50_caffe_fpn_gn-head_4x4_CIoU_soft-label_1x_4gpu_coco.py'
# model settings
mode = dict(
    pretrained='open-mmlab://resnet101_caffe_bgr',
    backbone=dict(
        depth=101,
        dcn=dict(type='DCN', deformable_groups=1, fallback_on_stride=False),
        stage_with_dcn=(False, True, True, True)),
    bbox_head=dict(use_dcn_in_tower=True)
    )
img_norm_cfg = dict(
    mean=[103.530, 116.280, 123.675], std=[1.0, 1.0, 1.0], to_rgb=False)
train_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='LoadAnnotations', with_bbox=True),
    dict(
        type='Resize',
        img_scale=[(1333, 640), (1333, 800)],
        multiscale_mode='value',
        keep_ratio=True),
    dict(type='RandomFlip', flip_ratio=0.5),
    dict(type='Normalize', **img_norm_cfg),
    dict(type='Pad', size_divisor=32),
    dict(type='DefaultFormatBundle'),
    dict(type='Collect', keys=['img', 'gt_bboxes', 'gt_labels']),
]
# learning policy
optimizer = dict(lr=0.01)
lr_config = dict(step=[16, 22])
total_epochs = 24
