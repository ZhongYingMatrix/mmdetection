# fp 16
#fp16 = dict(loss_scale=512.)

# model settings
norm_cfg = dict(type='SyncBN', requires_grad=True)

model = dict(
    type='HybridTaskCascade',
    num_stages=3,
    pretrained='torchvision://resnet101',
    interleaved=True,
    mask_info_flow=True,
    backbone=dict(
        type='ResNet',
        depth=101,
        num_stages=4,
        out_indices=(0, 1, 2, 3),
        frozen_stages=1,
        style='pytorch',
        dcn=dict(
            modulated=False, deformable_groups=1, fallback_on_stride =False),
        stage_with_dcn=(False, False, False, True),
        norm_eval=False,
        norm_cfg = norm_cfg
        ),
    neck=dict(
        type='FPN',
        in_channels=[256, 512, 1024, 2048],
        out_channels=256,
        num_outs=5),
    rpn_head=dict(
        #type='RPNHead',
        type='GARPNHead',

        in_channels=256,
        feat_channels=256,
        #anchor_scales=[8],
        #anchor_ratios=[0.5, 1.0, 2.0],
        octave_base_scale=8,
        scales_per_octave=3,
        octave_ratios=[0.5, 1.0, 2.0],

        anchor_strides=[4, 8, 16, 32, 64],
        #target_means=[.0, .0, .0, .0],
        #target_stds=[1.0, 1.0, 1.0, 1.0],
        anchor_base_sizes=None,
        anchoring_means=[.0, .0, .0, .0],
        anchoring_stds=[0.07, 0.07, 0.14, 0.14],
        target_means=(.0, .0, .0, .0),
        target_stds=[0.07, 0.07, 0.11, 0.11],
        loc_filter_thr=0.01,
            loss_loc=dict(
            type='FocalLoss',
            use_sigmoid=True,
            gamma=2.0,
            alpha=0.25,
            loss_weight=1.0),
        loss_shape=dict(type='BoundedIoULoss', beta=0.2, loss_weight=1.0),

        loss_cls=dict(
            type='CrossEntropyLoss', use_sigmoid=True, loss_weight=1.0),
        #loss_bbox=dict(type='SmoothL1Loss', beta=1.0 / 9.0, loss_weight=1.0)),
        loss_bbox=dict(type='SmoothL1Loss', beta=1.0, loss_weight=1.0)),
    
    bbox_roi_extractor=dict(
        type='SingleRoIExtractor',
        roi_layer=dict(type='RoIAlign', out_size=7, sample_num=2),
        out_channels=256,
        featmap_strides=[4, 8, 16, 32]),
    bbox_head=[
        dict(
            type='SharedFCBBoxHead',
            num_fcs=2,
            in_channels=256,
            fc_out_channels=1024,
            roi_feat_size=7,
            num_classes=81,
            target_means=[0., 0., 0., 0.],
            #target_stds=[0.1, 0.1, 0.2, 0.2],
            target_stds=[0.05, 0.05, 0.1, 0.1],
            
            reg_class_agnostic=True,
            loss_cls=dict(
                type='CrossEntropyLoss', use_sigmoid=False, loss_weight=1.0),
            loss_bbox=dict(type='SmoothL1Loss', beta=1.0, loss_weight=1.0)),
        dict(
            type='SharedFCBBoxHead',
            num_fcs=2,
            in_channels=256,
            fc_out_channels=1024,
            roi_feat_size=7,
            num_classes=81,
            target_means=[0., 0., 0., 0.],
            #target_stds=[0.05, 0.05, 0.1, 0.1],
            target_stds=[0.033, 0.033, 0.067, 0.067],
            
            reg_class_agnostic=True,
            loss_cls=dict(
                type='CrossEntropyLoss', use_sigmoid=False, loss_weight=1.0),
            loss_bbox=dict(type='SmoothL1Loss', beta=1.0, loss_weight=1.0)),
        dict(
            type='SharedFCBBoxHead',
            num_fcs=2,
            in_channels=256,
            fc_out_channels=1024,
            roi_feat_size=7,
            num_classes=81,
            target_means=[0., 0., 0., 0.],
            #target_stds=[0.033, 0.033, 0.067, 0.067],
            target_stds=[0.025, 0.025, 0.05, 0.05],
            
            reg_class_agnostic=True,
            loss_cls=dict(
                type='CrossEntropyLoss', use_sigmoid=False, loss_weight=1.0),
            loss_bbox=dict(type='SmoothL1Loss', beta=1.0, loss_weight=1.0))
    ],
    mask_roi_extractor=dict(
        type='SingleRoIExtractor',
        roi_layer=dict(type='RoIAlign', out_size=14, sample_num=2),
        out_channels=256,
        featmap_strides=[4, 8, 16, 32]),
    mask_head=dict(
        type='HTCMaskHead',
        num_convs=4,
        in_channels=256,
        conv_out_channels=256,
        num_classes=81,
        loss_mask=dict(
            type='CrossEntropyLoss', use_mask=True, loss_weight=1.0)))#,

# model training and testing settings
train_cfg = dict(
    rpn=dict(
        # Guided Anchoring
        ga_assigner=dict(
            type='ApproxMaxIoUAssigner',
            pos_iou_thr=0.7,
            neg_iou_thr=0.3,
            min_pos_iou=0.3,
            ignore_iof_thr=-1),
        ga_sampler=dict(
            type='RandomSampler',
            num=256,
            pos_fraction=0.5,
            neg_pos_ub=-1,
            add_gt_as_proposals=False),

        assigner=dict(
            type='MaxIoUAssigner',
            pos_iou_thr=0.7,
            neg_iou_thr=0.3,
            min_pos_iou=0.3,
            ignore_iof_thr=-1),
        sampler=dict(
            type='RandomSampler',
            num=256,
            pos_fraction=0.5,
            neg_pos_ub=-1,
            add_gt_as_proposals=False),
        #allowed_border=0,
        allowed_border=-1,
        
        pos_weight=-1,
        # Guided Anchoring
        center_ratio=0.2,
        ignore_ratio=0.5,

        debug=False),
    rpn_proposal=dict(
        nms_across_levels=False,
        nms_pre=2000,
        nms_post=2000,
        #max_num=2000,
        max_num=300,
        
        nms_thr=0.7,
        min_bbox_size=0),
    rcnn=[
        dict(
            assigner=dict(
                type='MaxIoUAssigner',
                # pos_iou_thr=0.5,
                # neg_iou_thr=0.5,
                # min_pos_iou=0.5,
                pos_iou_thr=0.6,
                neg_iou_thr=0.6,
                min_pos_iou=0.6,

                ignore_iof_thr=-1),
            sampler=dict(
                type='RandomSampler',
                #num=512,
                num=256,

                pos_fraction=0.25,
                neg_pos_ub=-1,
                add_gt_as_proposals=True),
            mask_size=28,
            pos_weight=-1,
            debug=False),
        dict(
            assigner=dict(
                type='MaxIoUAssigner',
                # pos_iou_thr=0.6,
                # neg_iou_thr=0.6,
                # min_pos_iou=0.6,
                pos_iou_thr=0.7,
                neg_iou_thr=0.7,
                min_pos_iou=0.7,

                ignore_iof_thr=-1),
            sampler=dict(
                type='RandomSampler',
                #num=512,
                num=256,

                pos_fraction=0.25,
                neg_pos_ub=-1,
                add_gt_as_proposals=True),
            mask_size=28,
            pos_weight=-1,
            debug=False),
        dict(
            assigner=dict(
                type='MaxIoUAssigner',
                # pos_iou_thr=0.7,
                # neg_iou_thr=0.7,
                # min_pos_iou=0.7,
                pos_iou_thr=0.8,
                neg_iou_thr=0.8,
                min_pos_iou=0.8,

                ignore_iof_thr=-1),
            sampler=dict(
                type='RandomSampler',
                #num=512,
                num=256,

                pos_fraction=0.25,
                neg_pos_ub=-1,
                add_gt_as_proposals=True),
            mask_size=28,
            pos_weight=-1,
            debug=False)
    ],
    stage_loss_weights=[1, 0.5, 0.25])
test_cfg = dict(
    rpn=dict(
        nms_across_levels=False,
        nms_pre=1000,
        nms_post=1000,
        #max_num=1000,
        max_num=300,

        nms_thr=0.7,
        min_bbox_size=0),
    rcnn=dict(
        score_thr=0.05,
        # in Guided Anchoring score_thr is 1e-3

        # nms=dict(type='soft_nms', iou_thr=0.5,min_score=0.05),
        nms=dict(type='nms', iou_thr=0.5),
        max_per_img=100,
        mask_thr_binary=0.5),
    keep_all_stages=False)
# dataset settings
dataset_type = 'CocoDataset'
data_root = "/home/zhongying/dataset/"
img_norm_cfg = dict(
    mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375], to_rgb=True)
train_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='LoadAnnotations', with_bbox=True, with_mask=True),
    dict(type='Resize', img_scale=(1024, 1024), keep_ratio=True),
    dict(type='RandomFlip', flip_ratio=0.5),
    dict(type='Normalize', **img_norm_cfg),
    dict(type='Pad', size_divisor=32),
    dict(type='DefaultFormatBundle'),
    dict(type='Collect', keys=['img', 'gt_bboxes', 'gt_labels', 'gt_masks']),
]
test_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(
        type='MultiScaleFlipAug',
        img_scale= (1024, 1024),#[ (1024, 1024) ,(2048,2048)],
        flip=False,
        transforms=[
            dict(type='Resize', keep_ratio=True),
            dict(type='RandomFlip', flip_ratio=0.5),
            dict(type='Normalize', **img_norm_cfg),
            dict(type='Pad', size_divisor=32),
            dict(type='ImageToTensor', keys=['img']),
            dict(type='Collect', keys=['img']),
        ])
]
data = dict(
    imgs_per_gpu=1,
    workers_per_gpu=1,
    train=[
        dict(
        type=dataset_type,
        ann_file=data_root + 'WalkTrainData/zy/new_set_instance.json',
        img_prefix=data_root ,
        pipeline=train_pipeline),
        dict(
        type=dataset_type,
        ann_file=data_root + 'WalkTrainData/total_instance.json',
        img_prefix=data_root ,
        pipeline=train_pipeline)
    ],
    val=dict(
        type=dataset_type,
        ann_file=data_root + "WalkValidationData/correct_validation_instance.json",
        img_prefix=data_root ,
        pipeline=test_pipeline),
    test=dict(
        type=dataset_type,
        ann_file=data_root + 'WalkTrainData/xihu_instance.json',
        img_prefix=data_root ,
        pipeline=test_pipeline))
# optimizer
optimizer = dict(type='SGD', lr=0.005, momentum=0.9, weight_decay=0.0001)
optimizer_config = dict(grad_clip=dict(max_norm=35, norm_type=2))
# learning policy
lr_config = dict(
    policy='step',
    warmup='linear',
    warmup_iters=500,
    warmup_ratio=1.0 / 3,
    step=[32, 38])
checkpoint_config = dict(interval=1)
# yapf:disable
log_config = dict(
    interval=50,
    hooks=[
        dict(type='TextLoggerHook'),
        # dict(type='TensorboardLoggerHook')
    ])
# yapf:enable
#evaluation = dict(interval=1)
# runtime settings
total_epochs = 40
dist_params = dict(backend='nccl')
log_level = 'INFO'
work_dir = '/home/zhongying/other/walkgis/mmdetection/zy/htc_new_set/2019_10_28'
load_from = None
resume_from = None
workflow = [('train', 1)]
