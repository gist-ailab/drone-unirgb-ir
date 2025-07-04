# default: 4 stages
# _base_ = [
#     '../../../../_base_/datasets/flir(aligned)/flir_dual_LSJ_1024_1bs.py',
# ]
_base_ = [
    '/SSDb/jemo_maeng/src/Project/Drone24/detection/UniRGB-IR/detection/configs/_base_/datasets/flir(aligned)/flir_dual_LSJ_768_1bs.py',
]

dataset_type = 'DualSpectralDataset'
classes = ('car', 'person', 'bicycle')  # part of classes listed in DualSpectralDataset.Metainfo

custom_imports = dict(imports=['projects.ViTDet.vitdet'], allow_failed_imports=False)

backbone_norm_cfg = dict(type='LN', requires_grad=True)
norm_cfg = dict(type='LN2d', requires_grad=True)
image_size = (768, 768)
backend_args = None

batch_augments = [
    dict(type='BatchFixedSizePad', size=image_size, pad_mask=True)
]

train_pipeline = [
    dict(type='LoadAlignedImagesFromFile', backend_args=backend_args),
    dict(type='LoadAnnotations', with_bbox=True, with_mask=False), 
    dict(type='AlignedImagesRandomFlip', prob=0.5, direction='horizontal'),
    dict(
        type='AlignedImagesRandomResize',
        scale=image_size,
        ratio_range=(0.1, 2.0),
        keep_ratio=True),
    dict(
        type='AlignedImagesRandomCrop',
        crop_type='absolute_range',
        crop_size=image_size,
        recompute_bbox=True,
        allow_negative_crop=True),
    dict(type='FilterAnnotations', min_gt_bbox_wh=(1e-2, 1e-2)),
    dict(type='AlignedImagesPad', size=image_size, pad_val=dict(img=(114, 114, 114))),
    dict(type='PackAlignedImagesDetInputs')
]

test_pipeline = [
    dict(type='LoadAlignedImagesFromFile', backend_args=backend_args),
    dict(type='AlignedImagesResize', scale=image_size, keep_ratio=False),
    dict(type='LoadAnnotations', with_bbox=True, with_mask=False),
    dict(
        type='PackAlignedImagesDetInputs',
        meta_keys=('img_id', 'img_path', 'img_ir_path', 'ori_shape', 'img_shape',
                   'scale_factor'))
]

# model settings, setting mostly inherits from cascade-rcnn_r50_fpn
model = dict(
    type='TwoStreamTwoStageDetectorFuseBeforeFPN',
    data_preprocessor=dict(
        type='DualSteramDetDataPreprocessor',
        mean=[159.8808906080302, 162.22057018543336, 160.28301196773916],
        mean_ir=[136.63746562356317, 136.63746562356317, 136.63746562356317],  # IR as 3-channel  input
        std=[56.96897676312916, 59.57937492901139, 63.11906486423505],
        std_ir=[64.97730349740912, 64.97730349740912, 64.97730349740912],  
        bgr_to_rgb=True,
        pad_size_divisor=32,
        batch_augments=batch_augments),  # NOTE: batch augmentation
    backbone=dict(
        # _delete_=True,
        type='ViTRGBTv15',
        method=None,
        img_size=1024,
        stage_ranges=[[0, 2], [3, 5], [6, 8], [9, 11]],
        conv_inplane=64,
        n_points=4,
        deform_num_heads=12,
        # cffn_ratio=0.25,
        deform_ratio=0.5,
        patch_size=16,
        in_chans=3, 
        embed_dim=768,
        depth=12,
        num_heads=12,
        mlp_ratio=4,
        qkv_bias=True,
        # layer_scale_init_values=None,  # no layer_scale for MAE pertrained weights
        # scale_factor=12,
        # drop_path_rate=0.1,
        drop_path_rate=0.3,
        # deform_norm_layer=partial(nn.LayerNorm, eps=1e-6),
        norm_cfg=backbone_norm_cfg,
        window_size=14,  # windowed attention
        pretrain_img_size=224,
        pretrain_use_cls_token=True,
        window_block_indexes=[
            0,
            1,
            3,
            4,
            6,
            7,
            9,
            10,
        ],
        # use_rel_pos=False,
        use_rel_pos=True,
        init_cfg=dict(
            type='Pretrained', checkpoint="/SSDb/jemo_maeng/src/Project/Drone24/detection/UniRGB-IR/detection/checkpoint/VitDet/vitb_coco_IN1k_mae_coco_cascade-mask-rcnn_224x224_withClsToken_noRel.pth")
    ), 
    neck=dict(  # ViTDet specify this SimpleFPN
        type='SimpleFPN',
        backbone_channel=768,
        in_channels=[192, 384, 768, 768],
        out_channels=256,
        num_outs=5,
        norm_cfg=norm_cfg),
    rpn_head=dict(
        type='RPNHead',
        num_convs=2,  # modification for ViTDet
        in_channels=256,
        feat_channels=256,
        anchor_generator=dict(
            type='AnchorGenerator',
            scales=[8],
            ratios=[0.5, 1.0, 2.0],
            strides=[4, 8, 16, 32, 64]),
        bbox_coder=dict(
            type='DeltaXYWHBBoxCoder',
            target_means=[.0, .0, .0, .0],
            target_stds=[1.0, 1.0, 1.0, 1.0]),
        loss_cls=dict(
            type='CrossEntropyLoss', use_sigmoid=True, loss_weight=1.0),
        # loss_bbox=dict(type='L1Loss', loss_weight=1.0)),
        loss_bbox=dict(type='SmoothL1Loss', beta=1.0 / 9.0, loss_weight=1.0)),
    roi_head=dict(
        type='CascadeRoIHead',
        num_stages=3,
        stage_loss_weights=[1, 0.5, 0.25],  # weight for loss from 3 stage
        bbox_roi_extractor=dict(
            type='SingleRoIExtractor',
            # output fixed featmap size: (7x7)
            roi_layer=dict(type='RoIAlign', output_size=7, sampling_ratio=0),
            out_channels=256,
            featmap_strides=[4, 8, 16, 32]),
        bbox_head=[
            dict(
                # with 4 shared convolution layer, plus a fully connected layer
                type='Shared4Conv1FCBBoxHead',  # modified
                in_channels=256,
                conv_out_channels=256,  # added
                norm_cfg=norm_cfg,  # added
                fc_out_channels=1024,
                roi_feat_size=7,
                num_classes=3,
                bbox_coder=dict(
                    type='DeltaXYWHBBoxCoder',
                    target_means=[0., 0., 0., 0.],
                    target_stds=[0.1, 0.1, 0.2, 0.2]),
                reg_class_agnostic=True,
                loss_cls=dict(
                    type='CrossEntropyLoss',
                    use_sigmoid=False,
                    loss_weight=1.0),
                loss_bbox=dict(type='SmoothL1Loss', beta=1.0,
                               loss_weight=1.0)),
            dict(
                type='Shared4Conv1FCBBoxHead',  # modified
                in_channels=256,
                conv_out_channels=256,  # added
                norm_cfg=norm_cfg,  # added
                fc_out_channels=1024,
                roi_feat_size=7,
                num_classes=3,
                bbox_coder=dict(
                    type='DeltaXYWHBBoxCoder',
                    target_means=[0., 0., 0., 0.],
                    target_stds=[0.05, 0.05, 0.1, 0.1]),
                reg_class_agnostic=True,
                loss_cls=dict(
                    type='CrossEntropyLoss',
                    use_sigmoid=False,
                    loss_weight=1.0),
                loss_bbox=dict(type='SmoothL1Loss', beta=1.0,
                               loss_weight=1.0)),
            dict(
                type='Shared4Conv1FCBBoxHead',  # modified
                in_channels=256,
                conv_out_channels=256,  # added
                norm_cfg=norm_cfg,  # added
                fc_out_channels=1024,
                roi_feat_size=7,
                num_classes=3,
                bbox_coder=dict(
                    type='DeltaXYWHBBoxCoder',
                    target_means=[0., 0., 0., 0.],
                    target_stds=[0.033, 0.033, 0.067, 0.067]),
                reg_class_agnostic=True,
                loss_cls=dict(
                    type='CrossEntropyLoss',
                    use_sigmoid=False,
                    loss_weight=1.0),
                loss_bbox=dict(type='SmoothL1Loss', beta=1.0, loss_weight=1.0))
        ]),
    # model training and testing settings
    train_cfg=dict(
        rpn=dict(
            assigner=dict(
                type='MaxIoUAssigner',
                pos_iou_thr=0.7,
                neg_iou_thr=0.3,
                min_pos_iou=0.3,
                match_low_quality=True,
                ignore_iof_thr=-1),
            sampler=dict(
                type='RandomSampler',
                num=256,
                pos_fraction=0.5,
                neg_pos_ub=-1,
                add_gt_as_proposals=False),
            allowed_border=0,
            pos_weight=-1,
            debug=False),
        rpn_proposal=dict(
            nms_pre=2000,
            max_per_img=2000,
            nms=dict(type='nms', iou_threshold=0.7),
            min_bbox_size=0),
        rcnn=[
            dict(
                assigner=dict(
                    type='MaxIoUAssigner',
                    pos_iou_thr=0.5,  # matcher
                    neg_iou_thr=0.5,
                    min_pos_iou=0.5,
                    match_low_quality=False,
                    ignore_iof_thr=-1),
                sampler=dict(
                    type='RandomSampler',
                    num=512,
                    pos_fraction=0.25,
                    neg_pos_ub=-1,
                    add_gt_as_proposals=True),
                pos_weight=-1,
                debug=False),
            dict(
                assigner=dict(
                    type='MaxIoUAssigner',
                    pos_iou_thr=0.6,
                    neg_iou_thr=0.6,
                    min_pos_iou=0.6,
                    match_low_quality=False,
                    ignore_iof_thr=-1),
                sampler=dict(
                    type='RandomSampler',
                    num=512,
                    pos_fraction=0.25,
                    neg_pos_ub=-1,
                    add_gt_as_proposals=True),
                pos_weight=-1,
                debug=False),
            dict(
                assigner=dict(
                    type='MaxIoUAssigner',
                    pos_iou_thr=0.7,
                    neg_iou_thr=0.7,
                    min_pos_iou=0.7,
                    match_low_quality=False,
                    ignore_iof_thr=-1),
                sampler=dict(
                    type='RandomSampler',
                    num=512,
                    pos_fraction=0.25,
                    neg_pos_ub=-1,
                    add_gt_as_proposals=True),
                pos_weight=-1,
                debug=False)
        ]),
    test_cfg=dict(
        rpn=dict(
            nms_pre=2000,  # before: 1000
            max_per_img=2000,  # before: 1000
            nms=dict(type='nms', iou_threshold=0.8),  # before: 0.7
            min_bbox_size=0),
        rcnn=dict(
            score_thr=0.05,
            nms=dict(type='nms', iou_threshold=0.5),
            max_per_img=2000)))  # before: 100

data_root = '/SSDb/jemo_maeng/dset/FLIR_aligned_unirgbir/'  # with separator '/'
train_dataloader = dict(
    batch_size=1,  
    num_workers=4,
    persistent_workers=True,
    # sampler=dict(type='DefaultSampler', shuffle=True),
    sampler=dict(type='InfiniteSampler', shuffle=True), 
    dataset=dict(
        type=dataset_type,
        data_root=data_root,
        metainfo=dict(classes=classes),
        ann_file='Annotation_train_updated.json',
        data_prefix=dict(img='train/'),
        filter_cfg=dict(filter_empty_gt=True, min_size=32),
        pipeline=train_pipeline))

val_dataloader = dict(
    batch_size=1,
    num_workers=2,
    persistent_workers=True,
    drop_last=False,
    sampler=dict(type='DefaultSampler', shuffle=False),
    dataset=dict(
        type=dataset_type,
        data_root=data_root,
        metainfo=dict(classes=classes),
        ann_file='Annotation_test_updated.json',
        data_prefix=dict(img='test/'),
        test_mode=True,
        pipeline=test_pipeline))
test_dataloader = val_dataloader


val_evaluator = dict(  
    type='CocoMetric',
    ann_file=data_root + 'Annotation_test_updated.json',
    metric=['bbox'], 
    format_only=False
)
test_evaluator = val_evaluator

optim_wrapper = dict(
    type='AmpOptimWrapper',
    constructor='LayerDecayOptimizerConstructor',
    paramwise_cfg={
        'decay_rate': 0.7,
        'decay_type': 'layer_wise',
        'num_layers': 12,
    },
    optimizer=dict(
        type='AdamW',
        lr=0.0001,  # basic lr
        # lr=0.00001,  # basic lr
        betas=(0.9, 0.999),
        weight_decay=0.1,
    ),
    # clip_grad=dict(max_norm=3)  # added in ver.6
)


# iters = 100ep * 4129 imges / (8gpu * 1)
# FLIR: train - 4129, val - 1013
# max_iters = 68817 # HACK: modified based on epochs and minibatch size
max_iters = 51613
interval = 500
dynamic_intervals = [(max_iters // interval * interval + 1, max_iters)]
param_scheduler = [
    dict(
        type='LinearLR', start_factor=0.001, by_epoch=False, begin=0, end=250),
    dict(
        type='MultiStepLR',
        begin=0,
        end=max_iters,
        by_epoch=False,
        # 88%, 96% 作为 milestones
        milestones=[45419, 49548],
        gamma=0.1)
]

train_cfg = dict(
    type='IterBasedTrainLoop',
    max_iters=max_iters,
    val_interval=interval,
    dynamic_intervals=dynamic_intervals)
val_cfg = dict(type='ValLoop')
test_cfg = dict(type='TestLoop')

default_hooks = dict(
    logger=dict(type='LoggerHook', interval=50),
    checkpoint=dict(
        type='CheckpointHook',
        by_epoch=False,
        save_last=True,
        save_best='coco/bbox_mAP',
        # rule='less',
        interval=interval,
        max_keep_ckpts=2,
    )
)

custom_hooks = [dict(type='Fp16CompresssionHook')]

log_processor = dict(type='LogProcessor', window_size=50, by_epoch=False)

auto_scale_lr = dict(base_batch_size=64, enable=True)
