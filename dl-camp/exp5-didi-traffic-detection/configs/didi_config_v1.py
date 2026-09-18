_base_ = [
    '../_base_/models/faster-rcnn_r50_fpn.py',
    '../_base_/datasets/didi_detection.py',
    '../_base_/schedules/schedule_1x.py', '../_base_/default_runtime.py'
]

train_cfg = dict(type='EpochBasedTrainLoop', max_epochs=24, val_interval=1)

optim_wrapper = dict(
    _delete_=True,
    type='OptimWrapper',
    optimizer=dict(
        type='AdamW',
        lr=0.0001,
        betas=(0.9, 0.999),
        weight_decay=0.05,
        eps=1e-8,
    ),
    clip_grad=dict(max_norm=35, norm_type=2)
)

param_scheduler = [
    dict(
        type='LinearLR',
        start_factor=0.001,
        by_epoch=False,
        begin=0,
        end=1000
    ),
    dict(
        type='CosineAnnealingLR',
        T_max=24,
        eta_min=1e-8,
        by_epoch=True,
        begin=1,
        end=24
    )
]

val_evaluator = dict(
    type='CocoMetric',
    metric='bbox',
    ann_file='data/didi/dataset_release/val.json',
    outfile_prefix='work_dirs/detailed_eval/epoch',
    classwise=True,
    proposal_nums=(100, 300, 1000),
    iou_thrs=[0.50, 0.55, 0.60, 0.65, 0.70, 0.75, 0.80, 0.85, 0.90, 0.95],
    metric_items=['mAP', 'mAP_50', 'mAP_75', 'mAP_s', 'mAP_m', 'mAP_l'],
    format_only=False,
)

test_evaluator = dict(
    type='CocoMetric',
    metric='bbox',
    ann_file='data/didi/dataset_release/test.json',
    outfile_prefix='work_dirs/detailed_eval/test',
    classwise=True,
    proposal_nums=(100, 300, 1000),
    iou_thrs=[0.50, 0.55, 0.60, 0.65, 0.70, 0.75, 0.80, 0.85, 0.90, 0.95],
    metric_items=['mAP', 'mAP_50', 'mAP_75', 'mAP_s', 'mAP_m', 'mAP_l'],
    format_only=False,
)
