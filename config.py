default_scope = 'mmocr'
env_cfg = dict(
    cudnn_benchmark=True,
    mp_cfg=dict(mp_start_method='fork', opencv_num_threads=0),
    dist_cfg=dict(backend='nccl'))
randomness = dict(seed=None)
default_hooks = dict(
    timer=dict(type='IterTimerHook'),
    logger=dict(type='LoggerHook', interval=100),
    param_scheduler=dict(type='ParamSchedulerHook'),
    checkpoint=dict(type='CheckpointHook', interval=1),
    sampler_seed=dict(type='DistSamplerSeedHook'),
    sync_buffer=dict(type='SyncBuffersHook'),
    visualization=dict(
        type='VisualizationHook',
        interval=1,
        enable=False,
        show=False,
        draw_gt=False,
        draw_pred=False))
log_level = 'INFO'
log_processor = dict(type='LogProcessor', window_size=10, by_epoch=True)
load_from = './model.pth'
resume = False
val_evaluator = dict(
    type='Evaluator',
    metrics=[
        dict(
            type='WordMetric',
            mode=['exact', 'ignore_case', 'ignore_case_symbol'],
            valid_symbol='[^A-Z^a-z^0-9^一-龥^å^ä^ö^Å^Ä^Ö]'),
        dict(type='CharMetric', valid_symbol='[^A-Z^a-z^0-9^一-龥^å^ä^ö^Å^Ä^Ö]'),
        dict(
            type='OneMinusNEDMetric',
            valid_symbol='[^A-Z^a-z^0-9^一-龥^å^ä^ö^Å^Ä^Ö]')
    ])
test_evaluator = dict(
    type='Evaluator',
    metrics=[
        dict(
            type='WordMetric',
            mode=['exact', 'ignore_case', 'ignore_case_symbol'],
            valid_symbol='[^A-Z^a-z^0-9^一-龥^å^ä^ö^Å^Ä^Ö]'),
        dict(type='CharMetric', valid_symbol='[^A-Z^a-z^0-9^一-龥^å^ä^ö^Å^Ä^Ö]'),
        dict(
            type='OneMinusNEDMetric',
            valid_symbol='[^A-Z^a-z^0-9^一-龥^å^ä^ö^Å^Ä^Ö]')
    ])
vis_backends = [dict(type='LocalVisBackend')]
visualizer = dict(
    type='TextRecogLocalVisualizer',
    name='visualizer',
    vis_backends=[dict(type='TensorboardVisBackend')])
optim_wrapper = dict(
    type='OptimWrapper', optimizer=dict(type='Adam', lr=0.0003))
train_cfg = dict(type='EpochBasedTrainLoop', max_epochs=5, val_interval=1)
val_cfg = dict(type='ValLoop')
test_cfg = dict(type='TestLoop')
param_scheduler = [dict(type='MultiStepLR', milestones=[3, 4], end=5)]
file_client_args = dict(backend='disk')
dictionary = dict(
    type='Dictionary',
    dict_file=
    '.cache/models--Riksarkivet--satrn_htr/snapshots/27812c88be2706696b0283c51ee68ceb7b969301/dictionary.txt',
    with_padding=True,
    with_unknown=True,
    same_start_end=True,
    with_start=True,
    with_end=True)
model = dict(
    type='SATRN',
    backbone=dict(type='ShallowCNN', input_channels=3, hidden_dim=512),
    encoder=dict(
        type='SATRNEncoder',
        n_layers=12,
        n_head=8,
        d_k=64,
        d_v=64,
        d_model=512,
        n_position=100,
        d_inner=2048,
        dropout=0.1),
    decoder=dict(
        type='NRTRDecoder',
        n_layers=6,
        d_embedding=512,
        n_head=8,
        d_model=512,
        d_inner=2048,
        d_k=64,
        d_v=64,
        module_loss=dict(
            type='CEModuleLoss', flatten=True, ignore_first_char=True),
        dictionary=dict(
            type='Dictionary',
            dict_file=
            '.cache/models--Riksarkivet--satrn_htr/snapshots/27812c88be2706696b0283c51ee68ceb7b969301/dictionary.txt',
            with_padding=True,
            with_unknown=True,
            same_start_end=True,
            with_start=True,
            with_end=True),
        max_seq_len=100,
        postprocessor=dict(type='AttentionPostprocessor')),
    data_preprocessor=dict(
        type='TextRecogDataPreprocessor',
        mean=[123.675, 116.28, 103.53],
        std=[58.395, 57.12, 57.375]))
train_pipeline = [
    dict(
        type='LoadImageFromFile',
        file_client_args=dict(backend='disk'),
        ignore_empty=True,
        min_size=2),
    dict(type='LoadOCRAnnotations', with_text=True),
    dict(type='Resize', scale=(400, 64), keep_ratio=False),
    dict(
        type='PackTextRecogInputs',
        meta_keys=('img_path', 'ori_shape', 'img_shape', 'valid_ratio'))
]
test_pipeline = [
    dict(type='LoadImageFromFile', file_client_args=dict(backend='disk')),
    dict(type='Resize', scale=(400, 64), keep_ratio=False),
    dict(type='LoadOCRAnnotations', with_text=True),
    dict(
        type='PackTextRecogInputs',
        meta_keys=('img_path', 'ori_shape', 'img_shape', 'valid_ratio'))
]
HTR_1700_combined_train = dict(
    type='RecogTextDataset',
    parser_cfg=dict(type='LineJsonParser', keys=['filename', 'text']),
    data_root='/ceph/hpc/scratch/user/euerikl/data/HTR_1700_clean',
    ann_file=
    '/ceph/hpc/home/euerikl/projects/hf_openmmlab_models/data/processed/1700_HTR_shuffled_train.jsonl',
    test_mode=False,
    pipeline=None)
HTR_1700_combined_test = dict(
    type='RecogTextDataset',
    parser_cfg=dict(type='LineJsonParser', keys=['filename', 'text']),
    data_root='/ceph/hpc/scratch/user/euerikl/data/HTR_1700_clean',
    ann_file=
    '/ceph/hpc/home/euerikl/projects/hf_openmmlab_models/data/processed/1700_HTR_shuffled_val.jsonl',
    test_mode=True,
    pipeline=None)
pr_cr_combined_train = dict(
    type='RecogTextDataset',
    parser_cfg=dict(
        type='LineStrParser', keys=['filename', 'text'], separator='|'),
    data_root='/ceph/hpc/scratch/user/euerikl/data/line_images',
    ann_file=
    '/ceph/hpc/home/euerikl/projects/htr_1800/gt_files/combined_train.txt',
    test_mode=False,
    pipeline=None)
pr_cr_combined_test = dict(
    type='RecogTextDataset',
    parser_cfg=dict(
        type='LineStrParser', keys=['filename', 'text'], separator='|'),
    data_root='/ceph/hpc/scratch/user/euerikl/data/line_images',
    ann_file=
    '/ceph/hpc/home/euerikl/projects/htr_1800/gt_files/combined_eval.txt',
    test_mode=True,
    pipeline=None)
out_of_domain_1700_all_test = dict(
    type='RecogTextDataset',
    parser_cfg=dict(type='LineJsonParser', keys=['filename', 'text']),
    data_root='/ceph/hpc/scratch/user/euerikl/data/HTR_1700_testsets_clean',
    ann_file=
    '/ceph/hpc/home/euerikl/projects/hf_openmmlab_models/data/processed/1700_testsets_gt/1700_HTR_testsets_all.jsonl',
    test_mode=True,
    pipeline=None)
train_list = [
    dict(
        type='RecogTextDataset',
        parser_cfg=dict(type='LineJsonParser', keys=['filename', 'text']),
        data_root='/ceph/hpc/scratch/user/euerikl/data/HTR_1700_clean',
        ann_file=
        '/ceph/hpc/home/euerikl/projects/hf_openmmlab_models/data/processed/1700_HTR_shuffled_train.jsonl',
        test_mode=False,
        pipeline=None),
    dict(
        type='RecogTextDataset',
        parser_cfg=dict(
            type='LineStrParser', keys=['filename', 'text'], separator='|'),
        data_root='/ceph/hpc/scratch/user/euerikl/data/line_images',
        ann_file=
        '/ceph/hpc/home/euerikl/projects/htr_1800/gt_files/combined_train.txt',
        test_mode=False,
        pipeline=None)
]
test_list = [
    dict(
        type='RecogTextDataset',
        parser_cfg=dict(type='LineJsonParser', keys=['filename', 'text']),
        data_root='/ceph/hpc/scratch/user/euerikl/data/HTR_1700_testsets_clean',
        ann_file=
        '/ceph/hpc/home/euerikl/projects/hf_openmmlab_models/data/processed/1700_testsets_gt/1700_HTR_testsets_all.jsonl',
        test_mode=True,
        pipeline=None)
]
train_dataset = dict(
    type='ConcatDataset',
    datasets=[
        dict(
            type='RecogTextDataset',
            parser_cfg=dict(type='LineJsonParser', keys=['filename', 'text']),
            data_root='/ceph/hpc/scratch/user/euerikl/data/HTR_1700_clean',
            ann_file=
            '/ceph/hpc/home/euerikl/projects/hf_openmmlab_models/data/processed/1700_HTR_shuffled_train.jsonl',
            test_mode=False,
            pipeline=None),
        dict(
            type='RecogTextDataset',
            parser_cfg=dict(
                type='LineStrParser', keys=['filename', 'text'],
                separator='|'),
            data_root='/ceph/hpc/scratch/user/euerikl/data/line_images',
            ann_file=
            '/ceph/hpc/home/euerikl/projects/htr_1800/gt_files/combined_train.txt',
            test_mode=False,
            pipeline=None)
    ],
    pipeline=[
        dict(
            type='LoadImageFromFile',
            file_client_args=dict(backend='disk'),
            ignore_empty=True,
            min_size=2),
        dict(type='LoadOCRAnnotations', with_text=True),
        dict(type='Resize', scale=(400, 64), keep_ratio=False),
        dict(
            type='PackTextRecogInputs',
            meta_keys=('img_path', 'ori_shape', 'img_shape', 'valid_ratio'))
    ])
test_dataset = dict(
    type='ConcatDataset',
    datasets=[
        dict(
            type='RecogTextDataset',
            parser_cfg=dict(type='LineJsonParser', keys=['filename', 'text']),
            data_root=
            '/ceph/hpc/scratch/user/euerikl/data/HTR_1700_testsets_clean',
            ann_file=
            '/ceph/hpc/home/euerikl/projects/hf_openmmlab_models/data/processed/1700_testsets_gt/1700_HTR_testsets_all.jsonl',
            test_mode=True,
            pipeline=None)
    ],
    pipeline=[
        dict(type='LoadImageFromFile', file_client_args=dict(backend='disk')),
        dict(type='Resize', scale=(400, 64), keep_ratio=False),
        dict(type='LoadOCRAnnotations', with_text=True),
        dict(
            type='PackTextRecogInputs',
            meta_keys=('img_path', 'ori_shape', 'img_shape', 'valid_ratio'))
    ])
train_dataloader = dict(
    batch_size=8,
    num_workers=1,
    persistent_workers=True,
    sampler=dict(type='DefaultSampler', shuffle=True),
    dataset=dict(
        type='ConcatDataset',
        datasets=[
            dict(
                type='RecogTextDataset',
                parser_cfg=dict(
                    type='LineJsonParser', keys=['filename', 'text']),
                data_root='/ceph/hpc/scratch/user/euerikl/data/HTR_1700_clean',
                ann_file=
                '/ceph/hpc/home/euerikl/projects/hf_openmmlab_models/data/processed/1700_HTR_shuffled_train.jsonl',
                test_mode=False,
                pipeline=None),
            dict(
                type='RecogTextDataset',
                parser_cfg=dict(
                    type='LineStrParser',
                    keys=['filename', 'text'],
                    separator='|'),
                data_root='/ceph/hpc/scratch/user/euerikl/data/line_images',
                ann_file=
                '/ceph/hpc/home/euerikl/projects/htr_1800/gt_files/combined_train.txt',
                test_mode=False,
                pipeline=None)
        ],
        pipeline=[
            dict(
                type='LoadImageFromFile',
                file_client_args=dict(backend='disk'),
                ignore_empty=True,
                min_size=2),
            dict(type='LoadOCRAnnotations', with_text=True),
            dict(type='Resize', scale=(400, 64), keep_ratio=False),
            dict(
                type='PackTextRecogInputs',
                meta_keys=('img_path', 'ori_shape', 'img_shape',
                           'valid_ratio'))
        ]))
test_dataloader = dict(
    batch_size=8,
    num_workers=1,
    persistent_workers=True,
    drop_last=False,
    sampler=dict(type='DefaultSampler', shuffle=False),
    dataset=dict(
        type='ConcatDataset',
        datasets=[
            dict(
                type='RecogTextDataset',
                parser_cfg=dict(
                    type='LineJsonParser', keys=['filename', 'text']),
                data_root=
                '/ceph/hpc/scratch/user/euerikl/data/HTR_1700_testsets_clean',
                ann_file=
                '/ceph/hpc/home/euerikl/projects/hf_openmmlab_models/data/processed/1700_testsets_gt/1700_HTR_testsets_all.jsonl',
                test_mode=True,
                pipeline=None)
        ],
        pipeline=[
            dict(
                type='LoadImageFromFile',
                file_client_args=dict(backend='disk')),
            dict(type='Resize', scale=(400, 64), keep_ratio=False),
            dict(type='LoadOCRAnnotations', with_text=True),
            dict(
                type='PackTextRecogInputs',
                meta_keys=('img_path', 'ori_shape', 'img_shape',
                           'valid_ratio'))
        ]))
val_dataloader = dict(
    batch_size=8,
    num_workers=1,
    persistent_workers=True,
    drop_last=False,
    sampler=dict(type='DefaultSampler', shuffle=False),
    dataset=dict(
        type='ConcatDataset',
        datasets=[
            dict(
                type='RecogTextDataset',
                parser_cfg=dict(
                    type='LineJsonParser', keys=['filename', 'text']),
                data_root=
                '/ceph/hpc/scratch/user/euerikl/data/HTR_1700_testsets_clean',
                ann_file=
                '/ceph/hpc/home/euerikl/projects/hf_openmmlab_models/data/processed/1700_testsets_gt/1700_HTR_testsets_all.jsonl',
                test_mode=True,
                pipeline=None)
        ],
        pipeline=[
            dict(
                type='LoadImageFromFile',
                file_client_args=dict(backend='disk')),
            dict(type='Resize', scale=(400, 64), keep_ratio=False),
            dict(type='LoadOCRAnnotations', with_text=True),
            dict(
                type='PackTextRecogInputs',
                meta_keys=('img_path', 'ori_shape', 'img_shape',
                           'valid_ratio'))
        ]))
gpu_ids = range(0, 4)
cudnn_benchmark = True
work_dir = '/ceph/hpc/home/euerikl/projects/hf_openmmlab_models/models/checkpoints/1700_1800_combined_satrn'
checkpoint_config = dict(interval=1)
auto_scale_lr = dict(base_batch_size=32)
launcher = 'pytorch'
