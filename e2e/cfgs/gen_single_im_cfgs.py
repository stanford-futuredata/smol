import os
import yaml


def gen_e2e_single(base_cfg_fname, yaml_out_dir,
                   dataset, model_depth, model_type,
                   data_type, data_loader,
                   yaml_out_fname=None):
    with open(base_cfg_fname, 'r') as f:
        cfg = yaml.safe_load(f.read())
    model_fname = model_type.format(model_depth)
    model_base = os.path.splitext(model_fname)[0].format(64)
    os.makedirs(yaml_out_dir, exist_ok=True)
    if yaml_out_fname is None:
        yaml_out_fname = os.path.join(
                yaml_out_dir, '{}-{}-{}.yaml'.format(
                        data_type, data_loader, model_base))
    print(yaml_out_fname)

    model_bs64 = model_fname.format(64)
    model_bs1 = model_fname.format(1)
    cfg['model-config']['model-single']['onnx-path'] = \
        cfg['model-config']['model-single']['onnx-path'].format(dataset, model_bs64)
    cfg['model-config']['model-single']['onnx-path-bs1'] = \
        cfg['model-config']['model-single']['onnx-path-bs1'].format(dataset, model_bs1)
    engine_fname = model_bs64.replace(
            'onnx', '{loader}-{dtype}.engine'.format(
                    loader=data_loader, dtype=data_type))
    cfg['model-config']['model-single']['engine-path'] = \
        cfg['model-config']['model-single']['engine-path'].format(
                dataset, engine_fname)

    cfg['model-config']['model-single']['data-path'] = \
        cfg['model-config']['model-single']['data-path'].format(dataset, data_type)
    cfg['model-config']['model-single']['data-loader'] = data_loader

    if dataset == 'imagenet':
        cfg['experiment-config']['multiplier'] = 4

    with open(yaml_out_fname, 'w') as f:
        yaml.dump(cfg, f)


# FIXME: full only?
def gen_preproc_ablation(base_cfg_fname, dataset, condition, data_loader):
    with open(base_cfg_fname, 'r') as f:
        cfg = yaml.safe_load(f.read())
    yaml_out_dir = os.path.join(dataset, 'preproc-ablation')
    os.makedirs(yaml_out_dir, exist_ok=True)
    yaml_out_fname = os.path.join(
            yaml_out_dir, 'full-{}-{}.yaml'.format(data_loader, condition))
    print(yaml_out_fname)

    cfg['model-config']['model-single']['onnx-path'] = \
        cfg['model-config']['model-single']['onnx-path'].format(dataset, 'fullres_rn18_ft.bs64.onnx')
    cfg['model-config']['model-single']['onnx-path-bs1'] = \
        cfg['model-config']['model-single']['onnx-path-bs1'].format(dataset, 'fullres_rn18_ft.bs1.onnx')
    cfg['model-config']['model-single']['engine-path'] = \
        cfg['model-config']['model-single']['engine-path'].format(dataset, 'fullres_rn18_ft.engine')

    cfg['model-config']['model-single']['data-path'] = \
        cfg['model-config']['model-single']['data-path'].format(dataset, 'full')
    cfg['model-config']['model-single']['data-loader'] = data_loader

    cfg['experiment-config']['run-infer'] = False
    cfg['experiment-config']['write-out'] = False
    cfg['experiment-config']['exp-type'] = condition
    if dataset == 'imagenet':
        cfg['experiment-config']['multiplier'] = 4

    with open(yaml_out_fname, 'w') as f:
        yaml.dump(cfg, f)


def gen_tahoma_single(base_cfg_fname, yaml_out_dir,
                      dataset, model_id,
                      yaml_out_fname=None):
    with open(base_cfg_fname, 'r') as f:
        cfg = yaml.safe_load(f.read())
    os.makedirs(yaml_out_dir, exist_ok=True)
    if yaml_out_fname is None:
        yaml_out_fname = os.path.join(
                yaml_out_dir, '{}.yaml'.format(model_id))
    print(yaml_out_fname)

    # FIXME: HORRIBLE HACK
    if int(model_id) < 4:
        cfg['model-config']['model-single']['input-dim'] = [30, 30]
        cfg['model-config']['model-single']['resize-dim'] = [34, 34]

    cfg['model-config']['model-single']['onnx-path'] = \
        '/lfs/1/ddkang/vision-inf/data/models/tahoma/{}/{}.bs64.onnx'.format(dataset, model_id)
    cfg['model-config']['model-single']['onnx-path-bs1'] = \
        '/lfs/1/ddkang/vision-inf/data/models/tahoma/{}/{}.bs1.onnx'.format(dataset, model_id)
    engine_fname = cfg['model-config']['model-single']['onnx-path'].replace(
            '.onnx', '{loader}-{dtype}.engine'.format(
                    loader='naive', dtype='full'))
    cfg['model-config']['model-single']['engine-path'] = engine_fname
    cfg['model-config']['model-single']['data-path'] = \
        cfg['model-config']['model-single']['data-path'].format(dataset, 'full')
    cfg['model-config']['model-single']['data-loader'] = 'naive'

    if dataset == 'imagenet':
        cfg['experiment-config']['multiplier'] = 4

    with open(yaml_out_fname, 'w') as f:
        yaml.dump(cfg, f)


def main():
    fname = './im-single-full-base.yaml'

    datasets = ['bike-bird', 'birds-200', 'animals-10', 'imagenet']
    model_types = ['fullres_{}_ft.bs{{}}.onnx',
                   'thumbnail_{}_upsample_ft.bs{{}}.onnx']
    dtl = [('full', 'naive'), ('full', 'opt-jpg'),
           ('161-jpeg-75', 'opt-jpg'), ('161-jpeg-95', 'opt-jpg'),
           ('161-png', 'opt-png')]

    # Single end-to-end cfgs
    for dataset in datasets:
        for model_depth in ['rn18', 'rn34', 'rn50']:
            for model_type in model_types:
                for data_type, data_loader in dtl:
                    yaml_out_dir = os.path.join(dataset, 'full')
                    gen_e2e_single(fname, yaml_out_dir,
                                   dataset, model_depth, model_type,
                                   data_type, data_loader)

    # Preproc ablations
    for dataset in datasets:
        for condition in ['decode-only', 'decode-resize', 'decode-resize-norm', 'all']:
            for data_loader in ['opt-jpg', 'naive']:
                gen_preproc_ablation(fname, dataset, condition, data_loader)

    # Tahoma
    for dataset in datasets:
        for model_id in map(str, range(8)):
            yaml_out_dir = os.path.join('tahoma', dataset)
            gen_tahoma_single(fname, yaml_out_dir, dataset, model_id)

if __name__ == '__main__':
    main()
