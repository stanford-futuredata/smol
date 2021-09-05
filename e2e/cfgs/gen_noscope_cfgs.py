import os

import yaml

from blazeit.data.video_data import get_video_data


# Date is train date
def gen_e2e_single(base_cfg_fname, yaml_out_dir,
                   dataset, date, model_id, resol_type, crop,
                   base_resol,
                   yaml_out_fname=None):
    with open(base_cfg_fname, 'r') as f:
        cfg = yaml.safe_load(f.read())
    os.makedirs(yaml_out_dir, exist_ok=True)
    if yaml_out_fname is None:
        yaml_out_fname = os.path.join(
                yaml_out_dir, '{}-{}.yaml'.format(resol_type, model_id))

    if 'rn50' in model_id:
        epoch = '00'
        input_dim = [224, 224]
    else:
        epoch = '04'
        input_dim = [50, 50]
    if resol_type == 'hires':
        mult = 1
    elif resol_type == '480p':
        mult = 480.0 / base_resol

    cfg['model-config']['model-single']['onnx-path'] = \
        cfg['model-config']['model-single']['onnx-path'].format(dataset, date, model_id, epoch)
    cfg['model-config']['model-single']['data-path'] = \
        cfg['model-config']['model-single']['data-path'].format(resol_type, dataset)
    cfg['model-config']['model-single']['input-dim'] = input_dim

    cfg['crop']['xmin'] = int(crop.xmin * mult)
    cfg['crop']['ymin'] = int(crop.ymin * mult)
    cfg['crop']['xmax'] = int(crop.xmax * mult)
    cfg['crop']['ymax'] = int(crop.ymax * mult)

    with open(yaml_out_fname, 'w') as f:
        yaml.dump(cfg, f)


def gen_all():
    base_cfg_fname = './vid-base.yaml'
    datasets = [('jackson-town-square', '2017-12-14', 1080),
                ('taipei-hires', '2017-04-08', 720),
                ('amsterdam', '2017-04-10', 720),
                ('archie-day', '2018-04-09', 2160),
                ('venice-grand-canal', '2018-01-17', 1080),
                ('venice-rialto', '2018-01-17', 1080)]
    resol_types = ['480p', 'hires']
    model_ids = ['rn50'] + [str(x) for x in range(6)]

    for dataset, date, base_resol in datasets:
        for resol_type in resol_types:
            for model_id in model_ids:
                vd = get_video_data(dataset)
                crop = vd.crop
                yaml_out_dir = os.path.join('noscope', dataset)
                gen_e2e_single(base_cfg_fname, yaml_out_dir,
                               dataset, date, model_id, resol_type, crop,
                               base_resol)

if __name__ == '__main__':
    gen_all()
