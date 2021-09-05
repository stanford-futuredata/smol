import glob
import os
import shutil
import subprocess
import time

import numpy as np
import pandas as pd
import yaml


def compute_accuracy(gt_labels, pred_fname):
    nb_classes = len(set(gt_labels))
    probs = np.fromfile(pred_fname, dtype=np.float32).reshape((-1, nb_classes))
    probs = probs[0:len(gt_labels)]
    preds = np.argmax(probs, axis=1)
    nb_correct = (preds == np.array(gt_labels)).sum()
    return float(nb_correct) / len(gt_labels) * 100


def run_single(executable, cfg_path, out_dir, NB_TRIALS=5):
    print('Running single experiment with:')
    print('  executable:', executable)
    print('  cfg_path:', cfg_path)
    print('  out_dir:', out_dir)
    print('  NB_TRIALS:', NB_TRIALS)

    exec_dir = os.path.dirname(executable)
    os.chdir(exec_dir)
    pred_fname = os.path.join(exec_dir, 'preds.out')

    # Delete cached files
    model_paths = glob.glob(os.path.join(exec_dir, '*batch*'))
    for path in model_paths:
        os.remove(path)
    if os.path.exists(pred_fname):
        os.remove(pred_fname)

    # Set up output dir, other stuff
    if os.path.exists(out_dir):
        shutil.rmtree(out_dir)
    os.makedirs(out_dir)
    cfg = yaml.safe_load(open(cfg_path, 'r').read())
    do_compute_acc = cfg['experiment-config']['write-out']
    # Doesn't matter which one
    data_path = cfg['model-config']['model-single']['data-path']
    classes = sorted(os.listdir(data_path))
    gt_labels = []
    for cls_idx, cls in enumerate(classes):
        for _ in os.listdir(os.path.join(data_path, cls)):
            gt_labels.append(cls_idx)
    print('Set up output dir')

    # Run one for generating the model and saving the outputs
    stdout_fname = os.path.join(out_dir, 'setup.stdout')
    stderr_fname = os.path.join(out_dir, 'setup.stderr')
    with open(stdout_fname, 'w') as stdout, open(stderr_fname, 'w') as stderr:
        ret_code = subprocess.call([executable, cfg_path], stdout=stdout, stderr=stderr)
    if ret_code != 0:
        print('Uh oh, something went wrong')
    time.sleep(1)
    print('Generated model file and outputs')
    if do_compute_acc:
        shutil.copy(pred_fname, os.path.join(out_dir, 'preds.out'))
        acc = compute_accuracy(gt_labels, os.path.join(out_dir, 'preds.out'))
        print('Accuracy:', acc)
    else:
        acc = 0.

    all_times = []
    for i in range(NB_TRIALS):
        print('Starting trial {}'.format(i))
        stdout_fname = os.path.join(out_dir, '{}.stdout'.format(i))
        stderr_fname = os.path.join(out_dir, '{}.stderr'.format(i))
        with open(stdout_fname, 'w') as stdout, open(stderr_fname, 'w') as stderr:
            ret_code = subprocess.call([executable, cfg_path], stdout=stdout, stderr=stderr)
        if ret_code != 0:
            print('Uh oh, something went wrong', ret_code)
            continue
        with open(stderr_fname, 'r') as f:
            lines = f.readlines()
        lines = filter(lambda x: 'Runtime' in x, lines)
        lines = map(lambda x: x.split(' ')[-1], lines)
        times = map(lambda x: float(x), lines)
        times = list(times)
        total_time = sum(times)
        print('Took', total_time, 'seconds')
        all_times.append(total_time)

    print('Finished run')
    print(acc, all_times)
    data = [[acc] + all_times]
    col_names = ['acc'] + [str(i) for i in range(len(all_times))]
    print(data)
    print(col_names)
    df = pd.DataFrame(data, columns=col_names)
    out_csv = os.path.join(out_dir, 'data.csv')
    df.to_csv(out_csv, index=False)


def run_many(dataset, cfg_dir, out_base):
    if not cfg_dir.endswith('/'):
        cfg_dir += '/'

    for root, dirs, files in os.walk(cfg_dir):
        for base_fname in files:
            if base_fname.endswith('.yaml'):
                cfg_fname = os.path.join(root, base_fname)
                no_ext = os.path.splitext(base_fname)[0]
                out_dir = os.path.join(out_base, dataset, root[len(cfg_dir):], no_ext)
                print(out_dir)
                run_single('/lfs/1/ddkang/vision-inf/image-serving/e2e/trt/build/runner',
                           cfg_fname,
                           out_dir,
                           5)


def main():
    # dataset = 'animals-10'
    # run_many(dataset,
    #          '/lfs/1/ddkang/vision-inf/image-serving/e2e/trt/cfgs/{}'.format(dataset),
    #          '/lfs/1/ddkang/vision-inf/data/output-test/')

    run_single('/lfs/1/ddkang/vision-inf/image-serving/e2e/trt/build/runner',
               '/lfs/1/ddkang/vision-inf/image-serving/e2e/cfgs/im-test.yaml',
               '/lfs/1/ddkang/vision-inf/data/output-test',
               )

if __name__ == '__main__':
    main()
