import torch
import random
import numpy as np
import os
import copy
import json
import statistics
import shutil
import yaml
import argparse
from munch import Munch
from tqdm import tqdm
from tensorboardX import SummaryWriter
from torch.utils.data import DataLoader, WeightedRandomSampler
from lifelines import exceptions
import warnings
warnings.filterwarnings("ignore", category=exceptions.ApproximationWarning)

from utils.survival_metrics import CIndexMeter, IPWCIndexMeter, BrierScoreMeter
from utils.utils import *
from models import create_WSI_model
from optimizers.optim_factory import create_optimizer
from optimizers import create_scheduler
from dataset.dataset_survival_egmll import DataGeneratorTCGASurvivalWSIEGMLL
from utils.loss_funcs import NLLELGE

torch.cuda.empty_cache()

def detach(data):
    if isinstance(data, torch.Tensor):
        return data.detach().cpu()
    if isinstance(data, dict):
        return {k: detach(v) for k, v in data.items()}
    if isinstance(data, list):
        return [detach(x) for x in data]
    return data
    

def summary_folds(root_dir, target='concordance_time_dependent', method='max'):
    vals_dict = {}
    res = {}
    if method == 'last':
        for fold in range(1, 6):
            path = os.path.join(root_dir, f"fold{fold}", 'results.json')
            with open(path, 'r') as f:
                ds = json.load(f)
                d = ds[-1]
                assert d['epoch'] == 20
                for key in d['eval']:
                    if fold == 1:
                        vals_dict[key] = [d['eval'][key]]
                    else:
                        vals_dict[key].append(d['eval'][key])
                res[f"fold{fold}"] = d['eval']
    elif method == 'max':
        for fold in range(1, 6):
            path = os.path.join(root_dir, f"fold{fold}", 'results.json')
            with open(path, 'r') as f:
                ds = json.load(f)
                maxi, maxI = -1, -1
                for i, d in enumerate(ds):
                    if d['eval'][target] > maxi:
                        maxi = d['eval'][target]
                        maxI = i
                d = ds[maxI]
                for key in d['eval']:
                    if fold == 1:
                        vals_dict[key] = [d['eval'][key]]
                    else:
                        vals_dict[key].append(d['eval'][key])
                res[f"fold{fold}"] = d['eval']
    res['summary'] = {'method': method}
    for key in vals_dict:
        res['summary'][key] = {
            'mean': statistics.mean(vals_dict[key]),
            'std': statistics.stdev(vals_dict[key]),
            'min': min(vals_dict[key]),
            'max': max(vals_dict[key])
        }
        print(f"{key}:{statistics.mean(vals_dict[key])}")
    with open(os.path.join(root_dir, 'summary.json'), 'w') as f:
        json.dump(res, f, indent='\t')

def train(model, train_loader, val_loader, cfg, save_dir, metrics, with_coords=False):
    weights_dir_model = os.path.join(save_dir, 'weights')
    os.makedirs(weights_dir_model, exist_ok=True)
    writer = SummaryWriter(save_dir, flush_secs=15)
    optimizer = create_optimizer(cfg.optimizer, model)
    scheduler = create_scheduler(getattr(cfg, 'scheduler', None), optimizer)
    if cfg.optimizer.loss_func == 'elge':
        # loss_func = NLLELGE(lambda_ent=0.001)
        loss_func = NLLELGE(lambda_ent=0.01, lambda_balance=0.01)
    else:
        raise NotImplementedError

    results = []
    lambda_reg = getattr(cfg, 'lambda_reg', 0)
    reg_type = getattr(cfg, 'reg_type', None)
    reg_fn = l1_all if reg_type == 'l1_all' else None

    for epoch in range(cfg.resume + 1, cfg.epoch + 1):
        print('\nEpoch: ', epoch)
        print('====Start train====')
        model.train()
        n_clusters = []
        optimizer.zero_grad()
        for step, batch in enumerate(train_loader):
            global_step = (epoch - 1) * len(train_loader) + step
            data, constant_dict = batch
            t = data['t'].to(cfg.device)
            c = data['c'].to(cfg.device)
            x = {k: v.to(cfg.device) for k, v in data.items() if k not in ['wid','t','c']}
            train_outputs = model.train_step(x, t, c, constant_dict)
            if 'n_cluster' in train_outputs:
                n_clusters.append(train_outputs['n_cluster'])
            loss = loss_func(train_outputs)
            if 'subloss' in train_outputs and train_outputs['subloss'] is not None:
                loss = loss + train_outputs['subloss']
            loss_reg = reg_fn(model) * lambda_reg if reg_fn is not None else 0
            loss = loss / cfg.gradient_accumulation + loss_reg
            loss.backward()
            if (step + 1) % cfg.gradient_accumulation == 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=0.5)
                optimizer.step()
                optimizer.zero_grad()
            loss_val = loss.item()
            if (step + 1) % cfg.print_step == 0:
                writer.add_scalar('train/loss', loss_val, global_step)
                print('step {}/{}, loss: {:.4f}'.format(step + 1, len(train_loader), loss_val))
        if scheduler is not None:
            scheduler.step()
        torch.save(model.state_dict(), os.path.join(weights_dir_model, f'epoch_{epoch}.pt'))
        if len(n_clusters) != 0:
            print(f"Cluster number: mean{statistics.mean(n_clusters)}, min{min(n_clusters)}, max{max(n_clusters)}, median{statistics.median(n_clusters)}")
        print('====End train====')

        # EVAL TRAIN (sanity)
        print('====Eval train====')
        model.eval()
        curr_metrics = copy.deepcopy(metrics)
        first_batch_risks = None
        with torch.no_grad():
            for step, batch in tqdm(enumerate(train_loader), total=len(train_loader)):
                data, constant_dict = batch
                t = data['t'].to(cfg.device)
                c = data['c'].to(cfg.device)
                x = {k: v.to(cfg.device) for k, v in data.items() if k not in ['wid','t','c']}

                # debug print for eval_t shape coming from dataset
                if 'eval_t' in constant_dict:
                    et_shape = getattr(constant_dict['eval_t'], 'shape', None)
                    if step == 0:
                        print(f"DEBUG: constant_dict['eval_t'].shape = {et_shape}")

                eval_outputs = model.eval_step(x, t, c, constant_dict)

                if step == 0 and first_batch_risks is None:
                    first_batch_risks = detach(eval_outputs['cum_hazard_seqs'][0, :])
                    print(f"DEBUG: First batch risks - min: {first_batch_risks.min():.4f}, max: {first_batch_risks.max():.4f}, std:{first_batch_risks.std():.4f}")

                for key in curr_metrics:
                    curr_metrics[key].add(detach(eval_outputs), detach(1 - c))

        metric_value_dict = {}
        for metric_name in curr_metrics:
            metric_value_dict[metric_name] = curr_metrics[metric_name].value()
            writer.add_scalar(f'train/{metric_name}', metric_value_dict[metric_name], epoch)
        print('Epoch: {}, CTD:{:.4f}, IPWCTD:{:.4f}, Brier:{:.4f}'.format(epoch,
            metric_value_dict['concordance_time_dependent'],
            metric_value_dict['ipw_concordance_time_dependent'],
            metric_value_dict['brier_score']))

        # EVAL VAL
        print('====Eval val====')
        model.eval()
        curr_metrics = copy.deepcopy(metrics)
        with torch.no_grad():
            for step, batch in tqdm(enumerate(val_loader), total=len(val_loader)):
                data, constant_dict = batch
                t = data['t'].to(cfg.device)
                c = data['c'].to(cfg.device)
                x = {k: v.to(cfg.device) for k, v in data.items() if k not in ['wid','t','c']}

                # debug print first batch eval_t shape
                if 'eval_t' in constant_dict and step == 0:
                    print(f"DEBUG val first batch eval_t.shape = {constant_dict['eval_t'].shape}")

                eval_outputs = model.eval_step(x, t, c, constant_dict)
                for key in curr_metrics:
                    curr_metrics[key].add(detach(eval_outputs), detach(1 - c))

        eval_metric_value_dict = {}
        for metric_name in curr_metrics:
            eval_metric_value_dict[metric_name] = curr_metrics[metric_name].value()
            writer.add_scalar(f'eval/{metric_name}', eval_metric_value_dict[metric_name], epoch)
        print('Epoch: {}, CTD:{:.4f}, IPWCTD:{:.4f}, Brier:{:.4f}'.format(epoch,
            eval_metric_value_dict['concordance_time_dependent'],
            eval_metric_value_dict['ipw_concordance_time_dependent'],
            eval_metric_value_dict['brier_score']))
        results.append({"epoch": epoch, 'train': metric_value_dict, 'eval': eval_metric_value_dict})
    return results


def set_random_seed(seed):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    

def main(args, cfg, save_dir, fold):
    metrics = {}
    metrics["concordance_time_dependent"] = CIndexMeter()
    metrics["ipw_concordance_time_dependent"] = IPWCIndexMeter()
    metrics["ipw_2_concordance_time_dependent"] = IPWCIndexMeter(eps=0.2)
    metrics["ipw_4_concordance_time_dependent"] = IPWCIndexMeter(eps=0.4)
    metrics["brier_score"] = BrierScoreMeter()
    metrics["brier_score_2"] = BrierScoreMeter(eps=0.2)
    metrics["brier_score_4"] = BrierScoreMeter(eps=0.4)
    os.makedirs(save_dir, exist_ok=True)
    shutil.copyfile(args.config, os.path.join(save_dir, os.path.basename(args.config)))
    set_random_seed(cfg.seed)

    model = create_WSI_model(cfg)
    print("Model created")
    if cfg.resume > 0:
        model.load_state_dict(torch.load(os.path.join(save_dir, 'weights', f'epoch_{cfg.resume}.pt')))
    print_trainable_parameters(model)
    model.to(cfg.device)
    print("Model moved to device")

    with_coords = getattr(cfg.datasets, 'with_coords', False)
    with_cluster_label = getattr(cfg.datasets, 'with_cluster_label', False)
    if with_cluster_label:
        cluster_label_path = os.path.join(cfg.datasets.root_dir, 'cluster_label', f'fw{cfg.model.feature_weight}.h5')
    else:
        cluster_label_path = None
    clinical_path = os.path.join(cfg.datasets.root_dir, cfg.datasets.clinical_file_path)
    train_ids_path = os.path.join(cfg.datasets.root_dir, cfg.datasets.folds_path, f"fold{fold}", 'train.txt')
    val_ids_path = os.path.join(cfg.datasets.root_dir, cfg.datasets.folds_path, f"fold{fold}", 'val.txt')
    if cfg.datasets.type == 'tcga-survival-egmll-wsi':
        anno_path = os.path.join(cfg.datasets.root_dir, cfg.datasets.wsi_file_path)
        train_ds = DataGeneratorTCGASurvivalWSIEGMLL(anno_path, train_ids_path, clinical_path, shuffle=True, with_coords=with_coords, cluster_label_path=cluster_label_path)
        val_ds = DataGeneratorTCGASurvivalWSIEGMLL(anno_path, val_ids_path, clinical_path, shuffle=True, with_coords=with_coords, cluster_label_path=cluster_label_path)
    else:
        raise NotImplementedError
    print(
        f'Datasets loaded! Train sample num: {len(train_ds)}. Val sample num: {len(val_ds)}.')
    torch.multiprocessing.set_start_method('spawn', force=True)
    Loader = DataLoader
    if getattr(cfg.datasets, 'weighted_sample', False):
        train_loader = Loader(train_ds, batch_size=cfg.batch_size, sampler=WeightedRandomSampler(train_ds.get_weights(), len(train_ds)), pin_memory=True, num_workers=8)
    else:
        train_loader = Loader(train_ds, batch_size=cfg.batch_size, shuffle=True, pin_memory=True, num_workers=8)
    val_loader = Loader(val_ds, batch_size=cfg.batch_size, shuffle=True, pin_memory=True, num_workers=8)
    results = train(model, train_loader, val_loader, cfg, save_dir, metrics, with_coords=with_coords)
    with open(os.path.join(save_dir, 'results.json'), 'w') as f:
        json.dump(results, f, indent='\t')


if __name__ == '__main__':
    import multiprocessing as mp
    mp.set_start_method("spawn", force=True)  # avoids many fork-related leaks
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', default='configs/kirc_sgcmll.yaml')
    args = parser.parse_args()

    with open(args.config, 'r', encoding='utf-8') as fin:
        cfg = yaml.load(fin, Loader=yaml.FullLoader)
    cfg = Munch.fromDict(cfg)
    
    for fold in cfg.datasets.fold:
        save_dir = os.path.join(cfg.save_dir, cfg.config_name, f"fold{fold}")
        main(args, cfg, save_dir, fold)
    summary_folds(os.path.join(cfg.save_dir, cfg.config_name))
