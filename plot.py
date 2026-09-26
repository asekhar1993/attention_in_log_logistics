import pdb
import yaml
import argparse
from tqdm import tqdm
from utils.utils import *
from models import create_WSI_model
from lifelines import KaplanMeierFitter
from lifelines.statistics import logrank_test
from dataset.dataset_survival_egmll import DataGeneratorTCGASurvivalWSIEGMLL
from munch import Munch
import torch
import numpy as np
import os
import json
import warnings
import seaborn as sns
import matplotlib.pyplot as plt
import copy
from torch.utils.data import DataLoader

def detach(data):
    if isinstance(data, torch.Tensor):
        return data.detach().cpu()
    if isinstance(data, dict):
        detached_data = {}
        for key in data:
            detached_data[key] = detach(data[key])
    elif type(data) == list:
        detached_data = []
        for x in data:
            detached_data.append(detach(x))
    else:
        raise NotImplementedError("Type {} not supported.".format(type(data)))
    return detached_data

def set_random_seed(seed):
    import random
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True



def plot_risk_stratified_cohort(high_risk_data, low_risk_data, p_value, save_path=None):
    """
    Plot Kaplan-Meier survival curves for high-risk and low-risk groups with p-value from log-rank test.
    high_risk_data/low_risk_data: Lists of tuples (t, c) where t is survival time and c is the
    CENSORSHIP indicator (c=1 -> censored, c=0 -> event/death observed), matching the convention
    used throughout the rest of this codebase (train.py, NLLELGE, etc.).

    NOTE: lifelines' KaplanMeierFitter/logrank_test expect `event_observed=1` to mean the event
    (death) happened -- the opposite convention. We invert c -> (1 - c) right before handing
    anything to lifelines so the curves/p-value are computed correctly.
    p_value: P-value from log-rank test to display.
    """
    plt.figure(facecolor='white')
    plt.rcParams.update({'font.size': 28})
    sns.set_theme(style="ticks")
    
    kmf = KaplanMeierFitter()
    high_y_last, low_y_last = None, None
    high_median_survival, low_median_survival = None, None
    
    # High-risk group
    if high_risk_data:
        high_ts, high_cs = zip(*high_risk_data)
        high_events = [1 - c for c in high_cs]  # censorship -> event indicator for lifelines
        kmf.fit(high_ts, event_observed=high_events, label='High Risk')
        survival_prob = kmf.survival_function_ * 100
        time_points = kmf.survival_function_.index
        plt.plot(time_points, survival_prob, color='red', drawstyle='steps-post', marker='+', linewidth=3)
        high_y_last = survival_prob.iloc[-1].item()  # Extract scalar value
        # Compute median survival time for debug
        surv_array = survival_prob.values / 100
        time_array = time_points.values
        idx = np.where(surv_array <= 0.5)[0]
        high_median_survival = time_array[idx[0]] if len(idx) > 0 else float('inf')
        print(f"High Risk: Median KM survival time = {high_median_survival:.1f} months, Last survival prob = {high_y_last:.1f}%")
    
    # Low-risk group
    if low_risk_data:
        low_ts, low_cs = zip(*low_risk_data)
        low_events = [1 - c for c in low_cs]  # censorship -> event indicator for lifelines
        kmf.fit(low_ts, event_observed=low_events, label='Low Risk')
        survival_prob = kmf.survival_function_ * 100
        time_points = kmf.survival_function_.index
        plt.plot(time_points, survival_prob, color='green', drawstyle='steps-post', marker='+', linewidth=3)
        low_y_last = survival_prob.iloc[-1].item()  # Extract scalar value
        # Compute median survival time for debug
        surv_array = survival_prob.values / 100
        time_array = time_points.values
        idx = np.where(surv_array <= 0.5)[0]
        low_median_survival = time_array[idx[0]] if len(idx) > 0 else float('inf')
        print(f"Low Risk: Median KM survival time = {low_median_survival:.1f} months, Last survival prob = {low_y_last:.1f}%")
    
    # Validate that high-risk has lower survival
    if high_median_survival is not None and low_median_survival is not None:
        if high_median_survival > low_median_survival:
            print("Warning: High-risk group has higher median survival time than low-risk group!")
    
    xmax = max([t for t, _ in high_risk_data + low_risk_data], default=100) + 1
    plt.xlim(0, xmax)
    plt.ylim(0, 100)
    plt.xlabel('Time (months)', fontsize=22)
    plt.ylabel('Survival Probability (%)', fontsize=22)
    ax = plt.gca()
    ax.set_facecolor('white')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_visible(False)
    ax.spines['bottom'].set_visible(False)
    ax.tick_params(axis='x', length=0)
    ax.tick_params(axis='y', length=0)
    plt.xticks(fontsize=16)
    plt.yticks(fontsize=16)
    
    # Add p-value annotation in top-right corner
    if p_value is not None:
        plt.text(0.93, 1.0, f'p={p_value:.3e}', transform=ax.transAxes, fontsize=20,  ha='right', va='top')

    
    # Add curve annotations in top-right corner
    plt.text(0.95, 0.90, 'High Risk', color='red', fontsize=20,
             transform=ax.transAxes, verticalalignment='top', horizontalalignment='right')
    plt.text(0.95, 0.80, 'Low Risk', color='green', fontsize=22,
             transform=ax.transAxes, verticalalignment='top', horizontalalignment='right')
    
    plt.tight_layout()
    # plt.legend(loc='upper left')
    ax.annotate('', xy=(xmax, 0), xytext=(-0.5, 0),
                arrowprops=dict(facecolor='black', edgecolor='black', arrowstyle='->', lw=2))
    ax.annotate('', xy=(0, 100), xytext=(0, -0.5),
                arrowprops=dict(facecolor='black', edgecolor='black', arrowstyle='->', lw=2))
    
    if save_path:
        plt.savefig(save_path, transparent=False, facecolor='white')
    plt.close()

def compute_median_survival_time(x, y):
    """
    Compute the median survival time (time where survival probability drops below 50%).
    x: Time points (list or array)
    y: Survival probabilities (list or array, in [0, 1])
    Returns: Median survival time or float('inf') if survival never drops below 50%.
    """
    y = np.array(y)
    x = np.array(x)
    idx = np.where(y <= 0.5)[0]
    if len(idx) == 0:
        return float('inf')
    return x[idx[0]]

def main(args, cfg, save_dir, fold, save_distribution_data=True):
    mini = 1e-6
    set_random_seed(cfg.seed)
    
    model = create_WSI_model(cfg)
    model.to(cfg.device)
    model.load_state_dict(torch.load(os.path.join(save_dir, 'weights', 'epoch_20.pt')))
    model.eval()

    with_coords = getattr(cfg.datasets, 'with_coords', False)
    if cfg.datasets.type == 'tcga-survival-egmll-wsi':
        anno_path = os.path.join(cfg.datasets.root_dir, cfg.datasets.wsi_file_path)
        clinical_path = os.path.join(cfg.datasets.root_dir, cfg.datasets.clinical_file_path)
        val_ids_path = os.path.join(cfg.datasets.root_dir, cfg.datasets.folds_path, f"fold{fold}", 'val.txt')
        val_ds = DataGeneratorTCGASurvivalWSIEGMLL(anno_path, val_ids_path, clinical_path, shuffle=False, with_coords=with_coords, with_ids=True)
        val_ds.summary('Val')
    else:
        raise NotImplementedError
    print(f'Datasets loaded! Val sample num: {len(val_ds)}.')
    val_loader = DataLoader(val_ds,
                            batch_size=cfg.batch_size,
                            shuffle=False,
                            pin_memory=True,
                            num_workers=8)
    
    # Collect times, censorships, and median survival times
    ts = []  # Survival times
    cs = []  # Censorship indicators
    median_survival_times = {}  # wid -> median survival time
    data_distribution = {}
    with torch.no_grad():
        for step, batch in tqdm(enumerate(val_loader), total=len(val_loader)):
            data, constant_dict = batch
            wid = data['wid'][0]
            t = data['t'].to(cfg.device)
            c = data['c'].to(cfg.device)
            x = {}
            for key in data:
                if key not in ['wid', 't', 'c']:
                    x[key] = data[key].to(cfg.device)
            ret = model.predict_step(x)
            ret = detach(ret)
            x_vals = ret['t'].numpy()
            y = ret['p_survival'].numpy()
            ids = y > mini
            x_vals = x_vals.tolist()
            y = y.tolist()
            if save_distribution_data:
                data_distribution[wid] = {
                    'censorship': c.item(),
                    't': t.item(),
                    'x': x_vals,
                    'y': y
                }
            ts.append(t.item())
            cs.append(c.item())
            median_survival_times[wid] = compute_median_survival_time(x_vals, y)
    
    if save_distribution_data:
        save_path = os.path.join(save_dir, 'val_distribution_data.json')
        with open(save_path, 'w') as f:
            json.dump(data_distribution, f)
    
    return data_distribution, ts, cs, median_survival_times

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', default='configs/brca_sgcmll.yaml')
    args = parser.parse_args()

    with open(args.config, 'r', encoding='utf-8') as fin:
        cfg = yaml.load(fin, Loader=yaml.FullLoader)
    cfg = Munch.fromDict(cfg)

    for fold in [3]:
        save_dir = os.path.join(cfg.save_dir, cfg.config_name, f"fold{fold}")
        save_curves_dir = os.path.join(save_dir, 'curves')
        os.makedirs(save_curves_dir, exist_ok=True)
        print(f"Fold: {fold}")
        
        # Run main to get data distribution, cohort data, and median survival times
        data_distribution, ts, cs, median_survival_times = main(args, cfg, save_dir, fold, save_distribution_data=True)
        
        valid_medians = [m for m in median_survival_times.values() if m != float('inf')]
        if valid_medians:
            # Test thresholds at percentiles (10th to 90th)
            percentiles = np.arange(10, 91, 5)
            best_p_value = float('inf')
            best_threshold = None
            best_high_risk_data = None
            best_low_risk_data = None
            
            for perc in percentiles:
                risk_threshold = np.percentile(valid_medians, perc)
                # Shorter predicted median survival -> high risk; longer -> low risk.
                high_risk_data = [(data_distribution[wid]['t'], data_distribution[wid]['censorship'])
                                  for wid, median in median_survival_times.items() if median <= risk_threshold]
                low_risk_data = [(data_distribution[wid]['t'], data_distribution[wid]['censorship'])
                                 for wid, median in median_survival_times.items() if median > risk_threshold]
                
                
                if high_risk_data and low_risk_data:
                    high_ts, high_cs = zip(*high_risk_data)
                    low_ts, low_cs = zip(*low_risk_data)
                    # high_cs/low_cs are censorship indicators (1=censored, 0=event); lifelines
                    # wants event_observed=1 to mean the event happened, so invert here.
                    high_events = [1 - c for c in high_cs]
                    low_events = [1 - c for c in low_cs]
                    results = logrank_test(high_ts, low_ts, event_observed_A=high_events, event_observed_B=low_events)
                    p_value = results.p_value
                    
                    # Debug: Print group sizes, p-value, and mean actual survival times
                    high_actual_ts = [t for t, c in high_risk_data]
                    low_actual_ts = [t for t, c in low_risk_data]
                    high_mean_t = np.mean(high_actual_ts) if high_actual_ts else float('inf')
                    low_mean_t = np.mean(low_actual_ts) if low_actual_ts else float('inf')
                    print(f"Percentile {perc}: High-risk size={len(high_risk_data)}, Low-risk size={len(low_risk_data)}, "
                          f"Mean actual survival (high)={high_mean_t:.1f}, (low)={low_mean_t:.1f}, p={p_value:.3f}")
                    # print(f"Percentile {perc}: High-risk size={len(high_risk_data)}, Low-risk size={len(low_risk_data)}, "
                          # f"Mean actual survival (high)={high_mean_t:.1f}, (low)={low_mean_t:.1f}, p={p_value:.3e}")
                    
                    if p_value < best_p_value:
                        best_p_value = p_value
                        best_threshold = risk_threshold
                        best_high_risk_data = high_risk_data
                        best_low_risk_data = low_risk_data
                    
                    if p_value < 0.01:
                        break  # Stop if we find a threshold with p < 0.01
            
            if best_high_risk_data and best_low_risk_data:
                save_path = os.path.join(save_curves_dir, 'risk_stratified_cohort.png')
                plot_risk_stratified_cohort(best_high_risk_data, best_low_risk_data, best_p_value, save_path)
                print(f"Risk-stratified plot saved at {save_path} with p-value={best_p_value:.3f}")
                # print(f"Risk-stratified plot saved at {save_path} with p-value={best_p_value:.3e}")
                if best_p_value >= 0.01:
                    print("Warning: No threshold found with p-value < 0.01. Using threshold with lowest p-value.")
            else:
                print("Warning: Insufficient data for high-risk or low-risk groups. Skipping risk-stratified plot.")
        else:
            print("Warning: No valid median survival times found. Skipping risk-stratified plot.")
