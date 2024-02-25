import os
import json
import numpy as np
from matplotlib import pyplot as plt
import pandas as pd

"""
 generate path list
 path_list = [
     './logs/CUB/ERM/base_full_clslr1_bigcolor_wd0.5_seed0'
 ]
"""

path_list = [    
    'domain-generalization/StochCA/logs/CUB/StochCA/datasetCUB_ratio0.15_ca_prob0.1_seed0_test', 
    'domain-generalization/StochCA/logs/CUB/StochCA/datasetCUB_ratio0.15_ca_prob0.1_seed1_test', 
    'domain-generalization/StochCA/logs/CUB/StochCA/datasetCUB_ratio0.15_ca_prob0.1_seed2_test'
]

rows = []
for path in path_list:
    print(path)
    with open(os.path.join(path, 'train_log.json'), 'r') as f:
        temp = list(f)
    # Define measures
    measures = {
        'train_step':[], 
        'train_loss':[], 
        'train_acc':[], 
        'val_loss':[], 
        'val_acc':[], 
        'test_loss':[], 
        'test_acc':[], 
    }
    for line in temp:
        result = json.loads(line)
        for k, v in measures.items():
            v.append(result[k])

    # Get best result
    best_step= np.argmax(measures['val_acc'])
    total_step = len(measures['val_acc'])
    val_acc = measures['val_acc'][best_step]
    test_acc = measures['test_acc'][best_step]
    print(f"Best validation acc at step {best_step+1}/{total_step} : Test acc : {test_acc}")
    rows.append([path, val_acc, test_acc])
    
    # Get last result
    total_step = len(measures['val_acc'])
    val_acc = measures['val_acc'][-1]
    test_acc = measures['test_acc'][-1]
    val_loss = measures['val_loss'][-1]
    print(f"Best validation acc at step {best_step+1}/{total_step} : Test acc : {test_acc}")
    rows.append([path, val_loss, val_acc, test_acc])
    
    # Get curve
    # Loss curve
    for k, v in measures.items():
        if 'loss' in k:
            plt.plot(measures['train_step'], v, label=k)
    plt.title(f"Loss Curve Best val acc at step {best_step+1}/{total_step} val acc : {val_acc:.2f}")
    # plt.title(f"Loss Curve val acc : {val_acc:.2f}")
    plt.legend(loc='upper right')
    plt.xlabel('step')
    plt.savefig(os.path.join(path, 'loss.png'))
    plt.close()
    # Acc curve
    for k, v in measures.items():
        if 'acc' in k:
            plt.plot(measures['train_step'], v, label=k)
    plt.title(f"Acc Curve Best val acc at step {best_step+1}/{total_step} val acc : {val_acc:.2f}")
    # plt.title(f"Acc Curve val acc : {val_acc:.2f}")
    plt.legend(loc='lower right')
    plt.xlabel('step')
    plt.savefig(os.path.join(path, 'acc.png'))
    plt.close()

df = pd.DataFrame(rows, columns=['Path', 'Val Loss', 'Val ACC', 'Test ACC'])
df.to_csv('result.csv')
