import foundations
import numpy as np


NUM_JOBS = 34


def generate_params():
    params = {'batch_size': int(np.random.choice([256, 128, 512])),
              'n_epochs': int(np.random.choice([30, 35, 40])),
              "pct_start": float(np.random.uniform(0.3, 0.5)),
              'weight_decay': float(np.random.uniform(0.01, 0.2)),
              'dropout': float(np.random.choice([0.5, 0.9, 0.75])),
              'max_lr': float(np.random.uniform(0.00005, 0.0001)),
              'use_lr_scheduler': int(np.random.choice([0, 1])),
              'use_hidden_layer': int(np.random.choice([0, 1])),
              # 'use_lr_scheduler': 1,
              # 'train_data': 'base',
              'train_data': np.random.choice(['augment', 'base', 'both'])
    }

    return params


for job_ in range(NUM_JOBS):
    print(f"packaging job {job_}")
    hyper_params = generate_params()
    print(hyper_params)
    foundations.submit(scheduler_config='scheduler', job_directory='.', command='main.py', params=hyper_params,
                       stream_job_logs=False)
