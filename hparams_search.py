import foundations
import numpy as np


NUM_JOBS = 1


def generate_params():
    params = {'batch_size': int(np.random.choice([64, 128])),
              'n_epochs': int(np.random.choice([30, 35, 40])),
              "pct_start": float(np.random.uniform(0.3, 0.5)),
              'weight_decay': float(np.random.uniform(0.01, 0.2)),
              'dropout': float(np.random.choice([0.5, 0.75, 0.9])),
              'use_lr_scheduler': int(np.random.choice([0, 1])),
              # 'use_lr_scheduler': 0,
              'train_data': 'base',
              # 'train_data': np.random.choice(['augment', 'base', 'both'])
    }

    params['max_lr'] = 0.00025 if bool(params['use_lr_scheduler']) else 0.0001
    return params


for job_ in range(NUM_JOBS):
    print(f"packaging job {job_}")
    hyper_params = generate_params()
    print(hyper_params)
    foundations.submit(scheduler_config='scheduler', job_directory='.', command='main.py', params=hyper_params,
                       stream_job_logs=False)
