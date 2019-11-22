import foundations
import numpy as np


NUM_JOBS = 140


def generate_params():
    params = {'batch_size': int(np.random.choice([256, 512, 1024])),
              'n_epochs': int(np.random.choice([20, 15, 25])),
              "pct_start": float(np.random.uniform(0.3, 0.5)),
              'weight_decay': float(np.random.uniform(0.01, 0.3)),
              'dropout': float(np.random.choice([0.8, 0.9, 0.75])),
              'max_lr': float(np.random.uniform(0.00003, 0.00007)),
              'use_lr_scheduler': int(np.random.choice([0, 1])),
              'use_hidden_layer': int(np.random.choice([0, 1])),
              # 'use_lr_scheduler': 1,
              'train_data': 'both',
              # 'train_data': np.random.choice(['augment', 'base', 'both'])
              'sample_ratio': float(np.random.choice([0.1, 0.25, 0.5, 0.75, 1.])),
    }

    return params


for job_ in range(NUM_JOBS):
    print(f"packaging job {job_}")
    hyper_params = generate_params()
    print(hyper_params)
    foundations.submit(scheduler_config='scheduler', job_directory='.', command='main.py', params=hyper_params,
                       stream_job_logs=False)
