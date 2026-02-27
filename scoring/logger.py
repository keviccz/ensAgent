import os
from datetime import datetime

def get_new_log_path(log_dir='output_log'):
    os.makedirs(log_dir, exist_ok=True)
    existing = [f for f in os.listdir(log_dir) if f.startswith('log_') and f.endswith('.txt')]
    if existing:
        nums = [int(f.split('_')[1].split('.')[0]) for f in existing if f.split('_')[1].split('.')[0].isdigit()]
        next_num = max(nums) + 1 if nums else 1
    else:
        next_num = 1
    return os.path.join(log_dir, f'log_{next_num:03d}.txt')

class Logger:
    def __init__(self, log_path=None):
        self.log_path = log_path or get_new_log_path()
    def log(self, msg):
        os.makedirs(os.path.dirname(self.log_path), exist_ok=True)
        with open(self.log_path, 'a', encoding='utf-8') as f:
            f.write(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] {msg}\n") 