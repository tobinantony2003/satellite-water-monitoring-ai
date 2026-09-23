"""
Main Entrypoint for the Satellite Water Monitoring AI Pipeline.
"""
import os
import warnings
if 'LOKY_MAX_CPU_COUNT' not in os.environ:
    os.environ['LOKY_MAX_CPU_COUNT'] = str(os.cpu_count() or 4)
warnings.filterwarnings('ignore', message='.*physical cores.*')
warnings.filterwarnings('ignore', category=UserWarning, module='.*loky.*')

from run_pipeline import main

if __name__ == '__main__':
    main()
