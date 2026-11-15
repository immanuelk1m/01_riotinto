"""Run the bounded causal attribution study on existing snapshots."""
import argparse
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
for key in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS'):
    os.environ.setdefault(key,'1')
from analysis.causal_study import run

if __name__ == '__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path,default=ROOT/'data/07_reporting/causal_study')
    parser.add_argument('--hours',type=float,default=12)
    parser.add_argument('--fit-minutes',type=float,default=45)
    parser.add_argument('--series',nargs='+',choices=['coffee','copper','gold'])
    parser.add_argument('--models',nargs='+',choices=['pcmci','robust','gpdc','lpcmci','varlingam'])
    parser.add_argument('--wandb-mode',choices=['online','offline','disabled'],default=os.environ.get('WANDB_MODE','online'))
    args=parser.parse_args()
    if args.hours<=0 or args.fit_minutes<=0: parser.error('Budgets must be positive')
    result=run(args.output,args.hours,args.fit_minutes,args.series,args.models)
    print(result)
    if args.series or args.models: print('Subset run: leaderboard upload skipped')  # 일부만 돌린 실행은 리더보드에 올리지 않는다
    elif args.wandb_mode!='disabled':
        from tools.log_causal_leaderboard import upload
        print(upload(result,args.wandb_mode))
