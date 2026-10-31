# Runs in the Colab kernel: start the 4 commodity jobs of direction __DIR__ (2 at a time).
import os, subprocess
os.makedirs("/content/E007/logs", exist_ok=True)
cmd = ('cd /content/E007 && printf "gold\\ncopper\\ncorn\\ncoffee\\n" | xargs -P 2 -I{} sh -c '
       '"OMP_NUM_THREADS=1 python E007.py __DIR__ {} > logs/run___DIR___{}.out 2>&1"; touch logs/DONE___DIR__')
subprocess.Popen(["nohup", "sh", "-c", cmd], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, start_new_session=True)
print("launched __DIR__")
