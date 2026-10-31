# Runs in the Colab kernel: start the 5 commodity jobs of direction __DIR__ (2 at a time).
import glob, os, subprocess
os.makedirs("/content/E008/logs", exist_ok=True)
commodities = ("gold", "copper", "corn", "coffee", "wheat")
cmd = ('cd /content/E008 && printf "gold\\ncopper\\ncorn\\ncoffee\\nwheat\\n" | xargs -P 2 -I{} sh -c '
       '"OMP_NUM_THREADS=1 python E008.py __DIR__ {} > logs/run___DIR___{}.out 2>&1"; touch logs/DONE___DIR__')
expected = {f"/content/E008/logs/run___DIR___{commodity}.out" for commodity in commodities}
if expected.issubset(glob.glob("/content/E008/logs/run___DIR___*.out")):  # restarted pane: resume polling
    print("already launched __DIR__")
else:
    try:
        os.unlink("/content/E008/logs/DONE___DIR__")
    except FileNotFoundError:
        pass
    subprocess.Popen(["nohup", "sh", "-c", cmd], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, start_new_session=True)
    print("launched __DIR__")
