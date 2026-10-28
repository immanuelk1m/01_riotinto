# Runs in the Colab kernel: progress of direction __DIR__.
import glob, os
L = "/content/E007/logs"
outs = {os.path.basename(f): open(f).read() for f in sorted(glob.glob(f"{L}/run___DIR___*.out"))}
done = sum("mp4 (" in t for t in outs.values())
err = [k for k, t in outs.items() if "Traceback" in t or "Killed" in t]
print(f"STATUS __DIR__ done={done}/5 err={err} finished={os.path.exists(f'{L}/DONE___DIR__')}")
for k, t in outs.items():
    print("  ", k[len("run___DIR___"):-4].ljust(14), (t.strip().splitlines() or [""])[-1][24:120])
