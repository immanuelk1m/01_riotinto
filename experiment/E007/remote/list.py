# Runs in the Colab kernel: files to download for direction __DIR__.
import glob
for f in sorted(glob.glob("/content/E007/results/__DIR__/*") + glob.glob("/content/E007/logs/*__DIR__*")):
    print(f)
