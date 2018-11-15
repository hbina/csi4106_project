import subprocess
import pip

if not (hasattr(pip, 'kaggle')):
    subprocess.run(['pip', 'install', 'kaggle'])
else:
    print("kaggle is already installed!")
subprocess.run(['kaggle', 'competitions', 'download', '-c', 'titanic'])
