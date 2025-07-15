import subprocess

#scripts = ["src/Train.py", "src/Conditional_Train_ajustable.py","src/DualSample_ajustable_evalgen.py"]
scripts = ["DualSample_ajustable_evalgen_boost.py"]
#scripts = ["DualSample_ajustable_evalgen.py"]


for script in scripts:
    print(f"Ejecutando {script}...")
    subprocess.run(["python", script])