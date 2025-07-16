import subprocess

scripts = ["src/Train_Image_Branch.py", "src/Train_Conditional_Partition.py"]
#scripts = ["src/Train_Image_Branch.py", "src/Conditional_Train_ajustable.py","src/DualSample_ajustable_evalgen_boost.py"]

for script in scripts:
    print(f"Ejecutando {script}...")
    subprocess.run(["python", script])