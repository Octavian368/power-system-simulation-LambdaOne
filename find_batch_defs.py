import os

base = r"C:\PowerSystemComputation\power-system-simulation-LambdaOne\.venv\Lib\site-packages\power_grid_model"

for root, dirs, files in os.walk(base):
    for file in files:
        if file.endswith(".py"):
            path = os.path.join(root, file)
            try:
                with open(path, encoding="utf-8") as f:
                    content = f.read()
                    if "BatchUpdateDataset" in content or "BatchResult" in content:
                        rel_path = os.path.relpath(path, base).replace("\\", ".").replace(".py", "")
                        print(f" Found in: {rel_path}")
            except Exception as e:
                print(f"Skipped {file}: {e}")
