import yaml
import sys

def parse_freeze(file_path):
    pkgs = {}
    try:
        with open(file_path, 'r') as f:
            for line in f:
                if '==' in line:
                    name, ver = line.strip().split('==')
                    pkgs[name.lower()] = ver
                elif '@' in line:
                    name = line.split('@')[0].strip()
                    pkgs[name.lower()] = 'link'
    except FileNotFoundError:
        pass
    return pkgs

def get_yaml_pip_deps(yaml_path):
    try:
        with open(yaml_path, 'r') as f:
            data = yaml.safe_load(f)
            for dep in data.get('dependencies', []):
                if isinstance(dep, dict) and 'pip' in dep:
                    pip_deps = {}
                    for p in dep['pip']:
                        if '==' in p:
                            name, ver = p.split('==')
                            pip_deps[name.lower()] = ver
                        elif '>=' in p:
                            name, ver = p.split('>=')
                            pip_deps[name.lower()] = '>=' + ver
                        else:
                            name = p.split('[')[0].split('==')[0].split('>=')[0].strip()
                            pip_deps[name.lower()] = 'any'
                    return pip_deps
    except Exception as e:
        print(f"Error reading yaml: {e}")
    return {}

venv_pkgs = parse_freeze('venv_dp_freeze.txt')
robo_pkgs = parse_freeze('robodiff_freeze.txt')
yaml_deps = get_yaml_pip_deps('conda_environment.yaml')

print("--- Comparison: venv_dp vs conda_environment.yaml (pip) ---")
for name, target_ver in yaml_deps.items():
    if name not in venv_pkgs:
        print(f"Missing in venv_dp: {name} (Expected: {target_ver})")
    elif target_ver != 'any' and not target_ver.startswith('>=') and venv_pkgs[name] != target_ver:
         print(f"Version Mismatch in venv_dp: {name} (Found: {venv_pkgs[name]}, Expected: {target_ver})")

print("\n--- Comparison: venv_dp vs robodiff ---")
if not robo_pkgs:
    print("robodiff env unavailable or empty.")
else:
    all_names = set(venv_pkgs.keys()) | set(robo_pkgs.keys())
    for name in sorted(all_names):
        v_ver = venv_pkgs.get(name)
        r_ver = robo_pkgs.get(name)
        if v_ver != r_ver:
            print(f"{name}: venv_dp={v_ver} | robodiff={r_ver}")

