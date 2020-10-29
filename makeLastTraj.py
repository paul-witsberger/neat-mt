import yaml
# Disable jit
with open('.numba_config.yaml', 'w') as f:
    lines = {'DISABLE_JIT': 1}
    yaml.dump(lines, f, Dumper=yaml.Dumper)

# NOTE: make sure the environment variable NUMBA_DISABLE_JIT is set to 1
from builder import make_last_traj, make_neat_network_diagram
make_last_traj()
make_neat_network_diagram()

# Re-enable jit
with open('.numba_config.yaml', 'w') as f:
    lines = {'DISABLE_JIT': 0}
    yaml.dump(lines, f, Dumper=yaml.Dumper)
