from yaml import dump, Dumper
from builder import make_last_traj, make_neat_network_diagram


# Disable jit
with open('.numba_config.yaml', 'w') as f:
    lines = {'DISABLE_JIT': 1}
    dump(lines, f, Dumper=Dumper)

# NOTE: make sure the environment variable NUMBA_DISABLE_JIT is set to 1
_config_name = 'coarse'
# _config_name = 'intermediate'
# _config_name = 'final'

make_neat_network_diagram(config_name=_config_name)
make_last_traj(config_name=_config_name)

# Re-enable jit
with open('.numba_config.yaml', 'w') as f:
    lines = {'DISABLE_JIT': 0}
    dump(lines, f, Dumper=Dumper)
