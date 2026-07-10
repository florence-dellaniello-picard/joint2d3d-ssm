"""
Author: Florence Dell'Aniello Picard
Config utilities: load configuration files with proper root and patient ID.
"""

import yaml
import os

def load_config(config_path, input_override=None, output_override=None, patient_id=None):
    """Load a YAML config and resolve all {root} placeholders in string values."""
    with open(config_path) as f:
        config = yaml.safe_load(f)
    
    if input_override:
        config['input_root'] = input_override
    else:
        config['input_root']  = os.path.expandvars(config['input_root'])
    if output_override:
        config['output_root'] = output_override
    else:
        config['output_root'] = os.path.expandvars(config['output_root'])
    
    input_root  = config['input_root']
    output_root = config['output_root']
    
    if patient_id is not None:
        config['patient_id'] = patient_id

    def resolve(obj):
        if isinstance(obj, str):
            return obj.replace('{input_root}', input_root).replace('{output_root}', output_root)
        if isinstance(obj, dict):
            return {k: resolve(v) for k, v in obj.items()}
        if isinstance(obj, list):
            return [resolve(v) for v in obj]
        return obj
    return resolve(config)

