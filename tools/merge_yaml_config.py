#!/usr/bin/env python3
"""Deep-merge two YAML config files: existing (user) + default (repo).
Existing values win; missing keys from defaults are added.

Usage: merge_yaml_config.py --existing <path> --default <path> --output <path>
"""

import argparse
import sys
from copy import deepcopy

try:
    import yaml
except ImportError:
    yaml = None


def deep_merge(existing, defaults):
    """Recursive dict merge: existing values take priority, missing keys from defaults are added."""
    if not isinstance(existing, dict) or not isinstance(defaults, dict):
        return existing
    result = deepcopy(defaults)
    for k, v in existing.items():
        if k in result and isinstance(result[k], dict) and isinstance(v, dict):
            result[k] = deep_merge(v, result[k])
        else:
            result[k] = deepcopy(v)
    return result


def main():
    parser = argparse.ArgumentParser(description="Merge YAML configs, preserving existing values")
    parser.add_argument("--existing", required=True, help="Existing user config file")
    parser.add_argument("--default", required=True, help="New default config file from repo")
    parser.add_argument("--output", required=True, help="Output merged config file")
    args = parser.parse_args()

    if yaml is None:
        print("ERROR: PyYAML not available", file=sys.stderr)
        sys.exit(1)

    with open(args.existing) as f:
        existing = yaml.safe_load(f) or {}
    with open(args.default) as f:
        defaults = yaml.safe_load(f) or {}

    merged = deep_merge(existing, defaults)

    with open(args.output, "w") as f:
        yaml.dump(merged, f, default_flow_style=False, allow_unicode=True, sort_keys=False)

    print(f"Merged config written to {args.output}")


if __name__ == "__main__":
    main()
