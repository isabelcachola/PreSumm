#! /usr/bin/env python

# Script to launch AllenNLP Beaker jobs.

import argparse
import os
import json
import random
import tempfile
import subprocess
import sys
import _jsonnet

# This has to happen before we import spacy (even indirectly), because for some crazy reason spacy
# thought it was a good idea to set the random seed on import...
random_int = random.randint(0, 2**32)

sys.path.insert(0, os.path.dirname(os.path.abspath(os.path.join(os.path.join(__file__, os.pardir), os.pardir))))

def main(args):
    # Do steps (1-3) manually
    print(f"Using the specified image: {args.image}")

    run_command = [
            "python",
            "src/exp.py"
            ]

    for t in args.targ:
        x,y = t.strip().split('=')
        run_command.append(f'-{x}')
        run_command.append(y)

    # (4) Automatically make spec file
    # For each dataset, mount to docker container
    dataset_mounts = []
    for source in args.source:
        datasetID, containerPath = source.split(':')
        dataset_mounts.append({
            "datasetId": datasetID,
            "containerPath": containerPath
        })

    env = {}

    # Override defaults
    for var in args.env:
        key, value = var.split("=")
        env[key] = value

    config_spec = {
        "image": args.image,
        "description": args.desc,
        "resultPath": "/output",
        "args": run_command,
        "datasetMounts": dataset_mounts,
        "env": env
    }
    config_task = {"spec": config_spec, "name": "training"}

    if args.cluster:
        config_task["cluster"] = args.cluster

    config = {
        "tasks": [config_task]
    }

    output_path = args.spec_output_path if args.spec_output_path else tempfile.mkstemp(".yaml",
            "beaker-config-")[1]
    with open(output_path, "w") as output:
        output.write(json.dumps(config, indent=4))
    print(f"Beaker spec written to {output_path}.")

    # (5) Create beaker experiment
    experiment_command = ["beaker", "experiment", "create", "--file", output_path]

    if args.dry_run:
        print(f"This is a dry run (--dry-run).  Launch your job with the following command:")
        print(f"    " + " ".join(experiment_command))
    else:
        print(f"Running the experiment:")
        print(f"    " + " ".join(experiment_command))
        subprocess.run(experiment_command)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument('image', type=str, help='Docker image to use')
    parser.add_argument('--cluster', type=str, help='A name for the experiment.')
    parser.add_argument('--desc', type=str, default="", help='A description for the experiment.')
    parser.add_argument('--targ', action='append', default=[], help='Training args (e.g. lr=0.1)')
    parser.add_argument('--config_file', type=str, default='default_beaker_configs.jsonnet', help='Path to json config file')
    parser.add_argument('--spec_output_path', type=str, help='The destination to write the experiment spec.')
    parser.add_argument('--dry-run', action='store_true', help='If specified, an experiment will not be created.')
    parser.add_argument('--source', action='append', default=[], help='Bind a remote data source (e.g. source-id:/target/path)')
    parser.add_argument('--env', action='append', default=[], help='Set environment variables (e.g. NAME=value or NAME)')
    args = parser.parse_args()

    main(args)