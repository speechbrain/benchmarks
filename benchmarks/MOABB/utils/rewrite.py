#!/usr/bin/python3
"""
Yaml file rewriter to add orion tags to the best hparams file from original yaml file.
Author
------
Victor Cruz, 20224
"""
import argparse
import yaml
import re

def readargs():
    parser = argparse.ArgumentParser()
    parser.add_argument("original_yaml_file", type=str, help="Original yaml file")
    parser.add_argument("best_hparams_file", type=str, help="Best hparams file")
    args = parser.parse_args()

    # Check if the file paths are valid
    if not args.original_yaml_file.endswith(".yaml"):
        raise ValueError("Original yaml file must be a yaml file")
    if not args.best_hparams_file.endswith(".yaml"):
        raise ValueError("Best hparams file must be a yaml file")
    return args

def extract_orion_tags(original_yaml_file):
    """
    Function to extract orion tags and variable names from the original yaml file.
    Orion tags are comments that start with '# @orion_step<stepid>'.
    """
    orion_tags = {}
    tag_pattern = re.compile(r"# @orion_step(\d+):\s*(.*)")
    
    with open(original_yaml_file, "r") as og_f:
        for line in og_f:
            # Extract lines that contain Orion tags
            tag_match = tag_pattern.search(line.strip())
            if tag_match:
                variable_name = line.split(":")[0].strip()  # Get the variable name before ":"
                tag_info = tag_match.group(0)  # Full tag line
                orion_tags[variable_name] = tag_info  # Store variable and tag info
    return orion_tags

def rewrite_with_orion_tags(original_yaml_file, best_hparams_file):
    """
    Function to add orion tags to the best hparams file.
    Matches based on the variable name from the original file to the target file.
    """
    orion_tags = extract_orion_tags(original_yaml_file)

    # Read the best_hparams YAML file
    with open(best_hparams_file, "r") as best_f:
        best_hparams_lines = best_f.readlines()

    # Add orion tags to the appropriate lines in the new file
    new_best_hparams_lines = []
    for line in best_hparams_lines:
        stripped_line = line.strip()
        # Extract variable name from the line in the best hparams file
        if ":" in stripped_line:
            variable_name = stripped_line.split(":")[0].strip()

            # Check if this variable has a corresponding orion tag
            if variable_name in orion_tags:
                # Append the orion tag to the same line, ensuring there's a space before the comment
                line = line.rstrip() + " " + orion_tags[variable_name] + "\n"
                new_best_hparams_lines.append(line)
            else:
                new_best_hparams_lines.append(line)
        else:
            new_best_hparams_lines.append(line)

    # Write the modified content back to the best_hparams file
    with open(best_hparams_file, "w") as best_f:
        best_f.writelines(new_best_hparams_lines)

if __name__ == "__main__":
    args = readargs()
    rewrite_with_orion_tags(args.original_yaml_file, args.best_hparams_file)
