import json
import random
from pathlib import Path

def train_valid_split(json_file_dir):
    # Load json file
    with open(json_file_dir, "r") as file:
        data = json.load(file)
    # Create a dictionary of objects with the same name and scale
    collected_objs = {}
    for object_id, object_data in data.items():
        object_name = object_data["object_name"]
        scale = object_data["scale"]
        corres_obj = str(object_name) + "_" + str(scale)
        if corres_obj not in collected_objs.keys():
            collected_objs[corres_obj] = [object_id]
        else:
            collected_objs[corres_obj].append(object_id)
    # Split the objects into train and valid sets (Choose 1 object from each key in collected_objs for valid set)
    train_objs = []
    valid_objs = []
    excluded_valid_objs = ["7187", "45087", "45606"]
    for corres_obj, object_ids in collected_objs.items():
        if corres_obj.split("_")[0] not in excluded_valid_objs:
            valid_objs.append(random.choice(object_ids))
        # let other objects be in the train set
        for object_id in object_ids:
            if object_id not in valid_objs:
                train_objs.append(object_id)
    # Create a json file with train and valid objs
    train_valid_objs = {"train": train_objs, "valid": valid_objs}
    output_file_path = "decoder_data_split.json"
    with open(output_file_path, "w") as output_file:
        json.dump(train_valid_objs, output_file, indent=4)
    print(f"Saved train and valid objects to {output_file_path}")
    return


if __name__ == "__main__":

    dataset_dir = Path.cwd() / "datasets"
    json_file_dir = Path.cwd() / "configs" / "object_configurations.json"
    train_valid_split(json_file_dir)