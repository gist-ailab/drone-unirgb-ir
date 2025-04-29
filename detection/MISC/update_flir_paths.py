import json
import os
import ntpath # Using ntpath for os-agnostic basename extraction

# Define input and output paths
input_json_path = '../../../dset/Drone-Detection-Benchmark/FLIR_aligned_unirgbir/Annotation_train.json'
output_json_path = '../../../dset/Drone-Detection-Benchmark/FLIR_aligned_unirgbir/Annotation_train_updated.json'

print(f"Loading annotations from: {input_json_path}")

# Load the original JSON data
try:
    with open(input_json_path, 'r') as f:
        data = json.load(f)
except FileNotFoundError:
    print(f"Error: Input file not found at {input_json_path}")
    exit(1)
except json.JSONDecodeError:
    print(f"Error: Could not decode JSON from {input_json_path}")
    exit(1)

print("Successfully loaded JSON data.")

# Check if 'images' key exists
if 'images' not in data:
    print("Error: 'images' key not found in the JSON data.")
    exit(1)

print(f"Processing {len(data['images'])} images...")

# Iterate through images and update paths
updated_count = 0
skipped_count = 0
for img_info in data['images']:
    original_path_source = None
    if 'img_path' in img_info and img_info['img_path']: # Check if img_path exists and is not empty
        original_path_source = img_info['img_path']
        source_key = 'img_path'
    elif 'file_name' in img_info and img_info['file_name']: # Fallback to file_name
        original_path_source = img_info['file_name']
        source_key = 'file_name'
    else:
        print(f"Warning: Neither 'img_path' nor 'file_name' key found or key is empty for image ID {img_info.get('id', 'N/A')}. Skipping.")
        skipped_count += 1
        continue

    # Extract the base ID (e.g., '08868' from 'FLIR_08868.jpeg' or 'folder/FLIR_08868.jpeg')
    try:
        # Use ntpath.basename to handle potential directory paths
        filename = ntpath.basename(original_path_source)
        # Assuming the format is still consistent like 'FLIR_xxxxx.ext'
        parts = filename.split('_')
        if len(parts) > 1:
             base_id = parts[1].split('.')[0]
             # Construct new paths
             new_rgb_path = f"{base_id}.png"
             new_ir_path = f"{base_id}_ir.png"

             # Update the paths in the dictionary
             img_info['img_path'] = new_rgb_path
             img_info['img_ir_path'] = new_ir_path # Use the correct key 'img_ir_path'
             updated_count += 1
        else:
             print(f"Warning: Could not parse base ID from {source_key}: '{original_path_source}' (format unexpected). Skipping entry for image ID {img_info.get('id', 'N/A')}.")
             skipped_count += 1

    except Exception as e:
        print(f"Warning: An error occurred processing {source_key} '{original_path_source}' for image ID {img_info.get('id', 'N/A')}: {e}. Skipping this entry.")
        skipped_count += 1


print(f"Finished processing. Updated paths for {updated_count} images. Skipped {skipped_count} entries.")

# Save the modified data to the new JSON file
print(f"Saving updated annotations to: {output_json_path}")
try:
    # Ensure the output directory exists (though relative paths should be fine here)
    # output_dir = os.path.dirname(output_json_path)
    # if output_dir:
    #     os.makedirs(output_dir, exist_ok=True)

    with open(output_json_path, 'w') as f:
        json.dump(data, f, indent=4) # Use indent for readability
    print("Successfully saved updated JSON data.")
except IOError as e:
    print(f"Error: Could not write to output file {output_json_path}. Error: {e}")
    exit(1)
except Exception as e:
    print(f"An unexpected error occurred during saving: {e}")
    exit(1)


print("Script finished.")
