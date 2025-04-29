import json
import cv2
import numpy as np
import os
import argparse
from tqdm import tqdm

def visualize_coco_rgbt(json_path, data_root, data_prefix):
    """
    Visualizes bounding boxes from a COCO format JSON file on concatenated RGB and IR images.

    Args:
        json_path (str): Path to the COCO format annotation JSON file.
        data_root (str): Root directory of the dataset.
        data_prefix (str): Prefix for the specific data split (e.g., 'train', 'test').
    """
    try:
        with open(json_path, 'r') as f:
            coco_data = json.load(f)
    except FileNotFoundError:
        print(f"Error: JSON file not found at {json_path}")
        return
    except json.JSONDecodeError:
        print(f"Error: Could not decode JSON from {json_path}")
        return

    images = coco_data.get('images', [])
    annotations = coco_data.get('annotations', [])
    categories = {cat['id']: cat['name'] for cat in coco_data.get('categories', [])}

    # Create a dictionary to map image_id to annotations
    image_id_to_anns = {}
    for ann in annotations:
        img_id = ann['image_id']
        if img_id not in image_id_to_anns:
            image_id_to_anns[img_id] = []
        image_id_to_anns[img_id].append(ann)

    print(f"Found {len(images)} images and {len(annotations)} annotations.")
    print(f"Categories: {categories}")
    print("Press 'q' to quit, any other key to view the next image (first 50 images).")

    output_dir = os.path.join(os.path.dirname(json_path), f'{data_prefix}_visualizations')
    os.makedirs(output_dir, exist_ok=True)
    print(f"Saving visualizations to: {output_dir}")

    vis_count = 0
    max_vis = 50 # Limit the number of visualizations shown interactively

    for img_info in tqdm(images):
        image_id = img_info['id']
        file_name = img_info['img_path'] # e.g., "video_00001/FLIR_00001.jpg" # Changed key from 'file_name'

        # Construct paths assuming file_name structure and separate rgb/ir folders
        # Modify this logic if your file structure is different
        rgb_path = os.path.join(data_root, data_prefix ,file_name)
        ir_path = os.path.join(data_root, data_prefix, img_info['img_ir_path'])

        # Check if paths exist before trying to read
        if not os.path.exists(rgb_path):
            # Try alternative common naming convention if the first fails (jpg vs jpeg)
            alt_file_name = file_name.replace('.jpeg', '.jpg') if '.jpeg' in file_name else file_name.replace('.jpg', '.jpeg')
            rgb_path_alt = os.path.join(data_root, data_prefix, 'rgb', alt_file_name)
            ir_path_alt = os.path.join(data_root, data_prefix, 'ir', alt_file_name)
            if os.path.exists(rgb_path_alt):
                 rgb_path = rgb_path_alt
                 ir_path = ir_path_alt # Assume if alt rgb exists, alt ir also uses same extension
            else:
                 print(f"Warning: RGB image not found at {rgb_path} or {rgb_path_alt}. Skipping image ID {image_id}.")
                 continue # Skip if primary and alternative RGB path don't exist

        if not os.path.exists(ir_path):
             print(f"Warning: IR image not found at {ir_path}. Trying alternative extension based on RGB.")
             # If we used alt RGB path, alt IR path is already set. If not, try alt IR path now.
             if 'alt_file_name' in locals() and os.path.exists(os.path.join(data_root, data_prefix, 'ir', alt_file_name)):
                 ir_path = os.path.join(data_root, data_prefix, 'ir', alt_file_name)
             else:
                 print(f"Warning: IR image not found at {ir_path} or alternative. Skipping image ID {image_id}.")
                 continue # Skip if IR path doesn't exist

        # Load images
        img_rgb = cv2.imread(rgb_path)
        img_ir = cv2.imread(ir_path)

        if img_rgb is None:
            print(f"Warning: Failed to load RGB image from {rgb_path}. Skipping image ID {image_id}.")
            continue
        if img_ir is None:
            print(f"Warning: Failed to load IR image from {ir_path}. Skipping image ID {image_id}.")
            continue

        # Ensure IR image is 3-channel BGR for concatenation
        if len(img_ir.shape) == 2:
            img_ir = cv2.cvtColor(img_ir, cv2.COLOR_GRAY2BGR)
        elif img_ir.shape[2] == 1:
             img_ir = cv2.cvtColor(img_ir, cv2.COLOR_GRAY2BGR)

        # Ensure images have the same height for horizontal concatenation
        h_rgb, w_rgb = img_rgb.shape[:2]
        h_ir, w_ir = img_ir.shape[:2]

        if h_rgb != h_ir:
            print(f"Warning: Image heights differ (RGB: {h_rgb}, IR: {h_ir}) for image ID {image_id}. Resizing IR to match RGB height.")
            target_height = h_rgb
            scale = target_height / h_ir
            new_w_ir = int(w_ir * scale)
            img_ir = cv2.resize(img_ir, (new_w_ir, target_height), interpolation=cv2.INTER_LINEAR)
            w_ir = new_w_ir # Update width after resize

        # Concatenate images horizontally
        vis_image = np.hstack((img_rgb, img_ir))

        # Draw bounding boxes
        if image_id in image_id_to_anns:
            for ann in image_id_to_anns[image_id]:
                bbox = ann['bbox'] # [x, y, width, height]
                category_id = ann['category_id']
                category_name = categories.get(category_id, 'Unknown')

                x, y, w, h = map(int, bbox)

                # Draw on RGB part
                cv2.rectangle(vis_image, (x, y), (x + w, y + h), (0, 255, 0), 2) # Green box
                cv2.putText(vis_image, category_name, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

                # Draw on IR part (offset x by width of RGB image)
                ir_x = x + w_rgb
                cv2.rectangle(vis_image, (ir_x, y), (ir_x + w, y + h), (0, 0, 255), 2) # Red box
                cv2.putText(vis_image, category_name, (ir_x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)

        # Save the visualization
        # Sanitize file_name for saving (replace slashes)
        safe_file_name = file_name.replace('/', '_')
        save_filename = f"vis_{image_id}_{safe_file_name}"
        save_path = os.path.join(output_dir, save_filename)
        try:
            cv2.imwrite(save_path, vis_image)
        except Exception as e:
            print(f"Error saving image {save_path}: {e}")


        # Display image (optional, limited number)
        if vis_count < max_vis:
            display_title = f'RGB(L)/IR(R) | ID:{image_id} | File:{file_name} | Press Q-Quit, Oth-Next'
            # Resize for display if too large
            max_display_width = 1800 # Adjust as needed
            current_width = vis_image.shape[1]
            if current_width > max_display_width:
                scale_factor = max_display_width / current_width
                display_height = int(vis_image.shape[0] * scale_factor)
                display_image = cv2.resize(vis_image, (max_display_width, display_height))
            else:
                display_image = vis_image

            cv2.imshow(display_title, display_image)
            key = cv2.waitKey(0) # Wait indefinitely for a key press
            cv2.destroyWindow(display_title) # Close the specific window
            if key == ord('q') or key == ord('Q'):
                print("Quit signal received.")
                break # Exit loop if 'q' is pressed
        vis_count += 1
        if vis_count == max_vis:
            print(f"Reached maximum interactive visualizations ({max_vis}). Continuing to save remaining images...")


    cv2.destroyAllWindows() # Ensure all OpenCV windows are closed
    print(f"Finished processing. Visualizations saved in {output_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Visualize COCO RGBT annotations.')
    parser.add_argument('--json_path', type=str, required=True,
                        help='Path to the COCO format annotation JSON file.')
    parser.add_argument('--data_root', type=str, required=True,
                        help='Root directory of the dataset (e.g., /path/to/FLIR_aligned_unirgbir).')
    parser.add_argument('--data_prefix', type=str, required=True,
                        help="Prefix for the data split (e.g., 'test', 'train').")

    args = parser.parse_args()

    visualize_coco_rgbt(args.json_path, args.data_root, args.data_prefix)
