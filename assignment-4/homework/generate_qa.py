import json
from pathlib import Path

import fire
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image, ImageDraw

# Define object type mapping
OBJECT_TYPES = {
    1: "Kart",
    2: "Track Boundary",
    3: "Track Element",
    4: "Special Element 1",
    5: "Special Element 2",
    6: "Special Element 3",
}

# Define colors for different object types (RGB format)
COLORS = {
    1: (0, 255, 0),  # Green for karts
    2: (255, 0, 0),  # Blue for track boundaries
    3: (0, 0, 255),  # Red for track elements
    4: (255, 255, 0),  # Cyan for special elements
    5: (255, 0, 255),  # Magenta for special elements
    6: (0, 255, 255),  # Yellow for special elements
}

# Original image dimensions for the bounding box coordinates
ORIGINAL_WIDTH = 600
ORIGINAL_HEIGHT = 400


def extract_frame_info(image_path: str) -> tuple[int, int]:
    """
    Extract frame ID and view index from image filename.

    Args:
        image_path: Path to the image file

    Returns:
        Tuple of (frame_id, view_index)
    """
    filename = Path(image_path).name
    # Format is typically: XXXXX_YY_im.png where XXXXX is frame_id and YY is view_index
    parts = filename.split("_")
    if len(parts) >= 2:
        frame_id = int(parts[0], 16)  # Convert hex to decimal
        view_index = int(parts[1])
        return frame_id, view_index
    return 0, 0  # Default values if parsing fails


def draw_detections(
    image_path: str, info_path: str, font_scale: float = 0.5, thickness: int = 1, min_box_size: int = 5
) -> np.ndarray:
    """
    Draw detection bounding boxes and labels on the image.

    Args:
        image_path: Path to the image file
        info_path: Path to the corresponding info.json file
        font_scale: Scale of the font for labels
        thickness: Thickness of the bounding box lines
        min_box_size: Minimum size for bounding boxes to be drawn

    Returns:
        The annotated image as a numpy array
    """
    # Read the image using PIL
    pil_image = Image.open(image_path)
    if pil_image is None:
        raise ValueError(f"Could not read image at {image_path}")

    # Get image dimensions
    img_width, img_height = pil_image.size

    # Create a drawing context
    draw = ImageDraw.Draw(pil_image)

    # Read the info.json file
    with open(info_path) as f:
        info = json.load(f)

    # Extract frame ID and view index from image filename
    _, view_index = extract_frame_info(image_path)

    # Get the correct detection frame based on view index
    if view_index < len(info["detections"]):
        frame_detections = info["detections"][view_index]
    else:
        print(f"Warning: View index {view_index} out of range for detections")
        return np.array(pil_image)

    # Calculate scaling factors
    scale_x = img_width / ORIGINAL_WIDTH
    scale_y = img_height / ORIGINAL_HEIGHT

    # Draw each detection
    for detection in frame_detections:
        class_id, track_id, x1, y1, x2, y2 = detection
        class_id = int(class_id)
        track_id = int(track_id)

        if class_id != 1:
            continue

        # Scale coordinates to fit the current image size
        x1_scaled = int(x1 * scale_x)
        y1_scaled = int(y1 * scale_y)
        x2_scaled = int(x2 * scale_x)
        y2_scaled = int(y2 * scale_y)

        # Skip if bounding box is too small
        if (x2_scaled - x1_scaled) < min_box_size or (y2_scaled - y1_scaled) < min_box_size:
            continue

        if x2_scaled < 0 or x1_scaled > img_width or y2_scaled < 0 or y1_scaled > img_height:
            continue

        # Get color for this object type
        if track_id == 0:
            color = (255, 0, 0)
        else:
            color = COLORS.get(class_id, (255, 255, 255))

        # Draw bounding box using PIL
        draw.rectangle([(x1_scaled, y1_scaled), (x2_scaled, y2_scaled)], outline=color, width=thickness)

    # Convert PIL image to numpy array for matplotlib
    return np.array(pil_image)


def extract_kart_objects(
    info_path: str, view_index: int, img_width: int = 150, img_height: int = 100, min_box_size: int = 5
) -> list:
    """
    Extract kart objects from the info.json file, including their center points and identify the center kart.
    Filters out karts that are out of sight (outside the image boundaries).

    Args:
        info_path: Path to the corresponding info.json file
        view_index: Index of the view to analyze
        img_width: Width of the image (default: 150)
        img_height: Height of the image (default: 100)

    Returns:
        List of kart objects, each containing:
        - instance_id: The track ID of the kart
        - kart_name: The name of the kart
        - center: (x, y) coordinates of the kart's center
        - is_center_kart: Boolean indicating if this is the kart closest to image center
    """
    with open(info_path) as f:
        info = json.load(f)
 
    kart_names = info["karts"]
    if view_index >= len(info["detections"]):
        return []
    frame_detections = info["detections"][view_index]
 
    # scale factors from the original 600x400 frame to the working resolution
    scale_x = img_width / ORIGINAL_WIDTH
    scale_y = img_height / ORIGINAL_HEIGHT
 
    # image center used to identify the ego (center) kart
    image_center = (img_width / 2, img_height / 2)
 
    # collect on-screen karts (class_id == 1) with their scaled centers
    karts = []
    for detection in frame_detections:
        class_id, track_id, x1, y1, x2, y2 = detection
        class_id = int(class_id)
        track_id = int(track_id)
 
        if class_id != 1:
            continue
 
        x1_s = x1 * scale_x
        y1_s = y1 * scale_y
        x2_s = x2 * scale_x
        y2_s = y2 * scale_y
 
        if (x2_s - x1_s) < min_box_size or (y2_s - y1_s) < min_box_size:
            continue
 
        center_x = (x1_s + x2_s) / 2
        center_y = (y1_s + y2_s) / 2
 
        if center_x < 0 or center_x > img_width or center_y < 0 or center_y > img_height:
            continue
 
        karts.append(
            {
                "instance_id": track_id,
                "kart_name": kart_names[track_id],
                "center": (center_x, center_y),
                "is_center_kart": False,
            }
        )
 
    # mark the kart whose center is closest to the image center as the ego car
    if karts:
        ego = min(
            karts,
            key=lambda k: (k["center"][0] - image_center[0]) ** 2 + (k["center"][1] - image_center[1]) ** 2,
        )
        ego["is_center_kart"] = True
 
    return karts


def extract_track_info(info_path: str) -> str:
    """
    Extract track information from the info.json file.

    Args:
        info_path: Path to the info.json file

    Returns:
        Track name as a string
    """
    with open(info_path) as f:
            info = json.load(f)
    return info["track"]


def generate_qa_pairs(info_path: str, view_index: int, img_width: int = 150, img_height: int = 100) -> list:
    """
    Generate question-answer pairs for a given view.

    Args:
        info_path: Path to the info.json file
        view_index: Index of the view to analyze
        img_width: Width of the image (default: 150)
        img_height: Height of the image (default: 100)

    Returns:
        List of dictionaries, each containing a question and answer
    """
    # 1. Ego car question
    # What kart is the ego car?

    # 2. Total karts question
    # How many karts are there in the scenario?

    # 3. Track information questions
    # What track is this?

    # 4. Relative position questions for each kart
    # Is {kart_name} to the left or right of the ego car?
    # Is {kart_name} in front of or behind the ego car?
    # Where is {kart_name} relative to the ego car?

    # 5. Counting questions
    # How many karts are to the left of the ego car?
    # How many karts are to the right of the ego car?
    # How many karts are in front of the ego car?
    # How many karts are behind the ego car?
    karts = extract_kart_objects(info_path, view_index, img_width, img_height)
    track_name = extract_track_info(info_path)
 
    qa_pairs = []
 
    if not karts:
        return qa_pairs
 
    # locate the ego kart to use as the reference frame
    ego = next((k for k in karts if k["is_center_kart"]), None)
    if ego is None:
        return qa_pairs
    ego_x, ego_y = ego["center"]
 
    # ego car identity
    qa_pairs.append({"question": "What kart is the ego car?", "answer": ego["kart_name"]})
 
    # total kart count
    qa_pairs.append(
        {"question": "How many karts are there in the scenario?", "answer": str(len(karts))}
    )
 
    qa_pairs.append({"question": "What track is this?", "answer": track_name})
 
    # counters for the directional counting questions
    left_count = right_count = front_count = back_count = 0
 
    # per-kart relative position questions (skip the ego itself)
    for kart in karts:
        if kart["is_center_kart"]:
            continue
 
        kx, ky = kart["center"]
        name = kart["kart_name"]
 
        lr = "left" if kx < ego_x else "right"
        fb = "front" if ky < ego_y else "back"
 
        if lr == "left":
            left_count += 1
        else:
            right_count += 1
        if fb == "front":
            front_count += 1
        else:
            back_count += 1
 
        qa_pairs.append(
            {"question": f"Is {name} to the left or right of the ego car?", "answer": lr}
        )
        qa_pairs.append(
            {"question": f"Is {name} in front of or behind the ego car?", "answer": fb}
        )
        qa_pairs.append(
            {
                "question": f"Where is {name} relative to the ego car?",
                "answer": f"{fb} and {lr}",
            }
        )
 
    # directional counting questions
    qa_pairs.append(
        {"question": "How many karts are to the left of the ego car?", "answer": str(left_count)}
    )
    qa_pairs.append(
        {"question": "How many karts are to the right of the ego car?", "answer": str(right_count)}
    )
    qa_pairs.append(
        {"question": "How many karts are in front of the ego car?", "answer": str(front_count)}
    )
    qa_pairs.append(
        {"question": "How many karts are behind the ego car?", "answer": str(back_count)}
    )
 
    return qa_pairs


def check_qa_pairs(info_file: str, view_index: int):
    """
    Check QA pairs for a specific info file and view index.

    Args:
        info_file: Path to the info.json file
        view_index: Index of the view to analyze
    """
    # Find corresponding image file
    info_path = Path(info_file)
    base_name = info_path.stem.replace("_info", "")
    image_file = list(info_path.parent.glob(f"{base_name}_{view_index:02d}_im.jpg"))[0]

    # Visualize detections
    annotated_image = draw_detections(str(image_file), info_file)

    # Display the image
    plt.figure(figsize=(12, 8))
    plt.imshow(annotated_image)
    plt.axis("off")
    plt.title(f"Frame {extract_frame_info(str(image_file))[0]}, View {view_index}")
    plt.show()

    # Generate QA pairs
    qa_pairs = generate_qa_pairs(info_file, view_index)

    # Print QA pairs
    print("\nQuestion-Answer Pairs:")
    print("-" * 50)
    for qa in qa_pairs:
        print(f"Q: {qa['question']}")
        print(f"A: {qa['answer']}")
        print("-" * 50)

def generate(output_dir: str = "data/train", img_width: int = 150, img_height: int = 100):
    # build the full VLM training set from every train info file and view
    data_root = Path(__file__).parent.parent
    train_dir = data_root / "data" / "train"
    out_dir = data_root / output_dir
    out_dir.mkdir(parents=True, exist_ok=True)
 
    info_files = sorted(train_dir.glob("*_info.json"))
    print(f"Found {len(info_files)} info files in {train_dir}")
 
    for info_file in info_files:
        with open(info_file) as f:
            info = json.load(f)
        num_views = len(info["detections"])
 
        all_pairs = []
        base_name = info_file.stem.replace("_info", "")
        for view_index in range(num_views):
            image_rel = f"train/{base_name}_{view_index:02d}_im.jpg"
            if not (data_root / "data" / image_rel).exists():
                continue
            for qa in generate_qa_pairs(str(info_file), view_index, img_width, img_height):
                qa["image_file"] = image_rel
                all_pairs.append(qa)
 
        if all_pairs:
            out_path = out_dir / f"{base_name}_qa_pairs.json"
            with open(out_path, "w") as f:
                json.dump(all_pairs, f, indent=2)
 
    print(f"Wrote QA pair files to {out_dir}")

"""
Usage Example: Visualize QA pairs for a specific file and view:
   python generate_qa.py check --info_file ../data/valid/00000_info.json --view_index 0

You probably need to add additional commands to Fire below.
"""


def main():
    fire.Fire({"check": check_qa_pairs, "generate": generate})
 
if __name__ == "__main__":
    main()
