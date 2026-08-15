import json
from pathlib import Path

import fire
from matplotlib import pyplot as plt

from .generate_qa import draw_detections, extract_frame_info, extract_kart_objects, extract_track_info


def generate_caption(info_path: str, view_index: int, img_width: int = 150, img_height: int = 100) -> list:
    """
    Generate caption for a specific view.
    """
    # 1. Ego car
    # {kart_name} is the ego car.

    # 2. Counting
    # There are {num_karts} karts in the scenario.

    # 3. Track name
    # The track is {track_name}.

    # 4. Relative position
    # {kart_name} is {position} of the ego car.
    karts = extract_kart_objects(info_path, view_index, img_width, img_height)
    track_name = extract_track_info(info_path)
 
    captions = []
    if not karts:
        return captions
 
    # locate ego kart for the reference frame
    ego = next((k for k in karts if k["is_center_kart"]), None)
    if ego is None:
        return captions
    ego_x, ego_y = ego["center"]
 
    # ego car caption
    captions.append(f"{ego['kart_name']} is the ego car.")
 
    # kart count caption
    captions.append(f"There are {len(karts)} karts in the scene.")
 
    captions.append(f"The track is {track_name}.")
 
    # relative position caption for each non-ego kart
    for kart in karts:
        if kart["is_center_kart"]:
            continue
        kx, ky = kart["center"]
        lr = "left" if kx < ego_x else "right"
        fb = "front" if ky < ego_y else "back"
        captions.append(f"{kart['kart_name']} is {fb} and {lr} of the ego car.")
 
    return captions


def check_caption(info_file: str, view_index: int):
    captions = generate_caption(info_file, view_index)

    print("\nCaption:")
    print("-" * 50)
    for i, caption in enumerate(captions):
        print(f"{i + 1}. {caption}")
        print("-" * 50)

    info_path = Path(info_file)
    base_name = info_path.stem.replace("_info", "")
    image_file = list(info_path.parent.glob(f"{base_name}_{view_index:02d}_im.jpg"))[0]

    annotated_image = draw_detections(str(image_file), info_file)

    plt.figure(figsize=(12, 8))
    plt.imshow(annotated_image)
    plt.axis("off")
    plt.title(f"Frame {extract_frame_info(str(image_file))[0]}, View {view_index}")
    plt.show()

def generate(output_dir: str = "data/train", img_width: int = 150, img_height: int = 100):
    # build the full CLIP caption set from every train info file and view
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
 
        entries = []
        base_name = info_file.stem.replace("_info", "")
        for view_index in range(num_views):
            image_rel = f"train/{base_name}_{view_index:02d}_im.jpg"
            if not (data_root / "data" / image_rel).exists():
                continue
            for caption in generate_caption(str(info_file), view_index, img_width, img_height):
                entries.append({"image_file": image_rel, "caption": caption})
 
        if entries:
            out_path = out_dir / f"{base_name}_captions.json"
            with open(out_path, "w") as f:
                json.dump(entries, f, indent=2)
 
    print(f"Wrote caption files to {out_dir}")

"""
Usage Example: Visualize QA pairs for a specific file and view:
   python generate_captions.py check --info_file ../data/valid/00000_info.json --view_index 0

You probably need to add additional commands to Fire below.
"""


def main():
    fire.Fire({"check": check_caption, "generate": generate})


if __name__ == "__main__":
    main()
