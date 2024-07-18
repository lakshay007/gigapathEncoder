import os
import glob
from gigapath.pipeline import load_tile_slide_encoder, run_inference_with_tile_encoder, run_inference_with_slide_encoder


os.environ["HF_TOKEN"] = ""


svs_file_path = "/Users/lakshaychauhan/miniconda3/envs/gigapath/tissueimg.svs"


tile_dir = "."
tile_size = 256
overlap = 0

def load_existing_tiles(tile_dir):

    image_paths = glob.glob(os.path.join(tile_dir, "*.png"))
    return image_paths

try:

    image_paths = load_existing_tiles(tile_dir)
    print(f"Found {len(image_paths)} image tiles")


    tile_encoder, slide_encoder_model = load_tile_slide_encoder()


    tile_encoder_outputs = run_inference_with_tile_encoder(image_paths, tile_encoder)


    for k in tile_encoder_outputs.keys():
        print(f"tile_encoder_outputs[{k}].shape: {tile_encoder_outputs[k].shape}")


    slide_embeds = run_inference_with_slide_encoder(slide_encoder_model=slide_encoder_model, **tile_encoder_outputs)
    print(slide_embeds.keys())

except Exception as e:
    print(f"An error occurred: {e}")
