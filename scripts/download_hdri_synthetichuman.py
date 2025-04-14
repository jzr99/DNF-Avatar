import json
import os

if __name__ == "__main__":
    hdri_dir = "./load/synthetichuman/hdri"
    with open('./load/synthetichuman/leonard/hdri_files.json') as f:
        data = json.load(f)
        for i in data:
            print("Processing: {}".format(i))
            url = "https://dl.polyhaven.org/file/ph-assets/HDRIs/hdr/4k/{}".format(
                i
            )
            hdri_file = os.path.join(hdri_dir, os.path.basename(url))
            if not os.path.exists(hdri_file):
                os.system("wget {} -P {}".format(url, hdri_dir))
