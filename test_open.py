from unittest import TestCase

import timm
import os

os.environ["HF_TOKEN"] = ""
model = timm.create_model("hf_hub:prov-gigapath/prov-gigapath", pretrained=True)


class Test(TestCase):
    def test_load_existing_tiles(self):
        self.fail()
