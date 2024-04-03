#参考 https://github.com/Hangover3832 把模型放到本地，checkponts下面

import os
from transformers import AutoModelForCausalLM, CodeGenTokenizerFast as Tokenizer
from PIL import Image
import torch
import gc
import numpy as np
import folder_paths

comfy_path = os.path.dirname(folder_paths.__file__)
custom_nodes_path = os.path.join(comfy_path, "custom_nodes")

# 指定本地分割模型文件夹的路径
model_folder_path = os.path.join(custom_nodes_path,"Comfyui_CXH_moondream2","checkpoints","moondream2")
model_name = "vikhyatk/moondream2"

class Moondream:
    DEVICES = ["cpu", "gpu"] if torch.cuda.is_available() else  ["cpu"]

    def __init__(self):
        self.model = None
        self.tokenizer = None

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "image": ("IMAGE",),
                "prompt": ("STRING", {"multiline": False, "default": "Please provide a detailed description of this image."},),
                "device": (s.DEVICES, {"default": s.DEVICES[1]},),
                "trust_remote_code": ("BOOLEAN", {"default": True},),
            }
        }

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("description",)
    FUNCTION = "gen"
    OUTPUT_NODE = False
    CATEGORY = "CXH"

    def gen(self, image:torch.Tensor, prompt:str,  device:str, trust_remote_code:bool):
        dev = "cuda" if device.lower() == "gpu" else "cpu"
        if (self.model == None) or (self.tokenizer == None)  or (device != self.device):
            del self.model
            del self.tokenizer
            gc.collect()
            if (device == "cpu") and torch.cuda.is_available():
                torch.cuda.empty_cache()
            self.model = None
            self.tokenizer = None
            try:
                self.model = AutoModelForCausalLM.from_pretrained(model_name, trust_remote_code=trust_remote_code,cache_dir=model_folder_path).to(dev)
            except ValueError:
                print("Moondream: You have to trust remote code to use this node!")
                return ("You have to trust remote code execution to use this node!",)
            
            self.tokenizer = Tokenizer.from_pretrained(model_name,cache_dir=model_folder_path)
            self.device = device

        descriptions = ""
        
        for im in image:
            i = 255. * im.cpu().numpy()
            img = Image.fromarray(np.clip(i, 0, 255).astype(np.uint8))
            enc_image = self.model.encode_image(img)
            answer = self.model.answer_question(enc_image, prompt, self.tokenizer)
            descriptions += answer
        
        return(descriptions,)
