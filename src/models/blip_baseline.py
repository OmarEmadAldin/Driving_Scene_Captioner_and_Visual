"""
blip_baseline.py
-----------------
Thin wrapper around Salesforce/blip2-opt-2.7b (BLIP-2), used here purely as
an image captioner -- no VQA/grounding conditions, no tracker context.

Hardware note: blip2-opt-2.7b is a ~3.6B-param model in total (vision
encoder + Q-Former + OPT-2.7b decoder) -- in fp16 that's ~7.2GB, too big for
a 4GB card. `load_in_4bit` (default true in configs/config.yaml) uses
bitsandbytes to bring weights down to ~1.8GB, which actually leaves headroom
on a 4GB card; set `load_in_8bit` instead if you have more like 6GB+, or
set both false if you have >=8GB VRAM.
"""
import torch
from PIL import Image
from transformers import Blip2ForConditionalGeneration, Blip2Processor, BitsAndBytesConfig


class BlipBaseline:
    def __init__(self, cfg: dict):
        m_cfg = cfg["models"]["blip2"]
        self.device = m_cfg["device"] if torch.cuda.is_available() else "cpu"
        load_in_8bit = bool(m_cfg.get("load_in_8bit")) and self.device == "cuda"
        load_in_4bit = bool(m_cfg.get("load_in_4bit")) and self.device == "cuda"
        dtype = torch.float16 if (self.device == "cuda" and m_cfg.get("dtype") == "float16") else torch.float32

        quant_config = None
        if load_in_4bit:
            quant_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=torch.float16,
                bnb_4bit_quant_type="nf4",
            )
        elif load_in_8bit:
            quant_config = BitsAndBytesConfig(load_in_8bit=True)

        self.processor = Blip2Processor.from_pretrained(m_cfg["name"])

        load_kwargs = dict(torch_dtype=dtype)
        if quant_config is not None:
            load_kwargs["quantization_config"] = quant_config
            load_kwargs["device_map"] = {"": 0}
        self.model = Blip2ForConditionalGeneration.from_pretrained(m_cfg["name"], **load_kwargs)
        if quant_config is None:
            self.model = self.model.to(self.device)
        self.model.eval()

        self.dtype = dtype
        self.quantized = quant_config is not None

    def _to_device(self, inputs):
        # Quantized models already pin their layers via device_map, so only
        # the pixel/text inputs need moving; dtype cast only applies to the
        # non-quantized path (bitsandbytes handles its own dtypes).
        if self.quantized:
            return inputs.to(self.device)
        return inputs.to(self.device, self.dtype)

    @torch.no_grad()
    def caption(self, image: Image.Image, max_new_tokens: int = 40) -> str:
        inputs = self.processor(images=image, return_tensors="pt")
        inputs = self._to_device(inputs)
        out = self.model.generate(**inputs, max_new_tokens=max_new_tokens)
        return self.processor.batch_decode(out, skip_special_tokens=True)[0].strip()
