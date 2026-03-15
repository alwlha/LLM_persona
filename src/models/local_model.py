import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
from .base import BaseModel


class LocalModel(BaseModel):
    """
    在本地加载并运行的开源模型。
    自动检测 CUDA / MPS / CPU，按优先级选择推理后端。
    """

    def __init__(self, model_path: str, model_name: str | None = None):
        self._name = model_name or model_path.split("/")[-1]
        print(f"[LocalModel] Loading '{self._name}' from: {model_path}")

        if torch.cuda.is_available():
            self.device = "cuda"
            dtype = torch.float16
        elif torch.backends.mps.is_available():
            self.device = "mps"
            dtype = torch.float16
        else:
            self.device = "cpu"
            dtype = torch.float32

        self.tokenizer = AutoTokenizer.from_pretrained(
            model_path, trust_remote_code=True
        )
        self.model = AutoModelForCausalLM.from_pretrained(
            model_path,
            torch_dtype=dtype,
            device_map=self.device,
            trust_remote_code=True,
        )
        self.model.eval()
        print(f"[LocalModel] '{self._name}' loaded on {self.device}.")

    @property
    def name(self) -> str:
        return self._name

    def query(self, prompt: str, system: str | None = None) -> str:
        # 优先使用 chat template（Llama/Qwen 等 Instruct 模型均支持）
        if self.tokenizer.chat_template:
            messages = []
            if system:
                messages.append({"role": "system", "content": system})
            messages.append({"role": "user", "content": prompt})
            full_prompt = self.tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )
        else:
            full_prompt = f"{system}\n\n{prompt}" if system else prompt

        inputs = self.tokenizer(full_prompt, return_tensors="pt").to(self.device)
        input_len = inputs["input_ids"].shape[-1]

        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=64,
                temperature=0.1,
                do_sample=True,
                pad_token_id=self.tokenizer.eos_token_id,
            )

        # 只解码新生成的 token，避免复读 prompt
        new_tokens = outputs[0][input_len:]
        return self.tokenizer.decode(new_tokens, skip_special_tokens=True).strip()
