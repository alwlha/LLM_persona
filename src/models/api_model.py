import time

from openai import OpenAI

from src.activation import ActivationConfig
from .base import BaseModel


class APIModel(BaseModel):
    """
    通过 OpenAI 兼容 API 调用的闭源模型。
    支持 GPT、Gemini、Claude 等通过统一接口暴露的模型。
    """

    def __init__(self, model_name: str, api_key: str, base_url: str):
        self._name = model_name
        self.client = OpenAI(
            api_key=api_key, 
            base_url=base_url,
            timeout=60.0  # 增加默认超时
        )

    @property
    def name(self) -> str:
        return self._name

    def query(
        self,
        prompt: str,
        system: str | None = None,
        activation: ActivationConfig | None = None,
        max_retries: int = 3,
    ) -> str:
        messages = []
        if system:
            messages.append({"role": "system", "content": system})
        messages.append({"role": "user", "content": prompt})

        for attempt in range(max_retries):
            try:
                response = self.client.chat.completions.create(
                    model=self._name,
                    messages=messages,
                    temperature=0.1,
                )
                content = response.choices[0].message.content
                return content.strip() if content else ""
            except Exception as e:
                if attempt < max_retries - 1:
                    wait_time = (attempt + 1) * 2  # 指数退避
                    print(f"\n  [API_RETRY] {self._name} connection issue: {e}. Retrying in {wait_time}s...")
                    time.sleep(wait_time)
                else:
                    return f"[API_ERROR] {str(e)}"
        
        return "[API_ERROR] Max retries exceeded"
