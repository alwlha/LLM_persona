from abc import ABC, abstractmethod

from src.activation import ActivationConfig


class BaseModel(ABC):
    """所有模型的基类，定义统一的调用接口"""

    @abstractmethod
    def query(
        self,
        prompt: str,
        system: str | None = None,
        activation: ActivationConfig | None = None,
    ) -> str:
        """
        向模型发送请求并返回文本响应。

        Args:
            prompt: 用户输入的提示文本
            system: 可选的系统提示（角色设定）
            activation: 完整激活配置（用于向量注入等高级激活方法）
        Returns:
            模型的文本回复
        """
        ...

    @property
    @abstractmethod
    def name(self) -> str:
        """模型的标识名称"""
        ...
