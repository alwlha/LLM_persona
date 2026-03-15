from abc import ABC, abstractmethod

from src.activation import ActivationConfig
from src.models.base import BaseModel


class BaseTask(ABC):
    """
    所有测试任务的基类。
    一个 Task 代表一种对模型人格/能力的测量方式。
    """

    @property
    @abstractmethod
    def name(self) -> str:
        """任务唯一标识符，用于结果文件命名"""
        ...

    @abstractmethod
    def run(self, model: BaseModel, activation: ActivationConfig) -> dict:
        """
        对给定模型执行完整的任务测试。

        Args:
            model: 待测试的模型实例
            activation: 当前激活配置（提示词或向量激活）
        Returns:
            包含任务名、模型名、激活方法名及各维度得分的结果字典
        """
        ...
