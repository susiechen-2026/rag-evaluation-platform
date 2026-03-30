import json
import os
from typing import Dict, Any

class ConfigManager:
    """配置管理类"""

    def __init__(self, config_file: str = "config.json"):
        """
        初始化配置管理
        
        Args:
            config_file: 配置文件路径
        """
        self.config_file = config_file
        self.config = self._load_config()

    def _load_config(self) -> Dict[str, Any]:
        """
        加载配置文件
        
        Returns:
            配置字典
        """
        if not os.path.exists(self.config_file):
            # 如果配置文件不存在，创建默认配置
            default_config = {
                "api": {
                    "dashscope_api_key": "YOUR_API_KEY_HERE"
                },
                "model": {
                    "name": "qwen-plus",
                    "base_url": "https://dashscope.aliyuncs.com/compatible-mode/v1",
                    "temperature": 0
                },
                "evaluation": {
                    "threshold": 0.7,
                    "batch_size": 5,
                    "max_workers": 4
                },
                "output": {
                    "eval_results_dir": "./eval_results",
                    "visualizations_dir": "./visualizations"
                },
                "rag": {
                    "type": "local",
                    "local_function": "rag.rag_observe.rag_chatbot"
                }
            }
            self._save_config(default_config)
            return default_config
        
        with open(self.config_file, 'r', encoding='utf-8') as f:
            return json.load(f)

    def _save_config(self, config: Dict[str, Any]):
        """
        保存配置文件
        
        Args:
            config: 配置字典
        """
        with open(self.config_file, 'w', encoding='utf-8') as f:
            json.dump(config, f, ensure_ascii=False, indent=2)

    def get(self, key: str, default: Any = None) -> Any:
        """
        获取配置值
        
        Args:
            key: 配置键，支持点号分隔的嵌套键
            default: 默认值
            
        Returns:
            配置值
        """
        keys = key.split('.')
        value = self.config
        
        for k in keys:
            if isinstance(value, dict) and k in value:
                value = value[k]
            else:
                return default
        
        return value

    def set(self, key: str, value: Any):
        """
        设置配置值
        
        Args:
            key: 配置键，支持点号分隔的嵌套键
            value: 配置值
        """
        keys = key.split('.')
        config = self.config
        
        for k in keys[:-1]:
            if k not in config:
                config[k] = {}
            config = config[k]
        
        config[keys[-1]] = value
        self._save_config(self.config)

    def get_api_key(self) -> str:
        """
        获取API密钥
        
        Returns:
            API密钥
        """
        api_key = self.get("api.dashscope_api_key")
        # 优先从环境变量获取
        env_api_key = os.getenv("DASHSCOPE_API_KEY") or os.getenv("DEEPSEEK_API_KEY")
        if env_api_key:
            return env_api_key
        return api_key

    def get_model_config(self) -> Dict[str, Any]:
        """
        获取模型配置
        
        Returns:
            模型配置
        """
        return self.get("model", {})

    def get_evaluation_config(self) -> Dict[str, Any]:
        """
        获取评测配置
        
        Returns:
            评测配置
        """
        return self.get("evaluation", {})

    def get_output_config(self) -> Dict[str, Any]:
        """
        获取输出配置
        
        Returns:
            输出配置
        """
        return self.get("output", {})

    def get_rag_config(self) -> Dict[str, Any]:
        """
        获取RAG配置
        
        Returns:
            RAG配置
        """
        return self.get("rag", {})

    def reload(self):
        """
        重新加载配置文件
        """
        self.config = self._load_config()

# 全局配置管理器实例
config_manager = ConfigManager()
