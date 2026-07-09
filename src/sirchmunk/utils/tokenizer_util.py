# Copyright (c) ModelScope Contributors. All rights reserved.
import threading
from typing import Any, Dict, List, Optional, Union


# ponytail: tokenizer 进程内单例缓存。
# AutoTokenizer.from_pretrained 每次调用都会让 modelscope 联网校验文件清单
# （日志里的 "Downloading 8 files"），而请求热路径每次都 new TokenizerUtil，
# 导致每问一次问题都重复校验。这里按 model_id 缓存实例，进程内只加载一次。
_TOKENIZER_LOCK = threading.Lock()
_tokenizer_cache: Dict[str, Any] = {}


def _get_tokenizer(model_id: str) -> Any:
    """按 model_id 返回进程内单例 tokenizer（线程安全）。"""
    cached = _tokenizer_cache.get(model_id)
    if cached is not None:
        return cached
    with _TOKENIZER_LOCK:
        cached = _tokenizer_cache.get(model_id)
        if cached is not None:
            return cached
        from modelscope import AutoTokenizer

        tokenizer = AutoTokenizer.from_pretrained(model_id)
        _tokenizer_cache[model_id] = tokenizer
        return tokenizer


class TokenizerUtil:
    """Fast tokenizer utility using modelscope AutoTokenizer."""

    def __init__(self, model_id: Optional[str] = None):
        """
        Tokenizer encoding and counting utility.
        Args:
            model_id: Model ID for loading the tokenizer. Defaults to "Qwen/Qwen3-8B".
        """
        model_id = model_id or "Qwen/Qwen3-8B"
        # 复用进程内单例，避免每次实例化都触发 modelscope 联网校验。
        self.tokenizer = _get_tokenizer(model_id)

    def encode(self, content: str) -> List[int]:
        """Encode text into token IDs.

        Args:
            content: Input text string.

        Returns:
            List of token IDs.
        """
        if not content.strip():
            return []
        return self.tokenizer.encode(content.strip())

    def decode(self, token_ids: List[int]) -> str:
        """Decode a list of token IDs back into a natural text string.

        Args:
            token_ids: List of token IDs to decode.

        Returns:
            Decoded text string.
        """
        if not token_ids:
            return ""
        return self.tokenizer.decode(token_ids, skip_special_tokens=True)

    def segment(self, content: str) -> List[str]:
        """Tokenize text into a list of token strings suitable for BM25-like algorithms indexing/retrieval.

        This method returns the actual sub-word tokens as strings (e.g., ["▁Hello", "▁world"]),
        preserving token boundaries. These tokens can be directly used as terms in BM25.

        Args:
            content: Input text string.

        Returns:
            List of token strings (not IDs), ready for BM25-style processing.
        """
        if not content.strip():
            return []
        token_ids = self.encode(content)
        # Decode each token ID individually to get its string representation
        token_strings = [
            self.tokenizer.decode([tid], skip_special_tokens=True)
            for tid in token_ids
        ]
        return token_strings

    def count_tokens(self, contents: Union[str, List[str]]) -> Union[int, List[int]]:
        """
        Batch count tokens for multiple texts.

        Args:
            contents: List of input text strings.

        Returns:
            List of token counts corresponding to each input text, or an integer if a single string is provided.
        """
        if isinstance(contents, str):
            contents = [contents]

        counts = []
        for content in contents:
            if not content.strip():
                counts.append(0)
            else:
                counts.append(len(self.tokenizer.encode(content.strip())))

        if len(contents) == 1:
            return counts[0]
        return counts
