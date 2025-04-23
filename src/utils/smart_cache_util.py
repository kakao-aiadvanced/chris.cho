from __future__ import annotations

import abc
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum, auto
from functools import wraps
import hashlib
import inspect
import json
import logging
import pickle
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Type, Union, TypeVar, Generic
import xml.etree.ElementTree as ET
import redis
import zlib

import numpy as np
import pandas as pd

T = TypeVar('T')


class CacheBackend(Enum):
    MEMORY = auto()
    FILE = auto()
    REDIS = auto()
    HYBRID = auto()


class DataType(Enum):
    JSON = auto()
    XML = auto()
    BINARY = auto()
    PICKLE = auto()


class KeyStrategy(Enum):
    IDENTIFIER = auto()
    CONTENT = auto()
    HYBRID = auto()


@dataclass
class SmartCacheConfig:
    """SmartCache 설정"""
    backend: CacheBackend = CacheBackend.HYBRID
    default_ttl: timedelta = timedelta(hours=1)
    redis_url: Optional[str] = None
    cache_dir: Path = Path("./.cache")
    compress: bool = True
    max_memory_items: int = 1000
    enable_stats: bool = True
    key_strategy: KeyStrategy = KeyStrategy.HYBRID

    def __post_init__(self):
        self.cache_dir.mkdir(parents=True, exist_ok=True)


@dataclass
class CacheStats:
    """캐시 통계"""
    hits: int = 0
    misses: int = 0
    updates: int = 0
    size: int = 0
    created_at: datetime = field(default_factory=datetime.now)

    @property
    def hit_ratio(self) -> float:
        total = self.hits + self.misses
        return self.hits / total if total > 0 else 0


class CacheKeyEncoder(json.JSONEncoder):
    """Enum을 포함한 다양한 타입을 지원하는 JSON 인코더"""

    def default(self, obj: Any) -> Any:
        # Enum 처리
        if isinstance(obj, Enum):
            return {
                "__enum__": True,
                "class": obj.__class__.__name__,
                "name": obj.name,
                "value": obj.value
            }

        # 기존 타입들 처리
        elif isinstance(obj, (np.ndarray, pd.DataFrame, pd.Series)):
            return obj.tobytes()
        elif isinstance(obj, pd.Timestamp):
            return obj.isoformat()
        elif isinstance(obj, datetime):
            return obj.isoformat()
        elif hasattr(obj, 'to_dict'):
            return obj.to_dict()
        elif hasattr(obj, '__dict__'):
            return {
                f"{obj.__class__.__name__}_{k}": v
                for k, v in obj.__dict__.items()
                if not k.startswith('_')
            }
        return str(obj)


class SafeJSONSerializer:
    """안전한 JSON 직렬화 처리"""

    @staticmethod
    def serialize(obj: Any) -> str:
        try:
            return json.dumps(obj, sort_keys=True, cls=CacheKeyEncoder)
        except TypeError as e:
            # 직렬화 실패 시 객체의 문자열 표현 사용
            return str(obj)


class CacheKey:
    """개선된 캐시 키 생성 클래스"""

    @staticmethod
    def create(
            prefix: str,
            params: Dict[str, Any],
            strategy: str,
            identifier_fields: Optional[List[str]] = None
    ) -> str:
        if strategy == KeyStrategy.IDENTIFIER:
            return CacheKey._create_identifier_key(prefix, params, identifier_fields or [])
        elif strategy == KeyStrategy.CONTENT:
            return CacheKey._create_content_key(prefix, params)
        else:  # HYBRID
            return CacheKey._create_hybrid_key(prefix, params, identifier_fields or [])

    @staticmethod
    def _create_identifier_key(prefix: str, params: Dict[str, Any], fields: List[str]) -> str:
        """식별자 기반 키 생성"""
        parts = [prefix] if prefix else []

        for field in fields:
            if field in params:
                value = params[field]
                # 안전한 직렬화
                safe_value = SafeJSONSerializer.serialize(value)
                parts.append(f"{field}:{safe_value}")

        return ":".join(parts)

    @staticmethod
    def _create_content_key(prefix: str, params: Dict[str, Any]) -> str:
        """컨텐츠 기반 키 생성"""
        try:
            # 안전한 직렬화
            content = SafeJSONSerializer.serialize(params)
            content_hash = hashlib.md5(content.encode()).hexdigest()

            return f"{prefix}:{content_hash}" if prefix else content_hash
        except Exception as e:
            # 실패 시 대체 키 생성
            fallback_key = hashlib.md5(str(params).encode()).hexdigest()
            return f"{prefix}:fallback:{fallback_key}" if prefix else f"fallback:{fallback_key}"

    @staticmethod
    def _create_hybrid_key(prefix: str, params: Dict[str, Any], fields: List[str]) -> str:
        """하이브리드 키 생성"""
        id_part = CacheKey._create_identifier_key("", params, fields)
        content_part = CacheKey._create_content_key("", params)

        parts = [p for p in [prefix, id_part, content_part] if p]
        return ":".join(parts)


class CacheBase(abc.ABC, Generic[T]):
    """캐시 기본 클래스"""
    @abc.abstractmethod
    def get(self, key: str) -> Optional[T]:
        pass

    @abc.abstractmethod
    def set(self, key: str, value: T, ttl: Optional[timedelta] = None) -> None:
        pass

    @abc.abstractmethod
    def delete(self, key: str) -> None:
        pass

    @abc.abstractmethod
    def clear(self) -> None:
        pass


class MemoryCache(CacheBase[T]):
    """메모리 캐시"""

    def __init__(self, max_items: int = 1000):
        self.cache: Dict[str, tuple[T, datetime]] = {}
        self.max_items = max_items

    def get(self, key: str) -> Optional[T]:
        if key not in self.cache:
            return None
        value, expiry = self.cache[key]
        if expiry < datetime.now():
            del self.cache[key]
            return None
        return value

    def set(self, key: str, value: T, ttl: Optional[timedelta] = None) -> None:
        if len(self.cache) >= self.max_items:
            # LRU 방식으로 오래된 항목 제거
            oldest = min(self.cache.items(), key=lambda x: x[1][1])
            del self.cache[oldest[0]]

        expiry = datetime.now() + (ttl or timedelta(hours=1))
        self.cache[key] = (value, expiry)

    def delete(self, key: str) -> None:
        self.cache.pop(key, None)

    def clear(self) -> None:
        self.cache.clear()


class FileCache(CacheBase[T]):
    """파일 캐시"""

    def __init__(self, cache_dir: Path, compress: bool = True):
        self.cache_dir = cache_dir
        self.compress = compress
        self.cache_dir.mkdir(parents=True, exist_ok=True)

    def _get_path(self, key: str) -> Path:
        return self.cache_dir / f"{hashlib.md5(key.encode()).hexdigest()}.cache"

    def get(self, key: str) -> Optional[T]:
        path = self._get_path(key)
        if not path.exists():
            return None

        try:
            data = path.read_bytes()
            if self.compress:
                data = zlib.decompress(data)
            return pickle.loads(data)
        except Exception as e:
            logging.error(f"File cache read error: {str(e)}")
            return None

    def set(self, key: str, value: T, ttl: Optional[timedelta] = None) -> None:
        path = self._get_path(key)
        try:
            data = pickle.dumps(value)
            if self.compress:
                data = zlib.compress(data)
            path.write_bytes(data)
        except Exception as e:
            logging.error(f"File cache write error: {str(e)}")

    def delete(self, key: str) -> None:
        path = self._get_path(key)
        path.unlink(missing_ok=True)

    def clear(self) -> None:
        for path in self.cache_dir.glob("*.cache"):
            path.unlink(missing_ok=True)


class RedisCache(CacheBase[T]):
    """Redis 캐시"""

    def __init__(self, redis_url: str, compress: bool = True):
        self.redis = redis.Redis.from_url(redis_url)
        self.compress = compress

    def get(self, key: str) -> Optional[T]:
        try:
            data = self.redis.get(key)
            if data is None:
                return None
            if self.compress:
                data = zlib.decompress(data)
            return pickle.loads(data)
        except Exception as e:
            logging.error(f"Redis cache read error: {str(e)}")
            return None

    def set(self, key: str, value: T, ttl: Optional[timedelta] = None) -> None:
        try:
            data = pickle.dumps(value)
            if self.compress:
                data = zlib.compress(data)
            if ttl:
                self.redis.setex(key, int(ttl.total_seconds()), data)
            else:
                self.redis.set(key, data)
        except Exception as e:
            logging.error(f"Redis cache write error: {str(e)}")

    def delete(self, key: str) -> None:
        self.redis.delete(key)

    def clear(self) -> None:
        self.redis.flushdb()


class XMLCache:
    """XML 전용 캐시 처리"""

    @staticmethod
    def hash_xml(xml_str: str) -> str:
        """XML 문자열의 해시 생성"""
        try:
            # XML 파싱 및 정규화
            tree = ET.fromstring(xml_str)
            normalized = ET.tostring(tree, encoding='utf-8', method='xml')
            return hashlib.md5(normalized).hexdigest()
        except ET.ParseError:
            # 파싱 실패시 원본 문자열 해시
            return hashlib.md5(xml_str.encode()).hexdigest()

    @staticmethod
    def normalize_xml(xml_str: str) -> str:
        """XML 정규화"""
        tree = ET.fromstring(xml_str)
        return ET.tostring(tree, encoding='utf-8', method='xml').decode()


class SmartCache:
    """통합 캐시 관리 클래스"""

    def __init__(self, config: SmartCacheConfig):
        self.config = config
        self.stats = CacheStats() if config.enable_stats else None

        # 백엔드 초기화
        if config.backend in [CacheBackend.MEMORY, CacheBackend.HYBRID]:
            self.memory_cache = MemoryCache(config.max_memory_items)
            self.memory_cache.clear()

        if config.backend in [CacheBackend.FILE, CacheBackend.HYBRID]:
            self.file_cache = FileCache(config.cache_dir, config.compress)
            self.file_cache_dict = {}

        if config.redis_url is not None and config.backend in [CacheBackend.REDIS, CacheBackend.HYBRID]:
            self.redis_cache = RedisCache(config.redis_url, config.compress)

    def get(self, major_key: str, key: str) -> Optional[Any]:
        """캐시에서 데이터 조회"""
        value = None

        # 메모리 캐시 확인
        if hasattr(self, 'memory_cache'):
            value = self.memory_cache.get(key)
            if value is not None:
                if self.stats:
                    self.stats.hits += 1
                return value

        # 파일 캐시 확인
        if hasattr(self, 'file_cache_dict'):
            if not major_key in self.file_cache_dict:
                self.file_cache_dict[major_key] = FileCache(self.config.cache_dir / major_key, self.config.compress)
            else:
                pass

            value = self.file_cache_dict[major_key].get(key)
            if value is not None:
                if hasattr(self, 'memory_cache'):
                    self.memory_cache.set(key, value)
                if self.stats:
                    self.stats.hits += 1
                return value

        # Redis 캐시 확인
        if hasattr(self, 'redis_cache'):
            value = self.redis_cache.get(key)
            if value is not None:
                if hasattr(self, 'memory_cache'):
                    self.memory_cache.set(key, value)
                if self.stats:
                    self.stats.hits += 1
                return value

        if self.stats:
            self.stats.misses += 1

        return None

    def set(self, major_key: str, key: str, value: Any, ttl: Optional[timedelta] = None) -> None:
        """캐시에 데이터 저장"""
        ttl = ttl or self.config.default_ttl

        if hasattr(self, 'memory_cache'):
            self.memory_cache.set(key, value, ttl)

        if hasattr(self, 'file_cache_dict'):
            if major_key in self.file_cache_dict:
                self.file_cache_dict[major_key].set(key, value, ttl)
            else:
                self.file_cache_dict[major_key] = FileCache(self.config.cache_dir / major_key, self.config.compress)

        if hasattr(self, 'redis_cache'):
            self.redis_cache.set(key, value, ttl)

        if self.stats:
            self.stats.updates += 1
            self.stats.size += 1
    def cleanup(self, force: bool = False):
        """캐시 정리"""
        if force:
            self._remove_directory(self.config.cache_dir)

    def _archive_directory(self, path: Path):
        """디렉토리 아카이브"""
        import tarfile
        archive_path = self.config.base_path / 'archive' / f"{path.name}.tar.gz"
        archive_path.parent.mkdir(parents=True, exist_ok=True)

        with tarfile.open(archive_path, 'w:gz') as tar:
            tar.add(path, arcname=path.name)

        self._remove_directory(path)

    def _remove_directory(self, path: Path):
        """디렉토리 삭제"""
        try:
            for item in path.rglob('*'):
                if item.is_file():
                    item.unlink()
            path.rmdir()
        except Exception as e:
            logging.error(f"Error removing directory {path}: {e}")

def smart_cache(
        ttl: Optional[timedelta] = None,
        prefix: str = "",
        key_strategy: Optional[KeyStrategy] = None,
        identifier_fields: Optional[List[str]] = None
):
    """스마트 캐시 데코레이터"""

    def decorator(func: Callable) -> Callable:
        @wraps(func)
        def wrapper(*args, **kwargs):
            # 전역 캐시 인스턴스 가져오기
            cache = get_cache_instance()

            # 함수 파라미터 추출
            sig = inspect.signature(func)
            bound_args = sig.bind(*args, **kwargs)
            bound_args.apply_defaults()
            params = dict(bound_args.arguments)

            # 캐시 키 생성
            strategy = key_strategy or cache.config.key_strategy

            ## 캐시 메이저 키
            major_key = prefix or func.__name__

            key = CacheKey.create(
                major_key,
                params,
                strategy,
                identifier_fields
            )



            cached_result = cache.get(major_key, key)
            if cached_result is not None:
                return cached_result

            # 함수 실행
            result = func(*args, **kwargs)

            # 결과 캐싱
            cache.set(major_key, key, result, ttl)

            return result

        return wrapper

    return decorator


# 싱글톤 캐시 인스턴스
_cache_instance: Optional[SmartCache] = None


def _initialize_cache(config: SmartCacheConfig) -> SmartCache:
    """캐시 초기화"""
    global _cache_instance
    _cache_instance = SmartCache(config)
    return _cache_instance


def get_cache_instance() -> SmartCache:
    """캐시 인스턴스 가져오기"""
    if _cache_instance is None:
        raise RuntimeError("Cache not initialized. Call initialize_cache first.")
    return _cache_instance

def cleanup_cache(force:bool = False) -> SmartCache:
    """캐시 초기화"""
    global _cache_instance
    _cache_instance.cleanup(force=force)


def initialize_cache_with_smart_cache(default_ttl):

    # 1. 캐시 초기화
    config = SmartCacheConfig(
        backend=CacheBackend.FILE,
        redis_url=None, #"redis://localhost:6379/0",
        cache_dir=Path("./.cache"),
        compress=True,
        enable_stats=True,
        default_ttl=default_ttl
    )
    _initialize_cache(config)
