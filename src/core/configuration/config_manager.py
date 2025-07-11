"""
Enterprise Configuration Manager with hot-reload and multi-source support.

Features:
- Multiple configuration sources with priority handling
- Real-time configuration updates with hot-reload
- Configuration validation and type safety
- Encrypted secrets management
- Configuration versioning and audit trails
- Thread-safe configuration access
"""

import asyncio
import threading
import time
from copy import deepcopy
from pathlib import Path
from typing import Any, Dict, List, Optional, Type, TypeVar, Union, Callable
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
import json

from .models import AppConfig, Environment
from .providers import BaseConfigProvider, FileConfigProvider, EnvironmentConfigProvider
from .validation import ConfigValidator, ValidationResult
from .encryption import ConfigEncryption, SecretManager
from .hot_reload import ConfigHotReloader, ConfigWatcher

T = TypeVar('T')


class ConfigSource(Enum):
    """Configuration source types with priority."""
    DEFAULTS = 1      # Lowest priority
    FILE = 2          # Configuration files
    ENVIRONMENT = 3   # Environment variables
    RUNTIME = 4       # Runtime overrides
    OVERRIDE = 5      # Highest priority - manual overrides


@dataclass
class ConfigChange:
    """Represents a configuration change event."""
    path: str
    old_value: Any
    new_value: Any
    source: ConfigSource
    timestamp: datetime = field(default_factory=datetime.utcnow)
    user: Optional[str] = None
    reason: Optional[str] = None


@dataclass
class ConfigSnapshot:
    """Configuration snapshot for versioning."""
    version: str
    config: Dict[str, Any]
    timestamp: datetime = field(default_factory=datetime.utcnow)
    changes: List[ConfigChange] = field(default_factory=list)
    description: Optional[str] = None


class ConfigurationManager:
    """
    Enterprise configuration manager with advanced features.
    
    Features:
    - Multi-source configuration with priority handling
    - Real-time hot-reload capabilities
    - Type-safe configuration access
    - Encrypted secrets management
    - Configuration validation and constraints
    - Audit trail and versioning
    - Thread-safe operations
    - Plugin architecture for custom providers
    """
    
    def __init__(self, config_class: Type[T] = AppConfig):
        self.config_class = config_class
        self._config: Optional[T] = None
        self._config_dict: Dict[str, Any] = {}
        self._providers: Dict[ConfigSource, List[BaseConfigProvider]] = {
            source: [] for source in ConfigSource
        }
        self._validator = ConfigValidator()
        self._encryption = ConfigEncryption()
        self._secret_manager = SecretManager()
        self._hot_reloader: Optional[ConfigHotReloader] = None
        self._watchers: List[ConfigWatcher] = []
        
        # Change tracking
        self._change_callbacks: List[Callable[[ConfigChange], None]] = []
        self._snapshots: List[ConfigSnapshot] = []
        self._current_version = "1.0.0"
        
        # Thread safety
        self._lock = threading.RLock()
        self._initialized = False
        
        # Default providers
        self._setup_default_providers()
    
    def _setup_default_providers(self):
        """Setup default configuration providers."""
        # Environment variables provider
        env_provider = EnvironmentConfigProvider()
        self.add_provider(ConfigSource.ENVIRONMENT, env_provider)
    
    def add_provider(self, source: ConfigSource, provider: BaseConfigProvider):
        """Add a configuration provider for a specific source."""
        with self._lock:
            self._providers[source].append(provider)
    
    def add_file_provider(self, file_path: Union[str, Path], source: ConfigSource = ConfigSource.FILE):
        """Add a file-based configuration provider."""
        provider = FileConfigProvider(Path(file_path))
        self.add_provider(source, provider)
    
    def initialize(self, config_files: Optional[List[Union[str, Path]]] = None,
                  enable_hot_reload: bool = True) -> T:
        """
        Initialize the configuration manager.
        
        Args:
            config_files: List of configuration files to load
            enable_hot_reload: Enable hot-reload for file changes
            
        Returns:
            Initialized configuration object
        """
        with self._lock:
            if self._initialized:
                return self._config
            
            # Add file providers
            if config_files:
                for file_path in config_files:
                    self.add_file_provider(file_path)
            
            # Load initial configuration
            self._load_configuration()
            
            # Setup hot-reload
            if enable_hot_reload:
                self._setup_hot_reload()
            
            self._initialized = True
            self._create_snapshot("Initial configuration loaded")
            
            return self._config
    
    def _load_configuration(self):
        """Load configuration from all providers in priority order."""
        merged_config = {}
        
        # Load from providers in priority order (lowest to highest)
        for source in ConfigSource:
            for provider in self._providers[source]:
                try:
                    provider_config = provider.load()
                    if provider_config:
                        merged_config = self._merge_configs(merged_config, provider_config)
                except Exception as e:
                    # Log error but continue with other providers
                    print(f"Error loading config from {provider}: {e}")
        
        # Decrypt encrypted values
        merged_config = self._decrypt_secrets(merged_config)
        
        # Validate configuration
        validation_result = self._validator.validate(merged_config, self.config_class)
        if not validation_result.is_valid:
            raise ValueError(f"Configuration validation failed: {validation_result.errors}")
        
        # Create configuration object
        try:
            self._config = self.config_class(**merged_config)
            self._config_dict = merged_config
        except Exception as e:
            raise ValueError(f"Failed to create configuration object: {e}")
    
    def _merge_configs(self, base: Dict[str, Any], override: Dict[str, Any]) -> Dict[str, Any]:
        """Deep merge two configuration dictionaries."""
        result = deepcopy(base)
        
        for key, value in override.items():
            if key in result and isinstance(result[key], dict) and isinstance(value, dict):
                result[key] = self._merge_configs(result[key], value)
            else:
                result[key] = deepcopy(value)
        
        return result
    
    def _decrypt_secrets(self, config: Dict[str, Any]) -> Dict[str, Any]:
        """Decrypt encrypted secrets in configuration."""
        result = deepcopy(config)
        
        def decrypt_recursive(obj):
            if isinstance(obj, dict):
                for key, value in obj.items():
                    if isinstance(value, str) and value.startswith("encrypted:"):
                        try:
                            obj[key] = self._encryption.decrypt(value[10:])  # Remove "encrypted:" prefix
                        except Exception:
                            # Keep original value if decryption fails
                            pass
                    elif isinstance(value, (dict, list)):
                        decrypt_recursive(value)
            elif isinstance(obj, list):
                for item in obj:
                    if isinstance(item, (dict, list)):
                        decrypt_recursive(item)
        
        decrypt_recursive(result)
        return result
    
    def _setup_hot_reload(self):
        """Setup hot-reload for configuration files."""
        file_providers = []
        for provider_list in self._providers.values():
            for provider in provider_list:
                if isinstance(provider, FileConfigProvider):
                    file_providers.append(provider)
        
        if file_providers:
            self._hot_reloader = ConfigHotReloader(
                file_providers=file_providers,
                reload_callback=self._on_config_reload
            )
            self._hot_reloader.start()
    
    def _on_config_reload(self, changed_files: List[Path]):
        """Handle configuration reload event."""
        try:
            old_config_dict = deepcopy(self._config_dict)
            self._load_configuration()
            
            # Detect changes
            changes = self._detect_changes(old_config_dict, self._config_dict)
            
            # Notify change callbacks
            for change in changes:
                for callback in self._change_callbacks:
                    try:
                        callback(change)
                    except Exception as e:
                        print(f"Error in config change callback: {e}")
            
            # Create snapshot if there are changes
            if changes:
                self._create_snapshot(f"Hot-reload from files: {[str(f) for f in changed_files]}")
            
        except Exception as e:
            print(f"Error during configuration reload: {e}")
    
    def _detect_changes(self, old_config: Dict[str, Any], new_config: Dict[str, Any]) -> List[ConfigChange]:
        """Detect changes between old and new configuration."""
        changes = []
        
        def compare_recursive(old_obj, new_obj, path=""):
            if isinstance(old_obj, dict) and isinstance(new_obj, dict):
                # Check for removed keys
                for key in old_obj:
                    if key not in new_obj:
                        changes.append(ConfigChange(
                            path=f"{path}.{key}" if path else key,
                            old_value=old_obj[key],
                            new_value=None,
                            source=ConfigSource.FILE
                        ))
                
                # Check for added or modified keys
                for key in new_obj:
                    new_path = f"{path}.{key}" if path else key
                    if key not in old_obj:
                        changes.append(ConfigChange(
                            path=new_path,
                            old_value=None,
                            new_value=new_obj[key],
                            source=ConfigSource.FILE
                        ))
                    elif old_obj[key] != new_obj[key]:
                        if isinstance(old_obj[key], dict) and isinstance(new_obj[key], dict):
                            compare_recursive(old_obj[key], new_obj[key], new_path)
                        else:
                            changes.append(ConfigChange(
                                path=new_path,
                                old_value=old_obj[key],
                                new_value=new_obj[key],
                                source=ConfigSource.FILE
                            ))
            elif old_obj != new_obj:
                changes.append(ConfigChange(
                    path=path,
                    old_value=old_obj,
                    new_value=new_obj,
                    source=ConfigSource.FILE
                ))
        
        compare_recursive(old_config, new_config)
        return changes
    
    def get_config(self) -> T:
        """Get the current configuration object."""
        if not self._initialized:
            raise RuntimeError("Configuration manager not initialized")
        return self._config
    
    def get_value(self, path: str, default: Any = None) -> Any:
        """
        Get a configuration value by dot-notation path.
        
        Args:
            path: Dot-separated path (e.g., "database.host")
            default: Default value if path not found
            
        Returns:
            Configuration value or default
        """
        try:
            value = self._config_dict
            for part in path.split('.'):
                value = value[part]
            return value
        except (KeyError, TypeError):
            return default
    
    def set_value(self, path: str, value: Any, source: ConfigSource = ConfigSource.RUNTIME,
                 user: Optional[str] = None, reason: Optional[str] = None) -> bool:
        """
        Set a configuration value at runtime.
        
        Args:
            path: Dot-separated path (e.g., "database.host")
            value: New value to set
            source: Configuration source
            user: User making the change
            reason: Reason for the change
            
        Returns:
            True if value was set successfully
        """
        with self._lock:
            try:
                old_value = self.get_value(path)
                
                # Update configuration dictionary
                config_dict = deepcopy(self._config_dict)
                current = config_dict
                parts = path.split('.')
                
                for part in parts[:-1]:
                    if part not in current:
                        current[part] = {}
                    current = current[part]
                
                current[parts[-1]] = value
                
                # Validate new configuration
                validation_result = self._validator.validate(config_dict, self.config_class)
                if not validation_result.is_valid:
                    return False
                
                # Update configuration object
                self._config = self.config_class(**config_dict)
                self._config_dict = config_dict
                
                # Record change
                change = ConfigChange(
                    path=path,
                    old_value=old_value,
                    new_value=value,
                    source=source,
                    user=user,
                    reason=reason
                )
                
                # Notify callbacks
                for callback in self._change_callbacks:
                    try:
                        callback(change)
                    except Exception as e:
                        print(f"Error in config change callback: {e}")
                
                return True
                
            except Exception as e:
                print(f"Error setting configuration value: {e}")
                return False
    
    def add_change_callback(self, callback: Callable[[ConfigChange], None]):
        """Add a callback for configuration changes."""
        self._change_callbacks.append(callback)
    
    def remove_change_callback(self, callback: Callable[[ConfigChange], None]):
        """Remove a configuration change callback."""
        if callback in self._change_callbacks:
            self._change_callbacks.remove(callback)
    
    def _create_snapshot(self, description: Optional[str] = None):
        """Create a configuration snapshot."""
        snapshot = ConfigSnapshot(
            version=self._current_version,
            config=deepcopy(self._config_dict),
            description=description
        )
        self._snapshots.append(snapshot)
        
        # Limit snapshot history
        if len(self._snapshots) > 50:
            self._snapshots = self._snapshots[-50:]
    
    def get_snapshots(self, limit: int = 10) -> List[ConfigSnapshot]:
        """Get recent configuration snapshots."""
        return self._snapshots[-limit:]
    
    def restore_snapshot(self, version: str) -> bool:
        """Restore configuration from a snapshot."""
        with self._lock:
            for snapshot in self._snapshots:
                if snapshot.version == version:
                    try:
                        self._config = self.config_class(**snapshot.config)
                        self._config_dict = deepcopy(snapshot.config)
                        self._create_snapshot(f"Restored from snapshot {version}")
                        return True
                    except Exception as e:
                        print(f"Error restoring snapshot: {e}")
                        return False
            return False
    
    def export_config(self, include_secrets: bool = False) -> Dict[str, Any]:
        """
        Export current configuration.
        
        Args:
            include_secrets: Whether to include decrypted secrets
            
        Returns:
            Configuration dictionary
        """
        config = deepcopy(self._config_dict)
        
        if not include_secrets:
            # Remove or mask sensitive fields
            def mask_secrets(obj, path=""):
                if isinstance(obj, dict):
                    for key, value in obj.items():
                        if any(secret_key in key.lower() for secret_key in 
                              ['password', 'secret', 'token', 'key', 'auth']):
                            obj[key] = "***MASKED***"
                        elif isinstance(value, (dict, list)):
                            mask_secrets(value, f"{path}.{key}")
                elif isinstance(obj, list):
                    for item in obj:
                        if isinstance(item, (dict, list)):
                            mask_secrets(item, path)
            
            mask_secrets(config)
        
        return config
    
    def validate_config(self, config_dict: Optional[Dict[str, Any]] = None) -> ValidationResult:
        """Validate configuration."""
        config_to_validate = config_dict or self._config_dict
        return self._validator.validate(config_to_validate, self.config_class)
    
    def reload(self) -> bool:
        """Manually reload configuration from all sources."""
        with self._lock:
            try:
                old_config_dict = deepcopy(self._config_dict)
                self._load_configuration()
                
                changes = self._detect_changes(old_config_dict, self._config_dict)
                if changes:
                    self._create_snapshot("Manual reload")
                
                return True
            except Exception as e:
                print(f"Error during manual reload: {e}")
                return False
    
    def shutdown(self):
        """Shutdown the configuration manager."""
        if self._hot_reloader:
            self._hot_reloader.stop()
        
        for watcher in self._watchers:
            watcher.stop()
        
        self._initialized = False


# Global configuration manager instance
_global_config_manager: Optional[ConfigurationManager] = None


def get_config_manager() -> ConfigurationManager:
    """Get the global configuration manager instance."""
    global _global_config_manager
    if _global_config_manager is None:
        _global_config_manager = ConfigurationManager()
    return _global_config_manager


def initialize_config(config_files: Optional[List[Union[str, Path]]] = None,
                     config_class: Type[T] = AppConfig,
                     enable_hot_reload: bool = True) -> T:
    """
    Initialize the global configuration system.
    
    Args:
        config_files: List of configuration files to load
        config_class: Configuration model class
        enable_hot_reload: Enable hot-reload for file changes
        
    Returns:
        Initialized configuration object
    """
    global _global_config_manager
    _global_config_manager = ConfigurationManager(config_class)
    return _global_config_manager.initialize(config_files, enable_hot_reload)


def get_config() -> AppConfig:
    """Get the current application configuration."""
    return get_config_manager().get_config()


def get_config_value(path: str, default: Any = None) -> Any:
    """Get a configuration value by path."""
    return get_config_manager().get_value(path, default)


def set_config_value(path: str, value: Any, user: Optional[str] = None, 
                    reason: Optional[str] = None) -> bool:
    """Set a configuration value at runtime."""
    return get_config_manager().set_value(path, value, ConfigSource.RUNTIME, user, reason)