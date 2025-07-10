# 🤝 Guía de Contribución

¡Gracias por tu interés en contribuir a Quantum Trading Platform! Este documento proporciona las pautas y mejores prácticas para contribuir al proyecto.

## 📋 Tabla de Contenidos

- [Código de Conducta](#código-de-conducta)
- [Cómo Contribuir](#cómo-contribuir)
- [Configuración del Entorno](#configuración-del-entorno)
- [Flujo de Trabajo](#flujo-de-trabajo)
- [Estándares de Código](#estándares-de-código)
- [Testing](#testing)
- [Documentación](#documentación)
- [Pull Requests](#pull-requests)

## 📜 Código de Conducta

Este proyecto sigue el [Código de Conducta de Contributor Covenant](https://www.contributor-covenant.org/). Al participar, se espera que respetes este código.

## 🚀 Cómo Contribuir

1. **Reportar Bugs**
   - Usa el template de issues para bugs
   - Incluye pasos para reproducir
   - Proporciona logs y capturas si es posible

2. **Sugerir Mejoras**
   - Abre un issue para discusión
   - Explica el caso de uso
   - Proporciona ejemplos

3. **Contribuir Código**
   - Fork el repositorio
   - Crea una rama feature
   - Implementa cambios con tests
   - Envía pull request

## 💻 Configuración del Entorno

```bash
# 1. Fork y clona el repositorio
git clone https://github.com/tu-usuario/quantum-trading-platform.git
cd quantum-trading-platform

# 2. Crea entorno virtual
python -m venv venv
source venv/bin/activate  # Linux/Mac
# o
venv\Scripts\activate     # Windows

# 3. Instala dependencias de desarrollo
make install

# 4. Configura pre-commit hooks
make setup-pre-commit

# 5. Copia configuración de ejemplo
cp .env.example .env

# 6. Inicia servicios locales
make dev
```

## 🔄 Flujo de Trabajo

### Branches

- `main` - Código estable en producción
- `develop` - Integración de features
- `feature/*` - Nuevas características
- `bugfix/*` - Corrección de bugs
- `hotfix/*` - Fixes urgentes

### Proceso

1. Crea rama desde `develop`:
   ```bash
   git checkout -b feature/mi-nueva-feature develop
   ```

2. Haz commits siguiendo [Conventional Commits](https://www.conventionalcommits.org/):
   ```bash
   git commit -m "feat: añadir análisis de sentimiento"
   git commit -m "fix: corregir cálculo de stop loss"
   git commit -m "docs: actualizar guía de API"
   ```

3. Mantén tu rama actualizada:
   ```bash
   git checkout develop
   git pull origin develop
   git checkout feature/mi-nueva-feature
   git rebase develop
   ```

## 📏 Estándares de Código

### Python Style Guide

- Seguimos [PEP 8](https://pep8.org/)
- Usamos [Black](https://black.readthedocs.io/) para formateo
- Type hints son obligatorios
- Docstrings en formato Google

### Ejemplo de Código

```python
from typing import List, Optional

from src.core.base import BaseModel


class TradingStrategy(BaseModel):
    """
    Estrategia de trading base.
    
    Esta clase define la interfaz para todas las estrategias
    de trading en el sistema.
    
    Attributes:
        name: Nombre único de la estrategia
        symbols: Lista de símbolos a operar
        risk_percentage: Porcentaje de riesgo por operación
    """
    
    def __init__(
        self,
        name: str,
        symbols: List[str],
        risk_percentage: float = 2.0
    ) -> None:
        """
        Inicializa la estrategia.
        
        Args:
            name: Nombre de la estrategia
            symbols: Símbolos a operar
            risk_percentage: Riesgo máximo por operación
            
        Raises:
            ValueError: Si risk_percentage no está entre 0 y 100
        """
        if not 0 < risk_percentage <= 100:
            raise ValueError("risk_percentage debe estar entre 0 y 100")
            
        self.name = name
        self.symbols = symbols
        self.risk_percentage = risk_percentage
    
    async def analyze(self, market_data: dict) -> Optional[Signal]:
        """
        Analiza datos de mercado y genera señales.
        
        Args:
            market_data: Datos actuales del mercado
            
        Returns:
            Señal de trading si se detecta oportunidad
        """
        raise NotImplementedError("Las subclases deben implementar analyze()")
```

### Verificación de Código

Antes de hacer commit:

```bash
# Formatear código
make format

# Verificar estilo
make lint

# Ejecutar tests
make test

# Todo junto
make format lint test
```

## 🧪 Testing

### Estructura de Tests

```
tests/
├── unit/           # Tests unitarios
├── integration/    # Tests de integración
├── performance/    # Tests de rendimiento
└── fixtures/       # Datos de prueba
```

### Escribir Tests

```python
import pytest
from unittest.mock import Mock, patch

from src.trading.strategy import TradingStrategy


class TestTradingStrategy:
    """Tests para TradingStrategy."""
    
    @pytest.fixture
    def strategy(self):
        """Fixture de estrategia básica."""
        return TradingStrategy(
            name="test_strategy",
            symbols=["BTC/USDT"],
            risk_percentage=2.0
        )
    
    def test_initialization(self, strategy):
        """Test de inicialización correcta."""
        assert strategy.name == "test_strategy"
        assert strategy.symbols == ["BTC/USDT"]
        assert strategy.risk_percentage == 2.0
    
    def test_invalid_risk_percentage(self):
        """Test de validación de risk_percentage."""
        with pytest.raises(ValueError, match="risk_percentage"):
            TradingStrategy("test", ["BTC/USDT"], risk_percentage=150)
    
    @patch("src.trading.strategy.market_client")
    async def test_analyze_with_mock(self, mock_client, strategy):
        """Test de análisis con datos mockeados."""
        mock_client.get_ticker.return_value = {"price": 50000}
        
        result = await strategy.analyze({"symbol": "BTC/USDT"})
        
        assert result is not None
        mock_client.get_ticker.assert_called_once()
```

### Coverage

Mantenemos un mínimo de 80% de coverage:

```bash
make test-coverage
```

## 📚 Documentación

### Docstrings

Todos los módulos, clases y funciones públicas deben tener docstrings:

```python
def calculate_position_size(
    balance: float,
    risk_percentage: float,
    stop_loss_percentage: float
) -> float:
    """
    Calcula el tamaño de posición basado en gestión de riesgo.
    
    Utiliza la fórmula de Kelly Criterion modificada para
    determinar el tamaño óptimo de la posición.
    
    Args:
        balance: Balance total de la cuenta
        risk_percentage: Porcentaje de riesgo por operación
        stop_loss_percentage: Distancia al stop loss en porcentaje
        
    Returns:
        Tamaño de posición en unidades base
        
    Example:
        >>> calculate_position_size(10000, 2.0, 5.0)
        400.0
    """
    return (balance * risk_percentage / 100) / (stop_loss_percentage / 100)
```

### Actualizar Docs

Cuando añadas nuevas features:

1. Actualiza docstrings
2. Añade ejemplos en `docs/examples/`
3. Actualiza API reference si es necesario
4. Ejecuta `make docs` para verificar

## 🔀 Pull Requests

### Checklist

- [ ] Tests pasan (`make test`)
- [ ] Código formateado (`make format`)
- [ ] Linting sin errores (`make lint`)
- [ ] Coverage > 80%
- [ ] Documentación actualizada
- [ ] Commits siguen convención
- [ ] Branch actualizada con develop

### Template de PR

```markdown
## Descripción
Breve descripción de los cambios

## Tipo de cambio
- [ ] Bug fix
- [ ] Nueva feature
- [ ] Breaking change
- [ ] Documentación

## ¿Cómo se ha probado?
Describe los tests realizados

## Checklist
- [ ] Mi código sigue los estándares del proyecto
- [ ] He añadido tests que prueban mi cambio
- [ ] Todos los tests pasan localmente
- [ ] He actualizado la documentación
```

## 🎯 Prioridades

Actualmente estamos enfocados en:

1. **Estabilidad**: Mejorar cobertura de tests
2. **Performance**: Optimizar latencia de órdenes
3. **Features**: Más indicadores técnicos
4. **Documentación**: Ejemplos y tutoriales

## 💬 Comunicación

- **Discord**: [Únete a nuestro servidor](https://discord.gg/quantum-trading)
- **Issues**: Para bugs y features
- **Discussions**: Para preguntas y ideas

---

¡Gracias por contribuir a hacer Quantum Trading Platform mejor! 🚀