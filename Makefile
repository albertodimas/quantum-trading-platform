# Makefile para Quantum Trading Platform
.PHONY: help install dev test lint format clean docs

# Variables
PYTHON := python3
PIP := pip3
PROJECT_NAME := quantum-trading-platform
VENV := venv
SRC_DIR := src
TEST_DIR := tests

# Colores para output
GREEN := \033[0;32m
NC := \033[0m # No Color

help: ## Mostrar esta ayuda
	@echo "$(GREEN)Comandos disponibles:$(NC)"
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | sort | awk 'BEGIN {FS = ":.*?## "}; {printf "  \033[36m%-20s\033[0m %s\n", $$1, $$2}'

install: ## Instalar dependencias del proyecto
	@echo "$(GREEN)Instalando dependencias...$(NC)"
	$(PIP) install --upgrade pip
	$(PIP) install -r requirements/development.txt
	$(PIP) install -e .
	@echo "$(GREEN)Instalación completada!$(NC)"

dev: ## Iniciar entorno de desarrollo
	@echo "$(GREEN)Iniciando servicios de desarrollo...$(NC)"
	docker-compose -f docker/docker-compose.dev.yml up -d
	@echo "$(GREEN)Servicios iniciados!$(NC)"

dev-down: ## Detener entorno de desarrollo
	docker-compose -f docker/docker-compose.dev.yml down

test: ## Ejecutar todos los tests
	@echo "$(GREEN)Ejecutando tests...$(NC)"
	pytest $(TEST_DIR) -v --cov=$(SRC_DIR) --cov-report=html

test-unit: ## Ejecutar solo tests unitarios
	pytest $(TEST_DIR)/unit -v

test-integration: ## Ejecutar solo tests de integración
	pytest $(TEST_DIR)/integration -v

test-coverage: ## Generar reporte de coverage
	pytest $(TEST_DIR) --cov=$(SRC_DIR) --cov-report=html --cov-report=term
	@echo "$(GREEN)Reporte HTML generado en htmlcov/index.html$(NC)"

lint: ## Verificar calidad del código
	@echo "$(GREEN)Verificando código...$(NC)"
	flake8 $(SRC_DIR) $(TEST_DIR)
	pylint $(SRC_DIR)
	mypy $(SRC_DIR) --ignore-missing-imports

format: ## Formatear código automáticamente
	@echo "$(GREEN)Formateando código...$(NC)"
	black $(SRC_DIR) $(TEST_DIR)
	isort $(SRC_DIR) $(TEST_DIR)

clean: ## Limpiar archivos temporales
	@echo "$(GREEN)Limpiando archivos temporales...$(NC)"
	find . -type d -name "__pycache__" -exec rm -rf {} +
	find . -type f -name "*.pyc" -delete
	find . -type f -name "*.pyo" -delete
	find . -type f -name "*.coverage" -delete
	rm -rf .pytest_cache
	rm -rf htmlcov
	rm -rf dist
	rm -rf build
	rm -rf *.egg-info

docs: ## Generar documentación
	@echo "$(GREEN)Generando documentación...$(NC)"
	cd docs && make html
	@echo "$(GREEN)Documentación generada en docs/_build/html/index.html$(NC)"

migrate: ## Ejecutar migraciones de base de datos
	@echo "$(GREEN)Ejecutando migraciones...$(NC)"
	alembic upgrade head

migrate-create: ## Crear nueva migración
	@echo "$(GREEN)Creando nueva migración...$(NC)"
	@read -p "Nombre de la migración: " name; \
	alembic revision --autogenerate -m "$$name"

run: ## Ejecutar aplicación
	@echo "$(GREEN)Iniciando Quantum Trading Platform...$(NC)"
	python -m src.main

run-worker: ## Ejecutar worker de Celery
	celery -A src.core.tasks worker --loglevel=info

run-beat: ## Ejecutar scheduler de Celery
	celery -A src.core.tasks beat --loglevel=info

docker-build: ## Construir imágenes Docker
	docker-compose -f docker/docker-compose.yml build

docker-up: ## Iniciar servicios con Docker
	docker-compose -f docker/docker-compose.yml up -d

docker-down: ## Detener servicios Docker
	docker-compose -f docker/docker-compose.yml down

docker-logs: ## Ver logs de Docker
	docker-compose -f docker/docker-compose.yml logs -f

setup-pre-commit: ## Configurar pre-commit hooks
	pre-commit install
	pre-commit run --all-files

security-check: ## Verificar vulnerabilidades de seguridad
	bandit -r $(SRC_DIR)
	safety check

performance-test: ## Ejecutar tests de performance
	locust -f $(TEST_DIR)/performance/locustfile.py

monitor: ## Abrir dashboards de monitoreo
	@echo "$(GREEN)Abriendo dashboards...$(NC)"
	@echo "Grafana: http://localhost:3000"
	@echo "Prometheus: http://localhost:9090"
	@echo "Jaeger: http://localhost:16686"

backup: ## Crear backup de base de datos
	@echo "$(GREEN)Creando backup...$(NC)"
	docker exec quantum_postgres pg_dump -U postgres trading_db > backups/backup_$$(date +%Y%m%d_%H%M%S).sql

restore: ## Restaurar backup de base de datos
	@echo "$(GREEN)Restaurando backup...$(NC)"
	@read -p "Archivo de backup (en backups/): " file; \
	docker exec -i quantum_postgres psql -U postgres trading_db < backups/$$file