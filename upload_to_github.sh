#!/bin/bash

# Script para subir el Quantum Trading Platform a GitHub
echo "🚀 Subiendo Quantum Trading Platform a GitHub..."

# Verificar que estamos en el directorio correcto
if [ ! -f "README.md" ] || [ ! -d "src" ]; then
    echo "❌ Error: No estamos en el directorio del proyecto"
    exit 1
fi

# Verificar estado del repositorio
echo "📋 Estado del repositorio:"
git status

# Verificar remote
echo "🔗 Remote configurado:"
git remote -v

# Intentar push
echo "⬆️  Subiendo código a GitHub..."
git push -u origin main

if [ $? -eq 0 ]; then
    echo "✅ ¡Código subido exitosamente a GitHub!"
    echo "🌐 Repositorio disponible en: https://github.com/albertodimasmorazaldivar/quantum-trading-platform"
    
    # Mostrar estadísticas del proyecto
    echo ""
    echo "📊 Estadísticas del proyecto:"
    echo "- Archivos Python: $(find src -name "*.py" | wc -l)"
    echo "- Archivos TypeScript: $(find frontend -name "*.ts" -o -name "*.tsx" | wc -l)"
    echo "- Tests: $(find tests -name "*.py" | wc -l)"
    echo "- Documentación: $(find docs -name "*.md" | wc -l)"
    echo "- Total líneas de código:"
    find src -name "*.py" -exec wc -l {} + | tail -1
    
else
    echo "❌ Error al subir el código. Verifique que el repositorio exista en GitHub."
    echo "Crear repositorio en: https://github.com/new"
    echo "Nombre: quantum-trading-platform"
fi