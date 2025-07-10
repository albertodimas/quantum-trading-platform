# Dashboard Web - Quantum Trading Platform

## 🚀 Descripción

Dashboard web moderno y profesional para la plataforma de trading cuántico. Desarrollado con React, TypeScript y Tailwind CSS, ofrece una interfaz intuitiva en español con actualizaciones en tiempo real.

## 📋 Características

### Páginas Principales
- **Dashboard**: Vista general con estadísticas, gráficos y estado del sistema
- **Trading**: Terminal de trading con libro de órdenes y gestión de posiciones
- **Estrategias**: Gestión y monitoreo de estrategias de trading
- **Análisis**: Análisis detallado con múltiples tipos de gráficos
- **Configuración**: Ajustes generales, API, notificaciones y seguridad

### Componentes Clave
- **PortfolioChart**: Evolución del portafolio con selector de rango temporal
- **AgentStatus**: Estado en tiempo real de los agentes IA
- **MarketOverview**: Vista general del mercado con mini gráficos
- **TradingSignals**: Señales de trading con indicadores de confianza
- **RecentTrades**: Historial de operaciones recientes con P&L

### Características Técnicas
- ⚡ Actualizaciones en tiempo real vía WebSocket
- 🌙 Modo oscuro/claro
- 📱 Diseño responsivo
- 🔄 Gestión de estado con React Query
- 🎨 Animaciones suaves con Framer Motion
- 🔐 Autenticación JWT
- 🌐 Interfaz completamente en español

## 🛠️ Tecnologías

- **React 18** - Framework UI
- **TypeScript** - Tipado estático
- **Vite** - Build tool ultrarrápido
- **Tailwind CSS** - Estilos utility-first
- **React Query** - Gestión de estado del servidor
- **Socket.io Client** - WebSocket para tiempo real
- **Recharts** - Visualización de datos
- **Framer Motion** - Animaciones
- **Radix UI** - Componentes accesibles
- **React Router** - Routing
- **Axios** - Cliente HTTP
- **date-fns** - Manipulación de fechas

## 📦 Instalación

```bash
# Instalar dependencias
npm install

# Configurar variables de entorno
cp .env.example .env
# Editar .env con tu configuración

# Desarrollo
npm run dev

# Build producción
npm run build

# Preview producción
npm run preview
```

## ⚙️ Configuración

### Variables de Entorno

```env
# API Configuration
VITE_API_BASE_URL=http://localhost:8000
VITE_WS_BASE_URL=ws://localhost:8000

# App Configuration
VITE_APP_NAME=Quantum Trading Platform
VITE_APP_VERSION=1.0.0

# Feature Flags
VITE_ENABLE_DEMO_MODE=true
VITE_ENABLE_LIVE_TRADING=false
```

## 📁 Estructura del Proyecto

```
src/
├── components/       # Componentes reutilizables
│   ├── Layout.tsx   # Layout principal con sidebar
│   ├── StatsCard.tsx # Tarjetas de estadísticas
│   └── ...          # Otros componentes
├── pages/           # Páginas principales
│   ├── Dashboard.tsx
│   ├── Trading.tsx
│   └── ...
├── services/        # Servicios y API
│   └── api.ts      # Cliente API y WebSocket
├── hooks/          # Custom hooks
│   ├── useMarketData.ts
│   ├── useTrading.ts
│   └── ...
├── styles/         # Estilos globales
└── App.tsx         # Componente raíz
```

## 🔌 API Endpoints

### Market Data
- `GET /api/market/:symbol` - Datos del mercado
- `GET /api/orderbook/:symbol` - Libro de órdenes
- `GET /api/ticker/:symbol` - Ticker
- `GET /api/candles/:symbol` - Velas

### Trading
- `GET /api/positions` - Posiciones abiertas
- `GET /api/orders` - Órdenes activas
- `POST /api/orders` - Crear orden
- `DELETE /api/orders/:id` - Cancelar orden

### Estrategias
- `GET /api/strategies` - Lista de estrategias
- `PUT /api/strategies/:id` - Actualizar estrategia
- `POST /api/strategies/:id/toggle` - Activar/desactivar

### Agentes IA
- `GET /api/agents/status` - Estado de agentes
- `GET /api/agents/:id/analysis` - Análisis del agente

## 🔄 WebSocket Events

### Suscripciones
- `subscribe_market` - Datos de mercado en tiempo real
- `subscribe_trades` - Actualizaciones de trades
- `subscribe_agents` - Estado de agentes IA

### Eventos
- `market_update` - Actualización de mercado
- `trade_update` - Nueva operación
- `agent_update` - Cambio en agente IA
- `position_opened` - Nueva posición
- `position_closed` - Posición cerrada

## 🎨 Personalización

### Tema
El tema se puede personalizar editando las variables CSS en `index.css`:

```css
:root {
  --primary: 0 70 243;
  --background: 255 255 255;
  --foreground: 0 0 0;
  /* ... más variables ... */
}
```

### Idioma
Todos los textos están en español. Para cambiar el idioma, buscar y reemplazar los strings en los componentes.

## 📊 Hooks Personalizados

### useMarketData
```typescript
const { marketData, orderBook, ticker, isLoading } = useMarketData('BTC/USDT');
```

### usePositions
```typescript
const { positions, isLoading, refetch } = usePositions();
```

### useStrategies
```typescript
const { strategies, isLoading } = useStrategies();
const toggleStrategy = useToggleStrategy();
```

### useAgentsStatus
```typescript
const { agents, isLoading } = useAgentsStatus();
```

## 🚀 Desarrollo

### Scripts NPM

- `npm run dev` - Servidor de desarrollo
- `npm run build` - Build de producción
- `npm run preview` - Preview del build
- `npm run lint` - Linting con ESLint
- `npm run type-check` - Verificación de tipos

### Mejores Prácticas

1. **Componentes**: Usar componentes funcionales con hooks
2. **Estado**: React Query para estado del servidor, useState/useContext para estado local
3. **Estilos**: Tailwind classes, evitar CSS inline
4. **Tipos**: Definir interfaces para todas las props y respuestas API
5. **Errores**: Manejar errores con try-catch y toast notifications

## 🐛 Debugging

### React Query DevTools
Las DevTools están incluidas en desarrollo:
```typescript
import { ReactQueryDevtools } from '@tanstack/react-query-devtools';
```

### WebSocket Debug
Activar logs en la consola:
```typescript
wsManager.enableDebug();
```

## 📈 Performance

- **Code Splitting**: Lazy loading de páginas
- **Memoización**: React.memo para componentes pesados
- **Virtualización**: Para listas largas (pendiente)
- **Optimización de re-renders**: useCallback y useMemo

## 🔐 Seguridad

- Autenticación JWT con refresh tokens
- HTTPS en producción
- Sanitización de inputs
- CORS configurado correctamente
- Variables de entorno para datos sensibles

## 📄 Licencia

Proprietary - Quantum Trading Platform