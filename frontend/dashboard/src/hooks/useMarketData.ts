import { useQuery, useQueryClient } from '@tanstack/react-query';
import { useEffect } from 'react';
import { tradingAPI, wsManager, wsEvents } from '../services/api';

export const useMarketData = (symbol: string) => {
  const queryClient = useQueryClient();

  // Obtener datos del mercado
  const marketDataQuery = useQuery({
    queryKey: ['marketData', symbol],
    queryFn: () => tradingAPI.getMarketData(symbol),
    refetchInterval: 30000, // Actualizar cada 30 segundos
  });

  // Obtener libro de órdenes
  const orderBookQuery = useQuery({
    queryKey: ['orderBook', symbol],
    queryFn: () => tradingAPI.getOrderBook(symbol),
    refetchInterval: 5000, // Actualizar cada 5 segundos
  });

  // Obtener ticker
  const tickerQuery = useQuery({
    queryKey: ['ticker', symbol],
    queryFn: () => tradingAPI.getTicker(symbol),
    refetchInterval: 2000, // Actualizar cada 2 segundos
  });

  // Suscribirse a actualizaciones en tiempo real
  useEffect(() => {
    // Conectar WebSocket
    wsManager.connect();

    // Suscribirse al mercado
    wsEvents.subscribeToMarket(symbol);

    // Escuchar actualizaciones
    const handleMarketUpdate = (data: any) => {
      if (data.symbol === symbol) {
        // Actualizar cache de React Query
        queryClient.setQueryData(['marketData', symbol], (oldData: any) => ({
          ...oldData,
          ...data,
        }));

        if (data.orderBook) {
          queryClient.setQueryData(['orderBook', symbol], data.orderBook);
        }

        if (data.ticker) {
          queryClient.setQueryData(['ticker', symbol], data.ticker);
        }
      }
    };

    wsEvents.onMarketUpdate(handleMarketUpdate);

    // Cleanup
    return () => {
      wsEvents.unsubscribeFromMarket(symbol);
      wsManager.off('market_update', handleMarketUpdate);
    };
  }, [symbol, queryClient]);

  return {
    marketData: marketDataQuery.data?.data,
    orderBook: orderBookQuery.data?.data,
    ticker: tickerQuery.data?.data,
    isLoading: marketDataQuery.isLoading || orderBookQuery.isLoading || tickerQuery.isLoading,
    error: marketDataQuery.error || orderBookQuery.error || tickerQuery.error,
  };
};

export const useCandles = (symbol: string, interval: string = '1h', limit: number = 100) => {
  return useQuery({
    queryKey: ['candles', symbol, interval, limit],
    queryFn: () => tradingAPI.getCandles(symbol, interval, limit),
    refetchInterval: 60000, // Actualizar cada minuto
  });
};