import { useQuery, useMutation, useQueryClient } from '@tanstack/react-query';
import { useEffect } from 'react';
import toast from 'react-hot-toast';
import { tradingAPI, wsManager, wsEvents } from '../services/api';

export const usePositions = () => {
  const queryClient = useQueryClient();

  const positionsQuery = useQuery({
    queryKey: ['positions'],
    queryFn: () => tradingAPI.getPositions(),
    refetchInterval: 10000, // Actualizar cada 10 segundos
  });

  // Suscribirse a actualizaciones de trades
  useEffect(() => {
    wsManager.connect();
    wsEvents.subscribeToTrades();

    const handleTradeUpdate = (data: any) => {
      // Actualizar posiciones cuando hay un nuevo trade
      queryClient.invalidateQueries({ queryKey: ['positions'] });
      
      // Mostrar notificación
      if (data.type === 'position_opened') {
        toast.success(`Posición abierta: ${data.symbol}`);
      } else if (data.type === 'position_closed') {
        const pnl = data.pnl;
        if (pnl > 0) {
          toast.success(`Posición cerrada con ganancia: +$${pnl.toFixed(2)}`);
        } else {
          toast.error(`Posición cerrada con pérdida: -$${Math.abs(pnl).toFixed(2)}`);
        }
      }
    };

    wsEvents.onTradeUpdate(handleTradeUpdate);

    return () => {
      wsManager.off('trade_update', handleTradeUpdate);
    };
  }, [queryClient]);

  return {
    positions: positionsQuery.data?.data || [],
    isLoading: positionsQuery.isLoading,
    error: positionsQuery.error,
    refetch: positionsQuery.refetch,
  };
};

export const useOrders = () => {
  const queryClient = useQueryClient();

  const ordersQuery = useQuery({
    queryKey: ['orders'],
    queryFn: () => tradingAPI.getOrders(),
    refetchInterval: 5000, // Actualizar cada 5 segundos
  });

  return {
    orders: ordersQuery.data?.data || [],
    isLoading: ordersQuery.isLoading,
    error: ordersQuery.error,
    refetch: ordersQuery.refetch,
  };
};

export const useCreateOrder = () => {
  const queryClient = useQueryClient();

  return useMutation({
    mutationFn: (orderData: any) => tradingAPI.createOrder(orderData),
    onSuccess: (data) => {
      queryClient.invalidateQueries({ queryKey: ['orders'] });
      toast.success('Orden creada exitosamente');
    },
    onError: (error: any) => {
      toast.error(error.response?.data?.message || 'Error al crear la orden');
    },
  });
};

export const useCancelOrder = () => {
  const queryClient = useQueryClient();

  return useMutation({
    mutationFn: (orderId: string) => tradingAPI.cancelOrder(orderId),
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: ['orders'] });
      toast.success('Orden cancelada');
    },
    onError: (error: any) => {
      toast.error(error.response?.data?.message || 'Error al cancelar la orden');
    },
  });
};

export const useTradeHistory = (limit?: number) => {
  return useQuery({
    queryKey: ['tradeHistory', limit],
    queryFn: () => tradingAPI.getTradeHistory(limit),
    refetchInterval: 30000, // Actualizar cada 30 segundos
  });
};