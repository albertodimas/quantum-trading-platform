import { useQuery, useMutation, useQueryClient } from '@tanstack/react-query';
import toast from 'react-hot-toast';
import { tradingAPI } from '../services/api';

export const useStrategies = () => {
  const strategiesQuery = useQuery({
    queryKey: ['strategies'],
    queryFn: () => tradingAPI.getStrategies(),
    refetchInterval: 30000, // Actualizar cada 30 segundos
  });

  return {
    strategies: strategiesQuery.data?.data || [],
    isLoading: strategiesQuery.isLoading,
    error: strategiesQuery.error,
    refetch: strategiesQuery.refetch,
  };
};

export const useStrategyStatus = (strategyId: string) => {
  return useQuery({
    queryKey: ['strategyStatus', strategyId],
    queryFn: () => tradingAPI.getStrategyStatus(strategyId),
    refetchInterval: 10000, // Actualizar cada 10 segundos
    enabled: !!strategyId,
  });
};

export const useToggleStrategy = () => {
  const queryClient = useQueryClient();

  return useMutation({
    mutationFn: ({ strategyId, active }: { strategyId: string; active: boolean }) =>
      tradingAPI.toggleStrategy(strategyId, active),
    onSuccess: (data, variables) => {
      queryClient.invalidateQueries({ queryKey: ['strategies'] });
      queryClient.invalidateQueries({ queryKey: ['strategyStatus', variables.strategyId] });
      
      const action = variables.active ? 'activada' : 'desactivada';
      toast.success(`Estrategia ${action} exitosamente`);
    },
    onError: (error: any) => {
      toast.error(error.response?.data?.message || 'Error al cambiar el estado de la estrategia');
    },
  });
};

export const useUpdateStrategy = () => {
  const queryClient = useQueryClient();

  return useMutation({
    mutationFn: ({ strategyId, data }: { strategyId: string; data: any }) =>
      tradingAPI.updateStrategy(strategyId, data),
    onSuccess: (data, variables) => {
      queryClient.invalidateQueries({ queryKey: ['strategies'] });
      queryClient.invalidateQueries({ queryKey: ['strategyStatus', variables.strategyId] });
      toast.success('Estrategia actualizada exitosamente');
    },
    onError: (error: any) => {
      toast.error(error.response?.data?.message || 'Error al actualizar la estrategia');
    },
  });
};