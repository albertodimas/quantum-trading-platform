import { useQuery } from '@tanstack/react-query';
import { useEffect } from 'react';
import { tradingAPI, wsManager, wsEvents } from '../services/api';

export const useAgentsStatus = () => {
  const agentsQuery = useQuery({
    queryKey: ['agentsStatus'],
    queryFn: () => tradingAPI.getAgentsStatus(),
    refetchInterval: 15000, // Actualizar cada 15 segundos
  });

  // Suscribirse a actualizaciones en tiempo real
  useEffect(() => {
    wsManager.connect();
    wsEvents.subscribeToAgents();

    const handleAgentUpdate = (data: any) => {
      // Los datos se actualizarán automáticamente con el refetchInterval
      // Aquí podrías manejar notificaciones específicas si lo deseas
      console.log('Agent update:', data);
    };

    wsEvents.onAgentUpdate(handleAgentUpdate);

    return () => {
      wsManager.off('agent_update', handleAgentUpdate);
    };
  }, []);

  return {
    agents: agentsQuery.data?.data || [],
    isLoading: agentsQuery.isLoading,
    error: agentsQuery.error,
    refetch: agentsQuery.refetch,
  };
};

export const useAgentAnalysis = (agentId: string) => {
  return useQuery({
    queryKey: ['agentAnalysis', agentId],
    queryFn: () => tradingAPI.getAgentAnalysis(agentId),
    refetchInterval: 30000, // Actualizar cada 30 segundos
    enabled: !!agentId,
  });
};