import { useCallback, useEffect, useRef, useState } from 'react';
import type { ServerMessage, ClientMessage, AnimationFrame, Status, NetworkTopology, TrainingUpdate } from '../lib/protocol';

// Diagnostic console logging is dev-only noise; gate it behind the Vite DEV flag
// so production bundles stay quiet. Behavior (state updates, reconnect, send) is
// unchanged either way. `import.meta.env` is injected by Vite; read it through a
// local cast so this compiles without the ambient vite/client types.
const DEV = (import.meta as { env?: { DEV?: boolean } }).env?.DEV ?? false;
const devLog = (...args: unknown[]) => {
  if (DEV) console.log(...args);
};
const devWarn = (...args: unknown[]) => {
  if (DEV) console.warn(...args);
};
const devError = (...args: unknown[]) => {
  if (DEV) console.error(...args);
};

interface UseWebSocketReturn {
  connected: boolean;
  frame: AnimationFrame | null;
  status: Status | null;
  topology: NetworkTopology | null;
  training: TrainingUpdate | null;
  error: string | null;
  send: (msg: ClientMessage) => void;
}

export function useWebSocket(url: string): UseWebSocketReturn {
  const wsRef = useRef<WebSocket | null>(null);
  const [connected, setConnected] = useState(false);
  const [frame, setFrame] = useState<AnimationFrame | null>(null);
  const [status, setStatus] = useState<Status | null>(null);
  const [topology, setTopology] = useState<NetworkTopology | null>(null);
  const [training, setTraining] = useState<TrainingUpdate | null>(null);
  const [error, setError] = useState<string | null>(null);

  const reconnectTimeoutRef = useRef<number | null>(null);

  const connect = useCallback(() => {
    if (wsRef.current?.readyState === WebSocket.OPEN) return;

    const ws = new WebSocket(url);
    wsRef.current = ws;

    ws.onopen = () => {
      setConnected(true);
      setError(null);
      devLog('WebSocket connected');
    };

    ws.onclose = () => {
      setConnected(false);
      devLog('WebSocket disconnected, reconnecting...');
      // Reconnect after 2 seconds
      reconnectTimeoutRef.current = window.setTimeout(connect, 2000);
    };

    ws.onerror = (e) => {
      devError('WebSocket error:', e);
      setError('Connection error');
    };

    ws.onmessage = (event) => {
      try {
        const msg = JSON.parse(event.data) as ServerMessage | NetworkTopology;

        // Check if it's a NetworkTopology message (no 'type' field, has 'layer_sizes')
        if ('layer_sizes' in msg) {
          setTopology(msg);
          return;
        }

        switch (msg.type) {
          case 'AnimationFrame':
            setFrame(msg);
            break;
          case 'Status':
            setStatus(msg);
            break;
          case 'TrainingUpdate':
            setTraining(msg);
            break;
          case 'Error':
            setError(msg.message);
            break;
          case 'Ack':
            // Command acknowledged
            break;
          case 'WeightMatrix':
            // Handle weight matrix visualization
            break;
        }
      } catch (e) {
        devError('Failed to parse message:', e);
      }
    };
  }, [url]);

  useEffect(() => {
    connect();
    return () => {
      if (reconnectTimeoutRef.current) {
        clearTimeout(reconnectTimeoutRef.current);
      }
      wsRef.current?.close();
    };
  }, [connect]);

  const send = useCallback((msg: ClientMessage) => {
    if (wsRef.current?.readyState === WebSocket.OPEN) {
      const json = JSON.stringify(msg);
      devLog('WebSocket sending:', json);
      wsRef.current.send(json);
    } else {
      devWarn('WebSocket not connected, cannot send:', msg);
    }
  }, []);

  return { connected, frame, status, topology, training, error, send };
}
