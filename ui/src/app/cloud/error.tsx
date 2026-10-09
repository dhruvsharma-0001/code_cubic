'use client';

import { useEffect } from 'react';
import { GlassCard } from '@/components/shared';

export default function ErrorBoundary({
  error,
  reset,
}: {
  error: Error & { digest?: string };
  reset: () => void;
}) {
  useEffect(() => {
    console.error("ErrorBoundary caught:", error);
  }, [error]);

  return (
    <div className="p-8 w-full h-full flex flex-col items-center justify-center">
      <GlassCard className="max-w-md p-6 border-red-500/30 bg-red-500/5">
        <h2 className="text-xl font-bold text-red-400 mb-4">Connection Failed</h2>
        <p className="text-sm text-gray-400 mb-6">
          Unable to communicate with the edge node. The service might be offline or unreachable.
        </p>
        <p className="text-xs text-red-400/70 mb-6 font-mono p-2 bg-black/40 rounded">
          {error.message || "Unknown API error"}
        </p>
        <button
          onClick={reset}
          className="px-4 py-2 bg-red-500/20 text-red-300 rounded-md hover:bg-red-500/30 transition-colors"
        >
          Try again
        </button>
      </GlassCard>
    </div>
  );
}
