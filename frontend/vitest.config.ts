import { defineConfig } from 'vitest/config';
import react from '@vitejs/plugin-react';

// Test-only config. Kept separate from vite.config.ts so the production
// build (tsc && vite build) is untouched. Tests live in ./tests, outside
// the production tsconfig "include": ["src"], so `tsc` in the build script
// never type-checks them.
export default defineConfig({
  plugins: [react()],
  test: {
    environment: 'jsdom',
    globals: true,
    setupFiles: ['./tests/setup.ts'],
    include: ['tests/**/*.test.{ts,tsx}'],
    // The build-smoke test shells out to tsc/vite build which can be slow.
    testTimeout: 180000,
  },
});
