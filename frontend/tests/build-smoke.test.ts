import { describe, it, expect } from 'vitest';
import { execFileSync } from 'node:child_process';
import { existsSync, readFileSync, readdirSync } from 'node:fs';
import { resolve, dirname } from 'node:path';
import { fileURLToPath } from 'node:url';

// CHARACTERIZATION build-smoke tests. Given the thin pure-logic surface, the
// type-check is the single most valuable lock: it catches any change to
// protocol.ts types or component prop contracts that breaks strict checking
// (strict, noUnusedLocals, noUnusedParameters, noFallthroughCasesInSwitch).
//
// These shell out to the project toolchain. They are deterministic and
// hermetic (no network, no dev server). They are slower than the unit tests;
// the per-test timeout is raised in vitest.config.ts.

const __dirname = dirname(fileURLToPath(import.meta.url));
const projectRoot = resolve(__dirname, '..');

function run(cmd: string, args: string[]): { code: number; out: string } {
  try {
    const out = execFileSync(cmd, args, {
      cwd: projectRoot,
      encoding: 'utf8',
      stdio: ['ignore', 'pipe', 'pipe'],
    });
    return { code: 0, out };
  } catch (e: any) {
    return {
      code: typeof e.status === 'number' ? e.status : 1,
      out: `${e.stdout ?? ''}${e.stderr ?? ''}`,
    };
  }
}

describe('build smoke: tsc --noEmit type-checks the production sources', () => {
  it('exits 0 with no type errors over tsconfig.json', () => {
    // Only type-checks ["src"] per tsconfig.json include; the test files in
    // ./tests are intentionally excluded so they cannot affect this lock.
    const { code, out } = run('bunx', ['tsc', '--noEmit', '-p', 'tsconfig.json']);
    expect(out).toBe('');
    expect(code).toBe(0);
  });
});

describe('build smoke: vite production build succeeds', () => {
  it('emits dist/index.html and a dist/assets/*.js bundle (no hash assertions)', () => {
    // The real build script is `tsc && vite build`; we run `vite build`
    // directly since tsc is exercised above. We do NOT assert hashed asset
    // filenames or byte sizes (non-deterministic across tool versions).
    const { code, out } = run('bunx', ['vite', 'build']);
    expect(code, out).toBe(0);

    const indexHtml = resolve(projectRoot, 'dist', 'index.html');
    expect(existsSync(indexHtml)).toBe(true);

    // Root mount div the SPA bootstraps into.
    expect(readFileSync(indexHtml, 'utf8')).toContain('id="root"');

    const assetsDir = resolve(projectRoot, 'dist', 'assets');
    expect(existsSync(assetsDir)).toBe(true);
    const jsBundles = readdirSync(assetsDir).filter((f) => f.endsWith('.js'));
    expect(jsBundles.length).toBeGreaterThan(0);
  });
});
