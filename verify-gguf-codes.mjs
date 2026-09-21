// Force fresh import by bypassing all caches
import { createRequire } from 'module';
import { fileURLToPath } from 'url';
import { dirname, resolve } from 'path';

const __filename = fileURLToPath(import.meta.url);
const __dirname = dirname(__filename);

// Clear module cache
const modulePath = resolve(__dirname, 'src/lib/backendError.ts');
if (require.cache[modulePath]) {
  delete require.cache[modulePath];
}

// Dynamic import to force fresh load
const { CODE_KEYS } = await import('./src/lib/backendError.ts');

console.log('GGUF_CONFIG_NOT_FOUND exists:', 'GGUF_CONFIG_NOT_FOUND' in CODE_KEYS);
console.log('GGUF_SERVER_NOT_FOUND exists:', 'GGUF_SERVER_NOT_FOUND' in CODE_KEYS);

if ('GGUF_CONFIG_NOT_FOUND' in CODE_KEYS && 'GGUF_SERVER_NOT_FOUND' in CODE_KEYS) {
  console.log('\n✅ Both GGUF error codes are present in CODE_KEYS');
  process.exit(0);
} else {
  console.log('\n❌ GGUF error codes NOT found in CODE_KEYS');
  process.exit(1);
}
