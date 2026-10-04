import { defineConfig } from 'vite';

export default defineConfig({
  // Relative asset URLs, so the same build works at a domain root, at
  // /Tera-Dark/aetheria/, and from a file:// path. Hard-coding the repo name here
  // would mean editing the config to deploy anywhere else.
  base: './',
  server: { port: 5180, open: false },
  build: {
    target: 'es2022',
    // A wallpaper is loaded once and then runs for hours, so the whole engine
    // should arrive in one request rather than waterfalling chunks.
    modulePreload: { polyfill: false },
  },
});
