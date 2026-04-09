const CACHE_NAME = 'stlite-cache-v2';
const STLITE_VERSION = '0.75.0';

// Assets to cache immediately on install
const PRE_CACHE_ASSETS = [
  '/',
  '/streamlit_app.py',
  '/backtest.py',
  '/tickbus.py',
  '/ssl.py',
  `https://cdn.jsdelivr.net/npm/@stlite/mountable@${STLITE_VERSION}/build/stlite.css`,
  `https://cdn.jsdelivr.net/npm/@stlite/mountable@${STLITE_VERSION}/build/stlite.js`,
];

self.addEventListener('install', (event) => {
  event.waitUntil(
    caches.open(CACHE_NAME).then((cache) => {
      console.log('Opened cache');
      return cache.addAll(PRE_CACHE_ASSETS);
    })
  );
  self.skipWaiting();
});

self.addEventListener('activate', (event) => {
  event.waitUntil(
    caches.keys().then((cacheNames) => {
      return Promise.all(
        cacheNames.map((cacheName) => {
          if (cacheName !== CACHE_NAME) {
            return caches.delete(cacheName);
          }
        })
      );
    })
  );
  self.clients.claim();
});

self.addEventListener('fetch', (event) => {
  const url = new URL(event.request.url);

  // Cache-First strategy for heavy assets (Wheels, JS, CSS, WASM)
  if (
    url.hostname === 'cdn.jsdelivr.net' ||
    url.pathname.endsWith('.whl') ||
    url.pathname.endsWith('.wasm') ||
    url.pathname.endsWith('.js') ||
    url.pathname.endsWith('.css')
  ) {
    event.respondWith(
      caches.match(event.request).then((response) => {
        if (response) {
          return response;
        }
        return fetch(event.request).then((fetchResponse) => {
          // Only cache successful responses
          if (!fetchResponse || fetchResponse.status !== 200 || fetchResponse.type !== 'basic' && fetchResponse.type !== 'cors') {
            return fetchResponse;
          }
          const responseToCache = fetchResponse.clone();
          caches.open(CACHE_NAME).then((cache) => {
            cache.put(event.request, responseToCache);
          });
          return fetchResponse;
        });
      })
    );
    return;
  }

  // Stale-while-revalidate for local app files
  if (url.origin === self.location.origin) {
    event.respondWith(
      caches.match(event.request).then((cachedResponse) => {
        const fetchPromise = fetch(event.request).then((networkResponse) => {
          caches.open(CACHE_NAME).then((cache) => {
            cache.put(event.request, networkResponse.clone());
          });
          return networkResponse;
        });
        return cachedResponse || fetchPromise;
      })
    );
    return;
  }

  // Default: Network first
  event.respondWith(fetch(event.request));
});
