'use client';

import { useEffect, useRef, useState } from 'react';

const LOADING_STEPS = [
  "Initializing AI Engine...",
  "Loading Data Science Libraries...",
  "Booting Python Runtime...",
  "Injecting Technical Indicators...",
  "Starting Dashboard..."
];

export default function Home() {
  const mountPoint = useRef<HTMLDivElement>(null);

  const [isLoading, setIsLoading] = useState(true);
  const [loadingStep, setLoadingStep] = useState(0);

  useEffect(() => {
    // Register Service Worker
    if ('serviceWorker' in navigator) {
      window.addEventListener('load', () => {
        navigator.serviceWorker.register('/sw.js').then(
          (registration) => console.log('SW register success: ', registration.scope),
          (err) => console.log('SW register fail: ', err)
        );
      });
    }

    // Step animation for loading
    const stepInterval = setInterval(() => {
      setLoadingStep((prev) => (prev < LOADING_STEPS.length - 1 ? prev + 1 : prev));
    }, 4000);

    // Inject the stlite styles (CDN)
    const link = document.createElement('link');
    link.rel = 'stylesheet';
    link.href = 'https://cdn.jsdelivr.net/npm/@stlite/mountable@0.75.0/build/stlite.css';
    document.head.appendChild(link);

    // Inject the stlite script (CDN)
    const script = document.createElement('script');
    script.src = 'https://cdn.jsdelivr.net/npm/@stlite/mountable@0.75.0/build/stlite.js';
    script.async = true;
    script.onload = () => {
      if ((window as any).stlite && mountPoint.current) {
        (window as any).stlite.mount({
          requirements: [
            "pandas", "numpy", "plotly", 
            "breeze-connect", "pytz"
          ],
          entrypoint: "streamlit_app.py",
          files: {
            "streamlit_app.py": { url: "/streamlit_app.py" },
            "backtest.py": { url: "/backtest.py" },
            "tickbus.py": { url: "/tickbus.py" },
            "ssl.py": { url: "/ssl.py" },
            "ta/__init__.py": { url: "/ta/__init__.py" },
            "ta/momentum.py": { url: "/ta/momentum.py" },
            "ta/others.py": { url: "/ta/others.py" },
            "ta/trend.py": { url: "/ta/trend.py" },
            "ta/utils.py": { url: "/ta/utils.py" },
            "ta/volatility.py": { url: "/ta/volatility.py" },
            "ta/volume.py": { url: "/ta/volume.py" },
            "ta/wrapper.py": { url: "/ta/wrapper.py" },
          },
          container: mountPoint.current,
        });

        // Dynamic "Ready" detection
        const checkReady = setInterval(() => {
          const iframe = mountPoint.current?.querySelector('iframe');
          if (iframe) {
            setIsLoading(false);
            clearInterval(checkReady);
            clearInterval(stepInterval);
          }
        }, 500);
      }
    };
    document.body.appendChild(script);

    return () => {
      document.head.removeChild(link);
      if (document.body.contains(script)) {
        document.body.removeChild(script);
      }
      clearInterval(stepInterval);
    };
  }, []);

  return (
    <>
      {isLoading && (
        <div style={{
          position: 'fixed',
          top: 0,
          left: 0,
          width: '100vw',
          height: '100vh',
          backgroundColor: '#0e1117',
          display: 'flex',
          flexDirection: 'column',
          alignItems: 'center',
          justifyContent: 'center',
          zIndex: 9999,
          color: 'white',
          fontFamily: '"Inter", sans-serif',
        }}>
          {/* Animated Logic Core */}
          <div style={{
            width: '80px',
            height: '80px',
            border: '4px solid rgba(255, 255, 255, 0.1)',
            borderTop: '4px solid #ff4b4b',
            borderRadius: '50%',
            animation: 'spin 1.5s linear infinite',
            marginBottom: '30px',
            boxShadow: '0 0 20px rgba(255, 75, 75, 0.3)'
          }} />
          
          <h2 style={{ 
            fontSize: '1.5rem', 
            fontWeight: 500, 
            marginBottom: '10px',
            letterSpacing: '-0.02em'
          }}>
            {LOADING_STEPS[loadingStep]}
          </h2>
          
          <div style={{
            width: '200px',
            height: '4px',
            backgroundColor: 'rgba(255, 255, 255, 0.1)',
            borderRadius: '2px',
            overflow: 'hidden',
            marginTop: '20px'
          }}>
            <div style={{
              width: `${((loadingStep + 1) / LOADING_STEPS.length) * 100}%`,
              height: '100%',
              backgroundColor: '#ff4b4b',
              transition: 'width 0.5s ease-out'
            }} />
          </div>

          <p style={{
            marginTop: '40px',
            fontSize: '0.8rem',
            color: 'rgba(255, 255, 255, 0.4)',
            letterSpacing: '0.05em',
            textTransform: 'uppercase'
          }}>
            Powered by stlite & Pyodide
          </p>

          <style>{`
            @keyframes spin {
              0% { transform: rotate(0deg); }
              100% { transform: rotate(360deg); }
            }
          `}</style>
        </div>
      )}
      <div 
        ref={mountPoint} 
        style={{ 
          width: '100vw', 
          height: '100vh', 
          position: 'fixed', 
          top: 0, 
          left: 0,
          backgroundColor: '#0e1117',
          visibility: isLoading ? 'hidden' : 'visible'
        }} 
      />
    </>
  );
}
