'use client';

import { useEffect, useRef } from 'react';

export default function Home() {
  const mountPoint = useRef<HTMLDivElement>(null);

  useEffect(() => {
    // Inject the stlite styles
    const link = document.createElement('link');
    link.rel = 'stylesheet';
    link.href = 'https://cdn.jsdelivr.net/npm/@stlite/mountable@0.75.0/build/stlite.css';
    document.head.appendChild(link);

    // Inject the stlite script
    const script = document.createElement('script');
    script.src = 'https://cdn.jsdelivr.net/npm/@stlite/mountable@0.75.0/build/stlite.js';
    script.async = true;
    script.onload = () => {
      if ((window as any).stlite && mountPoint.current) {
        (window as any).stlite.mount({
          requirements: [
            "pandas",
            "numpy",
            "matplotlib",
            "plotly",
            // "ta", // Bundled manually below
            "joblib",
            "scikit-learn",
            "breeze-connect",
            "pytz",
            "seaborn"
          ],
          entrypoint: "streamlit_app.py",
          files: {
            "streamlit_app.py": {
              url: "/streamlit_app.py"
            },
            "backtest.py": {
              url: "/backtest.py"
            },
            "tickbus.py": {
              url: "/tickbus.py"
            },
            // Mock compatibility layer
            "ssl.py": { url: "/ssl.py" },
            // Bundled ta library files
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
      }
    };
    document.body.appendChild(script);

    return () => {
      document.head.removeChild(link);
      if (document.body.contains(script)) {
        document.body.removeChild(script);
      }
    };
  }, []);

  return (
    <div 
      ref={mountPoint} 
      style={{ 
        width: '100vw', 
        height: '100vh', 
        position: 'fixed', 
        top: 0, 
        left: 0,
        backgroundColor: '#0e1117' 
      }} 
    />
  );
}
