'use client';

import { useEffect, useRef } from 'react';

export default function Home() {
  const mountPoint = useRef<HTMLDivElement>(null);

  useEffect(() => {
    if (typeof window !== 'undefined' && mountPoint.current) {
      // Dynamically import stlite to avoid SSR issues and ensure browser environment
      import('@stlite/mountable').then((stlite) => {
        stlite.mount({
          requirements: [
            "pandas",
            "numpy",
            "matplotlib",
            "plotly",
            "ta",
            "joblib",
            "scikit-learn",
            "breeze-connect",
            "pytz"
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
            }
          },
          container: mountPoint.current,
        });
      });
    }
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
        backgroundColor: '#0e1117' // Match Streamlit's dark theme default
      }} 
    />
  );
}
