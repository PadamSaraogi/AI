"use client";

import React, { useState, useEffect } from 'react';
import { LayoutDashboard, TrendingUp, Activity, BarChart3, Upload, Shield } from 'lucide-react';
import { motion, AnimatePresence } from 'framer-motion';
import dynamic from 'next/dynamic';

// Dynamically import Plotly to avoid SSR issues
const Plot = dynamic(() => import('react-plotly.js'), { ssr: false }) as any;

export default function Home() {
  const [activeTab, setActiveTab] = useState('backtesting');
  const [isUploading, setIsUploading] = useState(false);
  const [backtestResults, setBacktestResults] = useState<any>(null);
  
  // States for backtest parameters
  const [capital, setCapital] = useState(100000);
  const [risk, setRisk] = useState(1.0);
  const [signalFiles, setSignalFiles] = useState<FileList | null>(null);
  const [gridFiles, setGridFiles] = useState<FileList | null>(null);

  const handleBacktest = async () => {
    if (!signalFiles || !gridFiles) {
      alert("Please upload both signal and grid files!");
      return;
    }

    setIsUploading(true);
    const formData = new FormData();
    Array.from(signalFiles).forEach(file => formData.append('signal_files', file));
    Array.from(gridFiles).forEach(file => formData.append('grid_files', file));
    formData.append('total_capital', capital.toString());
    formData.append('risk_per_trade', risk.toString());

    try {
      const response = await fetch('/api/backtest', {
        method: 'POST',
        body: formData,
      });
      const data = await response.json();
      setBacktestResults(data);
    } catch (error) {
      console.error("Backtest failed:", error);
      alert("Backtest failed. Check the console for details.");
    } finally {
      setIsUploading(false);
    }
  };

  return (
    <main>
      {/* Sidebar / Configuration */}
      <div className="sidebar" style={{
        position: 'fixed',
        left: 0,
        top: 0,
        height: '100vh',
        width: '320px',
        backgroundColor: 'rgba(255, 255, 255, 0.02)',
        borderRight: '1px solid var(--card-border)',
        padding: '32px',
        zIndex: 100
      }}>
        <div style={{ display: 'flex', alignItems: 'center', gap: '12px', marginBottom: '48px' }}>
          <div style={{ backgroundColor: 'var(--primary)', padding: '8px', borderRadius: '8px' }}>
            <Activity size={24} color="#000" />
          </div>
          <h2 style={{ fontSize: '1.25rem', fontWeight: '700' }}>AI Trading</h2>
        </div>

        <div style={{ display: 'flex', flexDirection: 'column', gap: '24px' }}>
          <div>
            <label className="metric-label" style={{ display: 'block', marginBottom: '8px' }}>Portfolio Capital (₹)</label>
            <input 
              type="number" 
              value={capital}
              onChange={(e) => setCapital(Number(e.target.value))}
              style={{
                width: '100%',
                padding: '12px',
                borderRadius: '8px',
                background: 'rgba(255,255,255,0.05)',
                border: '1px solid var(--card-border)',
                color: '#fff',
                fontSize: '1rem'
              }}
            />
          </div>

          <div>
            <label className="metric-label" style={{ display: 'block', marginBottom: '8px' }}>Risk per Trade (%)</label>
            <input 
              type="range" 
              min="0.1" 
              max="10" 
              step="0.1"
              value={risk}
              onChange={(e) => setRisk(Number(e.target.value))}
              style={{ width: '100%' }}
            />
            <div style={{ textAlign: 'right', fontSize: '0.8rem', marginTop: '4px' }}>{risk}%</div>
          </div>

          <div className="glass-card" style={{ padding: '16px' }}>
            <label className="metric-label" style={{ display: 'flex', alignItems: 'center', gap: '8px', marginBottom: '12px' }}>
              <Upload size={14} /> Signals CSV
            </label>
            <input 
              type="file" 
              multiple 
              accept=".csv"
              onChange={(e) => setSignalFiles(e.target.files)}
              style={{ fontSize: '0.8rem', color: 'rgba(255,255,255,0.5)' }}
            />
          </div>

          <div className="glass-card" style={{ padding: '16px' }}>
            <label className="metric-label" style={{ display: 'flex', alignItems: 'center', gap: '8px', marginBottom: '12px' }}>
              <Upload size={14} /> Grid Search CSV
            </label>
            <input 
              type="file" 
              multiple 
              accept=".csv"
              onChange={(e) => setGridFiles(e.target.files)}
              style={{ fontSize: '0.8rem', color: 'rgba(255,255,255,0.5)' }}
            />
          </div>

          <button 
            className="btn btn-primary" 
            style={{ width: '100%', padding: '16px', fontSize: '1rem' }}
            onClick={handleBacktest}
            disabled={isUploading}
          >
            {isUploading ? "Running Simulation..." : "Run Simulation"}
          </button>
        </div>
      </div>

      {/* Main Content */}
      <div style={{ marginLeft: '320px', padding: '48px' }}>
        <header style={{ marginBottom: '48px' }}>
          <div className="tabs-header">
            <div 
              className={`tab-item ${activeTab === 'backtesting' ? 'active' : ''}`}
              onClick={() => setActiveTab('backtesting')}
            >
              Backtesting
            </div>
            <div 
              className={`tab-item ${activeTab === 'live' ? 'active' : ''}`}
              onClick={() => setActiveTab('live')}
            >
              Live Trading
            </div>
          </div>
        </header>

        <AnimatePresence mode="wait">
          {activeTab === 'backtesting' ? (
            <motion.div
              key="backtesting"
              initial={{ opacity: 0, y: 10 }}
              animate={{ opacity: 1, y: 0 }}
              exit={{ opacity: 0, y: -10 }}
            >
              {!backtestResults ? (
                <div style={{ textAlign: 'center', padding: '100px 0', opacity: 0.5 }}>
                  <TrendingUp size={48} style={{ marginBottom: '16px' }} />
                  <p>Upload data files and run simulation to see analytics.</p>
                </div>
              ) : (
                <div className="grid">
                  {/* Portfolio Metrics */}
                  <div className="grid grid-cols-3">
                    <div className="glass-card">
                      <div className="metric-label">Collective Net PnL</div>
                      <div className="metric-value" style={{ color: 'var(--primary)' }}>
                        ₹{Object.values(backtestResults).reduce((acc: any, curr: any) => acc + curr.metrics.net_pnl, 0).toLocaleString()}
                      </div>
                    </div>
                    <div className="glass-card">
                      <div className="metric-label">Avg Win Rate</div>
                      <div className="metric-value">
                        {(Object.values(backtestResults).reduce((acc: any, curr: any) => acc + curr.metrics.win_rate, 0) / Object.keys(backtestResults).length * 100).toFixed(1)}%
                      </div>
                    </div>
                    <div className="glass-card">
                      <div className="metric-label">Total Trades</div>
                      <div className="metric-value">
                        {Object.values(backtestResults).reduce((acc: any, curr: any) => acc + curr.metrics.total_trades, 0)}
                      </div>
                    </div>
                  </div>

                  {/* Per Symbol Analysis Section */}
                  <div style={{ marginTop: '32px' }}>
                    <h3 style={{ marginBottom: '24px' }}>Analyze Individual Stock</h3>
                    <div className="grid grid-cols-3" style={{ marginBottom: '24px' }}>
                      {Object.keys(backtestResults).map(symbol => (
                        <div 
                          key={symbol} 
                          className="glass-card" 
                          style={{ cursor: 'pointer', textAlign: 'center' }}
                          onClick={() => {
                            // Find element and scroll or just highlight
                          }}
                        >
                          <div className="metric-label">{symbol.toUpperCase()}</div>
                          <div className="metric-value">₹{backtestResults[symbol].metrics.net_pnl.toLocaleString()}</div>
                          <div style={{ fontSize: '0.8rem', opacity: 0.6 }}>Win Rate: {(backtestResults[symbol].metrics.win_rate * 100).toFixed(1)}%</div>
                        </div>
                      ))}
                    </div>

                    {Object.entries(backtestResults).map(([symbol, data]: [string, any]) => (
                      <div key={symbol} className="glass-card" style={{ marginBottom: '24px' }}>
                        <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: '16px' }}>
                          <h4 style={{ color: 'var(--primary)' }}>{symbol.toUpperCase()} Detailed Analytics</h4>
                          <div className="metric-label">{data.metrics.total_trades} Trades Executed</div>
                        </div>
                        <Plot
                          data={[{
                            x: Object.keys(data.equity_curve),
                            y: Object.values(data.equity_curve),
                            type: 'scatter',
                            fill: 'tozeroy',
                            fillcolor: 'rgba(16, 185, 129, 0.1)',
                            line: { color: 'var(--primary)' },
                            name: 'Equity Curve'
                          }]}
                          layout={{
                            autosize: true,
                            paper_bgcolor: 'rgba(0,0,0,0)',
                            plot_bgcolor: 'rgba(0,0,0,0)',
                            font: { color: '#fff', family: 'Inter' },
                            margin: { l: 40, r: 0, t: 10, b: 40 },
                            xaxis: { gridcolor: 'rgba(255,255,255,0.05)' },
                            yaxis: { gridcolor: 'rgba(255,255,255,0.05)' },
                          }}
                          style={{ width: '100%', height: '300px' }}
                          config={{ responsive: true, displayModeBar: false }}
                        />
                      </div>
                    ))}
                  </div>
                </div>
              )}
            </motion.div>
          ) : (
            <motion.div
              key="live"
              initial={{ opacity: 0, y: 10 }}
              animate={{ opacity: 1, y: 0 }}
              exit={{ opacity: 0, y: -10 }}
            >
              <div className="grid">
                <div className="glass-card" style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
                  <div>
                    <div style={{ display: 'flex', alignItems: 'center', gap: '8px', marginBottom: '4px' }}>
                      <div style={{ width: '8px', height: '8px', borderRadius: '50%', backgroundColor: 'var(--primary)', boxShadow: '0 0 8px var(--primary)' }}></div>
                      <h4 style={{ margin: 0 }}>System Operational</h4>
                    </div>
                    <p style={{ fontSize: '0.8rem', opacity: 0.5 }}>Breeze API Connected • 25ms Latency</p>
                  </div>
                  <div className="btn btn-primary">Start Live Engine</div>
                </div>

                <div className="grid grid-cols-3">
                  <div className="glass-card">
                    <div className="metric-label">Open Positions</div>
                    <div className="metric-value">3</div>
                  </div>
                  <div className="glass-card">
                    <div className="metric-label">Daily Realized PnL</div>
                    <div className="metric-value" style={{ color: 'var(--primary)' }}>+₹12,450.00</div>
                  </div>
                  <div className="glass-card">
                    <div className="metric-label">Unrealized PnL</div>
                    <div className="metric-value" style={{ color: 'var(--primary)' }}>+₹4,200.00</div>
                  </div>
                </div>

                <div className="glass-card">
                  <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: '24px' }}>
                    <div className="metric-label">Real-time Order Feed</div>
                    <div className="btn" style={{ fontSize: '0.7rem', padding: '4px 8px', background: 'rgba(255,255,255,0.05)' }}>Clear logs</div>
                  </div>
                  <div style={{ display: 'flex', flexDirection: 'column', gap: '12px', maxHeight: '300px', overflowY: 'auto', fontSize: '0.85rem' }}>
                    {[
                      { time: '14:24:01', type: 'BUY', symbol: 'NIFTY24OCT25000CE', price: '124.50', status: 'FILLED' },
                      { time: '14:23:45', type: 'SELL', symbol: 'RELIANCE', price: '2,845.00', status: 'FILLED' },
                      { time: '14:15:10', type: 'BUY', symbol: 'HDFCBANK', price: '1,650.00', status: 'FILLED' },
                    ].map((order, i) => (
                      <div key={i} style={{ display: 'flex', justifyContent: 'space-between', padding: '12px', borderRadius: '8px', background: 'rgba(255,255,255,0.02)', border: '1px solid var(--card-border)' }}>
                        <div style={{ display: 'flex', gap: '16px' }}>
                          <span style={{ opacity: 0.4 }}>{order.time}</span>
                          <span style={{ color: order.type === 'BUY' ? 'var(--primary)' : 'var(--danger)', fontWeight: '700' }}>{order.type}</span>
                          <span style={{ fontWeight: '600' }}>{order.symbol}</span>
                        </div>
                        <div style={{ display: 'flex', gap: '16px' }}>
                          <span>₹{order.price}</span>
                          <span style={{ color: 'var(--primary)', fontSize: '0.7rem' }}>● {order.status}</span>
                        </div>
                      </div>
                    ))}
                  </div>
                </div>
              </div>
            </motion.div>
          )}
        </AnimatePresence>
      </div>
    </main>
  );
}
