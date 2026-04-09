import type { Metadata } from "next";
import "./globals.css";

export const metadata: Metadata = {
  title: "AI Trading Dashboard | Premium Alpha",
  description: "High-performance multi-stock algorithm trading and backtesting dashboard.",
};

export default function RootLayout({
  children,
}: Readonly<{
  children: React.ReactNode;
}>) {
  return (
    <html lang="en">
      <head>
        <link rel="icon" href="https://upload.wikimedia.org/wikipedia/commons/thumb/a/a7/React-icon.svg/512px-React-icon.svg.png" />
        {/* Preload heavy stlite assets early from CDN */}
        <link rel="preload" href="https://cdn.jsdelivr.net/npm/@stlite/mountable@0.75.0/build/stlite.js" as="script" />
        <link rel="preload" href="https://cdn.jsdelivr.net/npm/@stlite/mountable@0.75.0/build/stlite.css" as="style" />
        <link rel="preload" href="/streamlit_app.py" as="fetch" crossOrigin="anonymous" />
      </head>
      <body>{children}</body>
    </html>
  );
}
