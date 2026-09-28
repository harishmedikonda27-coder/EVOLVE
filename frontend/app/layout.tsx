import React from 'react';
import './globals.css';

export const metadata = {
  title: 'EVOLVE — Research Console',
  description: 'Self-Adaptive Neural Architecture for Continual Generalization',
};

export default function RootLayout({
  children,
}: {
  children: React.ReactNode;
}) {
  return (
    <html lang="en" className="dark">
      <body className="bg-background text-slate-100 min-h-screen font-sans antialiased">
        {children}
      </body>
    </html>
  );
}
