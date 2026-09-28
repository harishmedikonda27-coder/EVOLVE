import Link from 'next/link';

export default function EmptyStatePage({ title }: { title: string }) {
  return (
    <main className="min-h-screen flex flex-col items-center justify-center bg-[#020812] text-slate-100 font-mono">
      <div className="text-center">
        <h1 className="text-2xl mb-4 tracking-widest text-slate-300 uppercase">{title}</h1>
        <p className="text-slate-500 mb-8 tracking-wider text-sm uppercase">Data not available from backend</p>
        <Link href="/" className="text-blue-400 hover:text-blue-300 border border-blue-900/50 hover:border-blue-500/50 px-6 py-2 rounded-full transition-colors">
          Return to Evolve
        </Link>
      </div>
    </main>
  );
}
