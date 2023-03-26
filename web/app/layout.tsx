import './globals.css';
import { Inter } from '@next/font/google';

const font = Inter({ subsets: ['latin'] });
const fontClass = font.className;

export default function RootLayout({ children }) {
  return (
    <html lang="en">
      <head />
      <body className={`bg-gradient-to-br from-kashmir-500 to-kashmir-700 ${fontClass}`}>
        <div className="flex h-full flex-col">
          <Nav />
          {children}
        </div>
      </body>
    </html>
  );
}

function Nav() {
  return (
    <nav className="flex flex-row justify-center py-8 px-12">
      <h1 className="text-6xl font-semibold text-slate-200">Chess Vision</h1>
    </nav>
  );
}
