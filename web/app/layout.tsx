import 'app/globals.css';
import { Inter } from 'next/font/google';

const font = Inter({ subsets: ['latin'] });
const fontClass = font.className;

export default function RootLayout({ children }) {
  return (
    <html lang="en">
      <head />
      <body className={`bg-gradient-to-br from-kashmir-600 to-kashmir-700 ${fontClass}`}>
        <div className="flex flex-col">
          <Banner />
          {children}
          <div className="h-16"></div>
        </div>
      </body>
    </html>
  );
}

function Banner() {
  return (
    <nav className="flex flex-row justify-center py-8 px-12">
      <h1 className="text-slate-200 text-6xl font-semibold">Chess Vision</h1>
    </nav>
  );
}
