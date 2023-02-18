import './globals.css';
import { Inter } from '@next/font/google';

const font = Inter({ subsets: ['latin'] });

export default function RootLayout({ children }) {
  return (
    <html lang='en'>
      <head />
      <body className={`bg-gradient-to-br from-cyan-500 to-indigo-500 ${font.className}`}>
        <Nav />
        {children}
      </body>
    </html>
  )
}

function Nav() {
  return (
    <nav className='py-8 px-12 flex flex-row'>
      <h1 className='text-6xl font-semibold'>
        Chess Vision
      </h1>
    </nav>
  );
}
