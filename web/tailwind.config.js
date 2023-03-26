const colors = require('tailwindcss/colors');

/** @type {import('tailwindcss').Config} */
module.exports = {
  content: [
    "./app/**/*.{js,ts,jsx,tsx}",
    "./pages/**/*.{js,ts,jsx,tsx}",
    "./components/**/*.{js,ts,jsx,tsx}",
    // Or if using `src` directory:
    "./src/**/*.{js,ts,jsx,tsx}",
  ],
  theme: {
    colors: {
      green: colors.green,
      red: colors.red,
      kashmir: {
        '50': '#f4f6fb',
        '100': '#e9ecf5',
        '200': '#ced7e9',
        '300': '#a2b4d7',
        '400': '#708cc0',
        '500': '#49679d',
        '600': '#3c568d',
        '700': '#314573',
        '800': '#2c3d60',
        '900': '#293551',
      },
    }
  },
  plugins: [],
}
