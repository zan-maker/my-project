const sharp = require('sharp');
const React = require('react');
const ReactDOMServer = require('react-dom/server');
const { FaRobot, FaCogs, FaChartLine, FaUsers, FaRocket, FaBolt, FaBrain, FaDatabase } = require('react-icons/fa');

async function createGradient(filename, c1, c2, dir = 'x1="0%" y1="0%" x2="100%" y2="100%"') {
  const svg = `<svg xmlns="http://www.w3.org/2000/svg" width="1440" height="810">
    <defs><linearGradient id="g" ${dir}><stop offset="0%" style="stop-color:${c1}"/><stop offset="100%" style="stop-color:${c2}"/></linearGradient></defs>
    <rect width="100%" height="100%" fill="url(#g)"/>
  </svg>`;
  await sharp(Buffer.from(svg)).png().toFile(filename);
}

async function rasterizeIcon(IconComponent, color, size, filename) {
  const svgString = ReactDOMServer.renderToStaticMarkup(
    React.createElement(IconComponent, { color: `#${color}`, size: String(size) })
  );
  await sharp(Buffer.from(svgString)).png().toFile(filename);
}

async function main() {
  // Cover gradient: deep dark blue to slightly lighter
  await createGradient('cover-bg.png', '#050710', '#0D1B2A');
  // Accent gradient bar
  await createGradient('accent-bar.png', '#00F5A0', '#00C9FF', 'x1="0%" y1="0%" x2="100%" y2="0%"');
  // Dark slide bg
  await createGradient('dark-bg.png', '#050710', '#0A0F1E');
  // Closing bg
  await createGradient('closing-bg.png', '#0D1320', '#162033', 'x1="0%" y1="0%" x2="0%" y2="100%"');

  // Icons (white for dark backgrounds, accent green for light cards)
  const icons = [
    [FaRobot, '00F5A0', 'icon-robot.png'],
    [FaCogs, '00F5A0', 'icon-cogs.png'],
    [FaChartLine, '00F5A0', 'icon-chart.png'],
    [FaUsers, '00F5A0', 'icon-users.png'],
    [FaRocket, '00F5A0', 'icon-rocket.png'],
    [FaBolt, '00F5A0', 'icon-bolt.png'],
    [FaBrain, '00F5A0', 'icon-brain.png'],
    [FaDatabase, '00F5A0', 'icon-database.png'],
  ];
  for (const [Ic, col, fn] of icons) {
    await rasterizeIcon(Ic, col, 128, fn);
  }

  // Copy logo
  const logoSrc = '/home/z/my-project/flagos-track3/logo.png';
  const { data, info } = await sharp(logoSrc).resize(200, 200).png().toBuffer({ resolveWithObject: true });
  require('fs').writeFileSync('logo-200.png', data);
  
  console.log('All assets generated!');
}

main().catch(console.error);
