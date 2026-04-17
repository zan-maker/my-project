const pptxgen = require('pptxgenjs');
const html2pptx = require('/home/z/my-project/skills/ppt/scripts/html2pptx');
const path = require('path');

async function build() {
  const pptx = new pptxgen();
  pptx.layout = 'LAYOUT_16x9';
  pptx.author = 'CubicZan';
  pptx.title = 'CubicZan Agent Swarm - Growth Engine Application';
  pptx.subject = 'Supermoon 2026';

  const slides = [
    'slide01-cover.html',
    'slide02-problem.html',
    'slide03-solution.html',
    'slide04-architecture.html',
    'slide05-market.html',
    'slide06-competitive.html',
    'slide07-traction.html',
    'slide08-business.html',
    'slide09-roadmap.html',
    'slide10-closing.html',
  ];

  const allWarnings = [];
  for (const s of slides) {
    const fp = path.join(__dirname, s);
    console.log(`Processing ${s}...`);
    try {
      const { slide, warnings } = await html2pptx(fp, pptx);
      allWarnings.push({ file: s, warnings });
      if (warnings.length > 0) {
        console.log(`  Warnings (${warnings.length}):`);
        warnings.forEach(w => console.log(`    - ${w}`));
      }
    } catch (err) {
      console.error(`  ERROR: ${err.message}`);
    }
  }

  const outPath = '/home/z/my-project/upload/cubiczan_agent_swarm.pptx';
  await pptx.writeFile({ fileName: outPath });
  console.log(`\nPresentation saved to: ${outPath}`);
  console.log(`Total slides: ${slides.length}`);
}

build().catch(e => { console.error(e); process.exit(1); });
