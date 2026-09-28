const puppeteer = require('puppeteer');

(async () => {
  const browser = await puppeteer.launch({ headless: 'new' });
  const page = await browser.newPage();
  
  page.on('console', msg => console.log('PAGE LOG:', msg.text()));
  
  await page.setViewport({ width: 1920, height: 1080 });
  await page.goto('http://localhost:3000', { waitUntil: 'networkidle0' });
  
  await page.screenshot({ path: 'C:/Users/Aryan/.gemini/antigravity-ide/brain/1cd84182-01d3-4d15-860c-2213b030154f/scratch/frontend_screen.png' });
  await browser.close();
  console.log("Screenshot saved.");
})();
