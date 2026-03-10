import asyncio
from playwright.async_api import async_playwright

async def main():
    async with async_playwright() as p:
        browser = await p.chromium.launch(headless=True)
        
        print("Capturing Simulation Studio populated...")
        page2 = await browser.new_page(viewport={"width": 1440, "height": 1080})
        await page2.goto('http://localhost:5173/studio')
        await page2.wait_for_timeout(2000)
        
        # Click Generate (it's the first btn-secondary inside the actions div)
        print("Clicking Generate...")
        await page2.evaluate('''() => {
            const btns = Array.from(document.querySelectorAll('button'));
            const genBtn = btns.find(b => b.textContent && b.textContent.includes('Generate'));
            if(genBtn) genBtn.click();
        }''')
        await page2.wait_for_timeout(2000)
        
        # Click Run Simulation
        print("Clicking Run Simulation...")
        await page2.evaluate('''() => {
            const btns = Array.from(document.querySelectorAll('button'));
            const runBtn = btns.find(b => b.textContent && b.textContent.includes('Run Simulation'));
            if(runBtn) runBtn.click();
        }''')
        
        print("Waiting for charts to appear...")
        await page2.wait_for_timeout(10000) # wait 10s for simulation
        
        await page2.evaluate("window.scrollTo(0, 400);")
        await page2.wait_for_timeout(1000)
        
        await page2.screenshot(path='../screenshot_studio.png')
        print("Saved screenshot_studio.png")
        
        await browser.close()

if __name__ == '__main__':
    asyncio.run(main())
