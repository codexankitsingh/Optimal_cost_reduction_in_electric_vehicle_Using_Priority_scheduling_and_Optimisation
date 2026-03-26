import asyncio
from playwright.async_api import async_playwright

async def main():
    async with async_playwright() as p:
        browser = await p.chromium.launch(headless=True)
        page = await browser.new_page(viewport={"width": 1440, "height": 1080})
        
        # 1. Capture Home Page
        print("Capturing Home Page...")
        await page.goto('http://127.0.0.1:4173/')
        await page.wait_for_timeout(3000)
        await page.screenshot(path='../assets/screenshot_home.png')
        print("Saved screenshot_home.png")
        
        # 2. Navigate to Simulation Studio
        print("Navigating to Studio...")
        await page.click('text="▶ Launch Simulation Studio"')
        await page.wait_for_timeout(2000)
        
        # Click Generate (it's the first btn-secondary inside the actions div)
        print("Clicking Generate...")
        await page.evaluate('''() => {
            const btns = Array.from(document.querySelectorAll('button'));
            const genBtn = btns.find(b => b.textContent && b.textContent.includes('Generate'));
            if(genBtn) genBtn.click();
        }''')
        await page.wait_for_timeout(2000)
        
        # Click Run Simulation
        print("Clicking Run Simulation...")
        await page.evaluate('''() => {
            const btns = Array.from(document.querySelectorAll('button'));
            const runBtn = btns.find(b => b.textContent && b.textContent.includes('Run Simulation'));
            if(runBtn) runBtn.click();
        }''')
        
        print("Waiting for charts to appear...")
        try:
            await page.wait_for_selector('text=Simulation complete', timeout=15000)
        except Exception as e:
            print("Warning: Did not find completion text, proceeding anyway...")
            await page.wait_for_timeout(5000)
            
        await page.evaluate("window.scrollTo(0, 400);")
        await page.wait_for_timeout(1000)
        
        await page.screenshot(path='../assets/screenshot_studio.png')
        print("Saved screenshot_studio.png")
        
        await browser.close()

if __name__ == '__main__':
    asyncio.run(main())
