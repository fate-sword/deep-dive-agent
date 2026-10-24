import os
import sys
import time
from playwright.sync_api import sync_playwright

def run_browser_verification():
    errors = []
    with sync_playwright() as p:
        browser = p.chromium.launch(headless=True)
        context = browser.new_context(viewport={"width": 1280, "height": 1000})
        page = context.new_page()

        page.on("console", lambda msg: errors.append(f"Console {msg.type}: {msg.text}") if msg.type == "error" else None)
        page.on("pageerror", lambda exc: errors.append(f"Page error: {exc}"))

        print("Navigating to http://localhost:7860...")
        page.goto("http://localhost:7860")
        page.wait_for_selector("#search-card")

        # Click the sample run button
        print("Clicking 'See a sample run'...")
        sample_btn = page.locator(".sample-cta")
        sample_btn.click()

        # Wait for Findings Board to become visible
        print("Waiting for findings-board-section to appear...")
        page.wait_for_selector("#findings-board-section.is-visible", timeout=25000)

        # Give 500ms for smooth transition
        page.wait_for_timeout(600)

        # Check mono label
        mono_label = page.locator(".board-mono-label").inner_text()
        print(f"Mono label: {mono_label}")
        assert "KEY FINDINGS · PINNED FROM VERIFIED CLAIMS" in mono_label, f"Unexpected mono label: {mono_label}"

        # Check accent word in Caveat
        accent_word = page.locator(".board-accent-word").inner_text()
        print(f"Accent word: {accent_word}")
        assert "curated" in accent_word

        # Check headline finding and marker highlight
        headline_el = page.locator("#board-headline")
        assert headline_el.is_visible()
        highlight_el = page.locator(".marker-highlight")
        assert highlight_el.is_visible()
        print(f"Headline highlight: '{highlight_el.inner_text()}'")

        # Check corner tag
        corner_tag = page.locator("#board-corner-tag")
        assert corner_tag.is_visible()
        tag_text = corner_tag.inner_text()
        print(f"Corner tag label: '{tag_text}'")
        assert "1 rejected, 1 unverified" in tag_text

        # Click corner tag to open popover
        print("Clicking corner tag to inspect quarantined claims...")
        corner_tag.click()
        page.wait_for_selector("#corner-tag-popover.open", timeout=3000)

        popover_text = page.locator("#quarantine-list").inner_text()
        print(f"Quarantined text snippet: {popover_text[:120]}...")
        assert "rejected" in popover_text.lower()
        assert "unverified" in popover_text.lower()

        # Verify struck-through <del> claims
        del_count = page.locator("#quarantine-list del").count()
        assert del_count == 2, f"Expected 2 quarantined claims, found {del_count}"

        # Check lanes
        lanes = page.locator(".board-lane")
        lane_count = lanes.count()
        print(f"Sub-question lanes count: {lane_count}")
        assert lane_count >= 2

        # Check lane titles in Caveat
        for i in range(lane_count):
            lane_title = lanes.nth(i).locator(".lane-title").inner_text()
            print(f"  Lane {i+1}: {lane_title}")

        # Check notes
        notes = page.locator(".note-card")
        note_count = notes.count()
        print(f"Total verified notes on board: {note_count}")
        assert note_count == 4, f"Expected 4 verified notes, found {note_count}"

        # Verify first note has pin dot, CID, checkmark 'Verified', verbatim text, and domain chip
        first_note = notes.first
        assert first_note.locator(".note-pin").is_visible()
        assert first_note.locator(".note-status-verified").inner_text().strip() == "Verified"
        first_text = first_note.locator(".note-claim-text").inner_text()
        print(f"  First note text: '{first_text}'")
        assert "Analysts expect EV battery demand to keep growing steadily through 2030." in first_text

        # Check domain chip
        assert first_note.locator(".domain-chip").count() >= 1

        # Check footer: grounded stamp and prose toggle
        stamp = page.locator(".board-stamp")
        assert stamp.is_visible()
        stamp_text = stamp.inner_text()
        print(f"Grounded stamp text: {stamp_text.split()}")
        assert "GROUNDED" in stamp_text
        assert "83%" in stamp_text

        prose_summary = page.locator("#board-prose-toggle")
        assert prose_summary.is_visible()

        # Check report body does not contain Executive Summary heading
        report_body = page.locator("#report-body").inner_text()
        assert "Executive Summary" not in report_body, "Executive Summary heading should be removed from report body!"

        # Take screenshot of the Findings Board
        screenshot_path = os.path.abspath("findings_board_browser.png")
        page.locator("#findings-board-section").screenshot(path=screenshot_path)
        print(f"Screenshot saved to: {screenshot_path}")

        # Verify console errors
        print(f"Console errors recorded: {len(errors)}")
        if errors:
            print("Errors:", errors)
        assert len(errors) == 0, f"Encountered console errors: {errors}"

        browser.close()
        print("ALL BROWSER VERIFICATIONS PASSED SUCCESSFULLY!")

if __name__ == "__main__":
    run_browser_verification()
