"""
Dataset Crawler and Image Pipeline for Kurta & Kurti images.
Uses BingImageCrawler to download high-resolution portrait photographs for dataset augmentation.
"""
import os
import sys
import time
import shutil
import logging
import argparse
from pathlib import Path
from PIL import Image
from tqdm import tqdm

import config

# Suppress noisy crawler logs
logging.getLogger("icrawler").setLevel(logging.CRITICAL)
logging.getLogger("urllib3").setLevel(logging.CRITICAL)

# Search queries optimized for portrait fashion photography
DEFAULT_QUERY_MAP = {
    "Men_Kurta": [
        "indian men kurta portrait",
        "mens kurta pajama fashion",
        "traditional mens kurta studio photo",
        "designer kurta for men photoshoot"
    ],
    "Women_Kurti": [
        "designer women kurti portrait",
        "traditional indian kurti photoshoot",
        "women kurti fashion studio",
        "ethnic wear kurti women portrait"
    ]
}


def filter_and_save_portrait_images(temp_dir: Path, final_dir: Path, category: str, min_height: int = 600) -> int:
    """
    Filters images in temp_dir to keep only portrait orientation (height > width and height >= min_height).
    Moves valid images to final_dir with unique timestamps and cleans up temp files.
    """
    saved = 0
    if not temp_dir.exists():
        return 0

    for file_path in temp_dir.iterdir():
        if not file_path.is_file():
            continue

        try:
            with Image.open(file_path) as img:
                w, h = img.size

                # Keep high-quality portrait images only
                if h > w and h >= min_height:
                    img_rgb = img.convert("RGB")
                    unique_name = f"bing_{category}_{int(time.time() * 1000)}_{saved}.jpg"
                    dest_file = final_dir / unique_name
                    img_rgb.save(dest_file, "JPEG", quality=95)
                    saved += 1
        except Exception:
            # Skip invalid, corrupt, or unreadable files
            pass
        finally:
            # Delete temporary download
            try:
                if file_path.exists():
                    file_path.unlink()
            except OSError:
                pass

    return saved


def crawl_kurta_dataset(
    output_dir: str = str(config.DEFAULT_DOWNLOAD_DIR),
    target_per_category: int = 500,
    min_height: int = 600,
    batch_crawl_size: int = 50,
):
    """Crawls and curates portrait images for each defined category."""
    try:
        from icrawler.builtin import BingImageCrawler
        import icrawler.parser
        
        # Safe patch for known icrawler Bing parser NoneType edge cases if encountered
        def safe_parse(self, response, **kwargs):
            try:
                return super(icrawler.parser.Parser, self).parse(response, **kwargs)
            except Exception:
                return []
                
    except ImportError:
        print("[ERROR] icrawler package is not installed. Install it with: pip install icrawler")
        return

    base_save_path = Path(output_dir)
    temp_dir = base_save_path / "temp"
    temp_dir.mkdir(parents=True, exist_ok=True)

    print(f"[INFO] Output Directory: {base_save_path}")
    print(f"[INFO] Target per Category: {target_per_category} images")

    for category, queries in DEFAULT_QUERY_MAP.items():
        cat_dir = base_save_path / category
        cat_dir.mkdir(parents=True, exist_ok=True)

        existing_images = len([f for f in cat_dir.glob("*.jpg")])
        pbar = tqdm(total=target_per_category, initial=existing_images, desc=f"Collecting {category}")
        current_total = existing_images
        query_idx = 0

        while current_total < target_per_category:
            query = queries[query_idx % len(queries)]

            crawler = BingImageCrawler(
                storage={"root_dir": str(temp_dir)},
                log_level=logging.CRITICAL
            )

            try:
                crawler.crawl(keyword=query, max_num=batch_crawl_size)
            except Exception as e:
                print(f"\n[WARN] Crawler exception on query '{query}': {e}")

            saved = filter_and_save_portrait_images(temp_dir, cat_dir, category, min_height=min_height)
            current_total += saved
            pbar.update(saved)
            query_idx += 1

            # Prevent endless loop if query exhausts results
            if saved == 0 and query_idx > len(queries) * 3:
                print(f"\n[WARN] Search queries exhausted for {category}. Moving forward.")
                break

        pbar.close()

    # Clean up temp folder
    if temp_dir.exists():
        shutil.rmtree(temp_dir, ignore_errors=True)

    print("\n[SUCCESS] Image crawl and curation pipeline completed!")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Download & Filter Kurta/Kurti Images via Bing Crawler")
    parser.add_argument("--output-dir", type=str, default=str(config.DEFAULT_DOWNLOAD_DIR), help="Root directory for downloaded images")
    parser.add_argument("--target", type=int, default=500, help="Target number of portrait images per category")
    parser.add_argument("--min-height", type=int, default=600, help="Minimum pixel height for portrait filter")
    parser.add_argument("--batch-size", type=int, default=50, help="Number of images per crawler query batch")

    args = parser.parse_args()
    crawl_kurta_dataset(
        output_dir=args.output_dir,
        target_per_category=args.target,
        min_height=args.min_height,
        batch_crawl_size=args.batch_size
    )