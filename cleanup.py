import os
import shutil
import argparse
from datetime import datetime
from pathlib import Path

def get_size_format(b, factor=1024, suffix="B"):
    """Scale bytes to its proper byte format."""
    for unit in ["", "K", "M", "G", "T", "P", "E", "Z"]:
        if b < factor:
            return f"{b:.2f}{unit}{suffix}"
        b /= factor
    return f"{b:.2f}Y{suffix}"

def get_dir_size(path):
    """Returns the total size of a directory in bytes."""
    total = 0
    with os.scandir(path) as it:
        for entry in it:
            if entry.is_file():
                total += entry.stat().st_size
            elif entry.is_dir():
                total += get_dir_size(entry.path)
    return total

def get_file_or_dir_size(path_obj):
    if path_obj.is_file():
        return path_obj.stat().st_size
    elif path_obj.is_dir():
        return get_dir_size(path_obj)
    return 0

def cleanup(purge=False, archive=False, dry_run=True):
    root_dir = Path(".")
    
    # 1. Purge Targets: __pycache__, .DS_Store, wandb offline folders
    purge_patterns = ["__pycache__", ".DS_Store", "wandb"]
    to_purge = []
    
    # 2. Archive Targets: loose artifacts in the root directory
    archive_patterns = ["iter_*", "latents_*", "checkpoints", "*.csv", "*.png"]
    to_archive = []
    
    # Find purge targets recursively
    for pattern in purge_patterns:
        for p in root_dir.rglob(pattern):
            if p.exists():
                to_purge.append(p)
                
    # Find archive targets in ROOT only
    for pattern in archive_patterns:
        for p in root_dir.glob(pattern):
            if p.exists() and p.is_dir() and p.name.startswith("experiments"):
                continue # don't archive the experiments folder itself
            if p.exists():
                to_archive.append(p)
                
    # Deduplicate
    to_purge = list(set(to_purge))
    to_archive = list(set(to_archive))
    
    if dry_run:
        print("\n=== DRY RUN MODE (No changes will be made) ===")
        
        print("\n🗑️  TARGETS TO PURGE (Permanently Delete):")
        total_purge_size = 0
        for p in to_purge:
            size = get_file_or_dir_size(p)
            total_purge_size += size
            print(f"  - {p} ({get_size_format(size)})")
        if not to_purge: print("  None")
            
        print(f"\n📦 TARGETS TO ARCHIVE (Move to experiments/archive_.../):")
        total_archive_size = 0
        for p in to_archive:
            size = get_file_or_dir_size(p)
            total_archive_size += size
            print(f"  - {p} ({get_size_format(size)})")
        if not to_archive: print("  None")
            
        print(f"\n💡 Summary:")
        print(f"  - Disk space to be freed: {get_size_format(total_purge_size)}")
        print(f"  - Artifacts to be safely archived: {get_size_format(total_archive_size)}")
        print("\nTo execute these changes, run:")
        print("  python cleanup.py --purge --archive")
        return

    # Execute Purge
    if purge:
        print("\n🗑️  Executing Purge...")
        freed = 0
        for p in to_purge:
            if not p.exists(): continue
            size = get_file_or_dir_size(p)
            freed += size
            print(f"Deleting {p}...")
            if p.is_dir():
                shutil.rmtree(p)
            else:
                p.unlink()
        print(f"✅ Purged {get_size_format(freed)} of system junk.")
        
    # Execute Archive
    if archive:
        if to_archive:
            print("\n📦 Executing Archive...")
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            archive_dir = root_dir / "experiments" / f"archive_{timestamp}"
            archive_dir.mkdir(parents=True, exist_ok=True)
            
            archived_size = 0
            for p in to_archive:
                if not p.exists(): continue
                size = get_file_or_dir_size(p)
                archived_size += size
                print(f"Moving {p} to {archive_dir}/")
                shutil.move(str(p), str(archive_dir / p.name))
                
            print(f"✅ Safely archived {get_size_format(archived_size)} of artifacts to {archive_dir}/")
        else:
            print("\n📦 Nothing to archive.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Smart Garbage Collector for ML Repository")
    parser.add_argument("--purge", action="store_true", help="Permanently delete system junk (__pycache__, wandb logs)")
    parser.add_argument("--archive", action="store_true", help="Move loose root artifacts to experiments/archive/")
    
    args = parser.parse_args()
    
    # If neither flag is provided, default to dry_run
    dry_run = not (args.purge or args.archive)
    
    cleanup(purge=args.purge, archive=args.archive, dry_run=dry_run)
