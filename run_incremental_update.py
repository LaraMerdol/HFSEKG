#!/usr/bin/env python3
"""
Run incremental update of HuggingFace Knowledge Graph
Only fetches entities created or modified after Dec 15, 2024
Skips ~1.2M existing entities to save API quota
"""

from HuggingKG import KGConstructor

def main():
    print("\n" + "=" * 80)
    print("INCREMENTAL KNOWLEDGE GRAPH UPDATE")
    print("=" * 80)
    print("\nConfiguration:")
    print("  • Previous snapshot: HuggingKG_V20251215174821")
    print("  • Cutoff date: 2024-12-15T17:48:21 (will only process entities after this)")
    print("  • API rate limit: 1000 requests per 5 minutes")
    print("  • Batch checkpoint: Every 5000 entities")
    print("\n" + "-" * 80)
    
    # Initialize with previous data and cutoff date
    print("Initializing Knowledge Graph Constructor...")
    kg = KGConstructor(
        previous_data_dir="HuggingKG_V20251215174821",
        cutoff_date="2024-12-15T17:48:21"
    )
    
    print(f"✓ Initialized successfully")
    print(f"  Output directory: {kg.output_dir}")
    print(f"  Previous entities loaded:")
    print(f"    - Models: {len(kg.processed_models):,}")
    print(f"    - Datasets: {len(kg.processed_datasets):,}")
    print(f"    - Spaces: {len(kg.processed_spaces):,}")
    print(f"    - Tasks: {len(kg.processed_tasks):,}")
    
    print("\n" + "-" * 80)
    print("Starting incremental update run...\n")
    
    # Run the incremental update
    kg.run()
    
    print("\n" + "=" * 80)
    print("INCREMENTAL UPDATE COMPLETED")
    print("=" * 80)
    print(f"Output directory: {kg.output_dir}")
    print("\nFinal entity counts:")
    print(f"  • Models: {len(kg.processed_models):,}")
    print(f"  • Datasets: {len(kg.processed_datasets):,}")
    print(f"  • Spaces: {len(kg.processed_spaces):,}")
    print(f"  • Tasks: {len(kg.processed_tasks):,}")
    print(f"  • Papers: {len(kg.processed_papers):,}")
    print(f"  • Collections: {len(kg.processed_collections):,}")
    print(f"  • Users: {len(kg.processed_users):,}")
    print(f"  • Organizations: {len(kg.processed_orgs):,}")
    print("\nCheck {kg.output_dir}/logs.log for detailed run information.")
    print("=" * 80 + "\n")

if __name__ == "__main__":
    main()
