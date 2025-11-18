#!/usr/bin/env python3
"""
Quantum Torrent Manager for H'uhul Multi Hive OS
Distributed data sharding and verification for multi-hive synchronization
"""

import hashlib
import os
import json
from typing import List, Dict, Optional
from pathlib import Path


class QuantumTorrentManager:
    """
    Manages distributed data sharding for multi-hive training and synchronization
    Uses quantum-resistant hashing for data integrity verification
    """

    def __init__(self, data_dir: str = "./huhul_data/shards"):
        self.data_dir = Path(data_dir)
        self.data_dir.mkdir(parents=True, exist_ok=True)
        self.manifest_file = self.data_dir / "shard_manifest.json"
        self.manifest = self._load_manifest()

    def _load_manifest(self) -> Dict:
        """Load or create shard manifest"""
        if self.manifest_file.exists():
            with open(self.manifest_file, 'r') as f:
                return json.load(f)
        return {
            "version": "1.0.0",
            "shards": {},
            "total_records": 0
        }

    def _save_manifest(self):
        """Save shard manifest"""
        with open(self.manifest_file, 'w') as f:
            json.dump(self.manifest, f, indent=2)

    def create_data_shard(
        self,
        data: List[Dict],
        shard_size: int = 1000,
        category: str = "general"
    ) -> List[str]:
        """
        Split data into shards for distributed training/processing

        Args:
            data: List of data records
            shard_size: Number of records per shard
            category: Category/namespace for the shards

        Returns:
            List of shard file paths
        """
        shard_files = []
        category_dir = self.data_dir / category
        category_dir.mkdir(exist_ok=True)

        for i in range(0, len(data), shard_size):
            shard = data[i:i + shard_size]
            shard_hash = hashlib.sha256(
                json.dumps(shard, sort_keys=True).encode()
            ).hexdigest()[:16]

            shard_id = f"shard_{category}_{shard_hash}"
            shard_file = category_dir / f"{shard_id}.json"

            # Calculate integrity hash
            integrity_hash = self.generate_torrent_hash(json.dumps(shard))

            # Save shard
            with open(shard_file, 'w') as f:
                json.dump({
                    "shard_id": shard_id,
                    "category": category,
                    "records": len(shard),
                    "integrity_hash": integrity_hash,
                    "data": shard
                }, f, indent=2)

            shard_files.append(str(shard_file))

            # Update manifest
            self.manifest["shards"][shard_id] = {
                "file": str(shard_file),
                "category": category,
                "records": len(shard),
                "integrity_hash": integrity_hash
            }

        self.manifest["total_records"] += len(data)
        self._save_manifest()

        return shard_files

    def generate_torrent_hash(self, data: str) -> str:
        """
        Generate quantum-resistant hash for data verification
        Uses SHA3-512 (Keccak) which has better quantum resistance
        """
        h = hashlib.sha3_512()
        h.update(data.encode())
        return h.hexdigest()

    def validate_data_integrity(self, shard_files: Optional[List[str]] = None) -> Dict:
        """
        Validate all shards for integrity

        Args:
            shard_files: Specific shard files to validate, or None for all

        Returns:
            Validation results with status and details
        """
        if shard_files is None:
            shard_files = [
                v["file"] for v in self.manifest["shards"].values()
            ]

        results = {
            "total_shards": len(shard_files),
            "validated": 0,
            "corrupted": 0,
            "missing": 0,
            "details": []
        }

        for shard_file in shard_files:
            if not os.path.exists(shard_file):
                results["missing"] += 1
                results["details"].append({
                    "file": shard_file,
                    "status": "missing"
                })
                continue

            try:
                with open(shard_file, 'r') as f:
                    shard_data = json.load(f)

                # Recalculate hash
                data_str = json.dumps(shard_data.get("data", []))
                current_hash = self.generate_torrent_hash(data_str)
                stored_hash = shard_data.get("integrity_hash", "")

                if current_hash == stored_hash:
                    results["validated"] += 1
                    results["details"].append({
                        "file": shard_file,
                        "status": "valid",
                        "shard_id": shard_data.get("shard_id")
                    })
                else:
                    results["corrupted"] += 1
                    results["details"].append({
                        "file": shard_file,
                        "status": "corrupted",
                        "expected": stored_hash,
                        "actual": current_hash
                    })

            except Exception as e:
                results["corrupted"] += 1
                results["details"].append({
                    "file": shard_file,
                    "status": "error",
                    "error": str(e)
                })

        results["integrity_valid"] = (
            results["corrupted"] == 0 and results["missing"] == 0
        )

        return results

    def merge_shards(self, category: str) -> List[Dict]:
        """
        Merge all shards from a category back into a single dataset

        Args:
            category: Category to merge

        Returns:
            Merged data records
        """
        merged_data = []

        for shard_id, shard_info in self.manifest["shards"].items():
            if shard_info["category"] == category:
                with open(shard_info["file"], 'r') as f:
                    shard_data = json.load(f)
                    merged_data.extend(shard_data.get("data", []))

        return merged_data

    def get_shard_statistics(self) -> Dict:
        """Get statistics about all shards"""
        categories = {}

        for shard_id, shard_info in self.manifest["shards"].items():
            cat = shard_info["category"]
            if cat not in categories:
                categories[cat] = {
                    "shard_count": 0,
                    "total_records": 0
                }
            categories[cat]["shard_count"] += 1
            categories[cat]["total_records"] += shard_info["records"]

        return {
            "total_shards": len(self.manifest["shards"]),
            "total_records": self.manifest["total_records"],
            "categories": categories
        }

    def sync_with_remote_hive(self, remote_url: str) -> Dict:
        """
        Synchronize shards with remote hive node
        (Placeholder for distributed hive synchronization)
        """
        # This would implement torrent-like P2P synchronization
        # between multiple H'uhul hive nodes
        return {
            "status": "not_implemented",
            "message": "Remote hive sync requires network implementation"
        }


# ========== Example Usage ==========

if __name__ == "__main__":
    manager = QuantumTorrentManager()

    # Example: Prepare ASX data for distributed training
    asx_training_data = [
        {
            "text": "MX2LM: Process user query about system status",
            "type": "system_query",
            "agent": "queen"
        },
        {
            "text": "QWEN: Generate response using neural patterns",
            "type": "response_generation",
            "agent": "coder"
        },
        {
            "text": "ASX: Execute command via PRIME OS runtime",
            "type": "command_execution",
            "agent": "analyst"
        },
        {
            "text": "H'uhul Hive: Coordinate multi-agent task distribution",
            "type": "orchestration",
            "agent": "queen"
        },
        {
            "text": "SCXQ2: Compress knowledge base for efficient storage",
            "type": "compression",
            "agent": "memory"
        }
    ]

    # Create shards
    print("🔧 Creating data shards...")
    shards = manager.create_data_shard(
        asx_training_data,
        shard_size=2,
        category="training"
    )
    print(f"✅ Created {len(shards)} data shards")

    # Validate integrity
    print("\n🔍 Validating data integrity...")
    validation = manager.validate_data_integrity()
    print(f"✅ Validation: {validation['validated']}/{validation['total_shards']} valid")
    print(f"   Integrity: {'PASSED' if validation['integrity_valid'] else 'FAILED'}")

    # Get statistics
    print("\n📊 Shard Statistics:")
    stats = manager.get_shard_statistics()
    print(json.dumps(stats, indent=2))

    # Test merge
    print("\n🔄 Testing shard merge...")
    merged = manager.merge_shards("training")
    print(f"✅ Merged {len(merged)} records from training category")
