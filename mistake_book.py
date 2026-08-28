import hashlib
from typing import List, Dict

class MistakeBook:
    def __init__(self):
        # Maps problem_id -> list of records
        self.records: Dict[str, List[dict]] = {}
        
    def _hash_test(self, test_code: str) -> str:
        return hashlib.md5(test_code.strip().encode()).hexdigest()

    def add_failure(self, problem_id: str, test_code: str, step: int):
        if problem_id not in self.records:
            self.records[problem_id] = []
            
        test_hash = self._hash_test(test_code)
        
        # Check if already exists
        for record in self.records[problem_id]:
            if record["test_hash"] == test_hash:
                record["frequency"] += 1
                record["last_seen_step"] = step
                record["resolved"] = False
                return record
                
        # New failure
        new_record = {
            "test_hash": test_hash,
            "test_code": test_code.strip(),
            "frequency": 1,
            "first_seen_step": step,
            "last_seen_step": step,
            "resolved": False
        }
        self.records[problem_id].append(new_record)
        return new_record
        
    def mark_resolved(self, problem_id: str, test_code: str):
        """Mark a specific test as resolved if the Coder passes it."""
        if problem_id not in self.records:
            return
        test_hash = self._hash_test(test_code)
        for record in self.records[problem_id]:
            if record["test_hash"] == test_hash:
                record["resolved"] = True
                
    def get_all_tests(self, problem_id: str) -> List[str]:
        """Return all stored tests for a problem to evaluate the Coder."""
        if problem_id not in self.records:
            return []
        return [rec["test_code"] for rec in self.records[problem_id]]
                
    def retrieve_top_k(self, problem_id: str, k: int = 5) -> str:
        """Retrieve top k relevant tests for the Tester prompt."""
        if problem_id not in self.records:
            return "No previous mistakes recorded for this problem."
            
        # Priority: unresolved first, then highest frequency, then most recent
        sorted_records = sorted(
            self.records[problem_id],
            key=lambda x: (not x["resolved"], x["frequency"], x["last_seen_step"]),
            reverse=True
        )
        
        top_k = sorted_records[:k]
        if not top_k:
            return "No previous mistakes recorded for this problem."
            
        formatted_mistakes = "\n".join(
            f"- Test: `{rec['test_code']}` (Failed {rec['frequency']} times, Resolved: {rec['resolved']})"
            for rec in top_k
        )
        return formatted_mistakes
