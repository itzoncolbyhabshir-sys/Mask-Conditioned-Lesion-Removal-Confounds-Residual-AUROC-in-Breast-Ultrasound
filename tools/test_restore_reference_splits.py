import unittest
from restore_reference_splits import align


class ReferenceSplitTests(unittest.TestCase):
    def setUp(self):
        self.reference = [dict(dataset_name="BUS-BRA", image_name=f"{f}_{label}.png", patient_id=f"p{f}_{label}", label=label, split_id=str(f)) for f in range(5) for label in ("benign", "malignant")]
        self.local = [dict(dataset_name=r["dataset_name"], image_path="D:\\new-root\\" + r["image_name"], patient_id=r["patient_id"], label=r["label"], split_id="4") for r in self.reference]

    def test_relocated_paths_preserved_and_folds_restored(self):
        updated, changed = align(self.local, self.reference)
        self.assertEqual(changed, 8)
        self.assertEqual([r["split_id"] for r in updated], [r["split_id"] for r in self.reference])
        self.assertEqual(updated[0]["image_path"], self.local[0]["image_path"])
        self.assertEqual(self.local[0]["split_id"], "4")

    def test_rejects_incomplete_cohort(self):
        with self.assertRaises(ValueError):
            align(self.local[:-1], self.reference)

    def test_rejects_duplicate_image(self):
        with self.assertRaises(ValueError):
            align(self.local + self.local[:1], self.reference)

    def test_rejects_label_change(self):
        self.local[0]["label"] = "malignant"
        with self.assertRaises(ValueError):
            align(self.local, self.reference)

    def test_rejects_patient_crossing_folds(self):
        self.local[2]["patient_id"] = self.local[0]["patient_id"]
        self.reference[2]["patient_id"] = self.reference[0]["patient_id"]
        with self.assertRaises(ValueError):
            align(self.local, self.reference)

    def test_allows_complete_single_dataset(self):
        extra = [dict(r, dataset_name="BUS-UCLM") for r in self.reference]
        updated, _ = align(self.local, self.reference + extra)
        self.assertEqual(len(updated), 10)


if __name__ == "__main__":
    unittest.main()
