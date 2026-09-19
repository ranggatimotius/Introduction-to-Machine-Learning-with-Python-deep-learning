import json
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
EXPECTED_NOTEBOOKS = [
    ROOT / "Chapter_01_Introduction" / "notebook.ipynb",
    ROOT / "Chapter_02_Supervised_Learning" / "notebook.ipynb",
    ROOT / "Chapter_03_Unsupervised_Learning_and_Preprocessing" / "notebook.ipynb",
    ROOT / "Chapter_04_Representing_Data_and_Engineering_Features" / "notebook.ipynb",
    ROOT / "Chapter_05_Model_Evaluation_and_Improvement" / "notebook.ipynb",
    ROOT / "Chapter_06_Algorithm_Chains_and_Pipelines" / "notebook.ipynb",
    ROOT / "Chapter_07_Working_with_Text_Data" / "notebook.ipynb",
    ROOT / "Chapter_08_Wrapping_Up" / "notebook.ipynb",
]


class NotebookQualityTests(unittest.TestCase):
    def test_all_chapter_notebooks_exist(self):
        missing = [str(path.relative_to(ROOT)) for path in EXPECTED_NOTEBOOKS if not path.exists()]
        self.assertEqual(missing, [], f"Missing notebooks: {missing}")

    def test_notebooks_have_theory_code_takeaways_and_clean_outputs(self):
        for path in EXPECTED_NOTEBOOKS:
            with self.subTest(notebook=path.parent.name):
                notebook = json.loads(path.read_text(encoding="utf-8"))
                cells = notebook.get("cells", [])
                markdown = "\n".join(
                    "".join(cell.get("source", []))
                    for cell in cells
                    if cell.get("cell_type") == "markdown"
                )
                code_cells = [cell for cell in cells if cell.get("cell_type") == "code"]

                self.assertIn("## Chapter Summary", markdown)
                self.assertIn("## Theoretical Deep-Dive", markdown)
                self.assertIn("## Key Takeaways", markdown)
                self.assertGreaterEqual(len(code_cells), 3)
                self.assertTrue(all(cell.get("execution_count") is not None for cell in code_cells))

                errors = [
                    output
                    for cell in code_cells
                    for output in cell.get("outputs", [])
                    if output.get("output_type") == "error"
                ]
                self.assertEqual(errors, [], f"Execution errors in {path}")

                stderr_outputs = [
                    output
                    for cell in code_cells
                    for output in cell.get("outputs", [])
                    if output.get("name") == "stderr"
                ]
                self.assertEqual(stderr_outputs, [], f"Warnings or stderr output in {path}")


if __name__ == "__main__":
    unittest.main()
