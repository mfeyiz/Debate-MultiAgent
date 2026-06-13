"""ModernBERT pipeline smoke tests."""

from pathlib import Path
import unittest

from app.services.bert_service import ModernBERTPipeline


class ModernBERTPipelineSmokeTest(unittest.TestCase):
    """Checks that the bundled model emits usable, non-empty spans."""

    def test_text_units_do_not_split_numeric_or_roman_citations(self) -> None:
        pipeline = object.__new__(ModernBERTPipeline)
        text = (
            "UNICEF verilerine göre her gün 15.000 çocuk önlenebilir nedenlerle ölüyor. "
            "Treatise III. II. I bölümünde ebeveyn sevgisi tartışılır. "
            "Sonuç cümlesi burada biter."
        )

        units = [text[start:end] for start, end in pipeline._text_units(text)]

        self.assertEqual(len(units), 3)
        self.assertIn("15.000 çocuk", units[0])
        self.assertIn("III. II. I", units[1])

    def test_clean_unit_text_removes_markdown_list_chrome(self) -> None:
        pipeline = object.__new__(ModernBERTPipeline)
        text = "1. **BM verileriyle çelişki**: Bölgesel nüfus düşüşü küresel üreme yükümlülüğü doğurmaz."

        start, cleaned = pipeline._clean_unit_text(text, 0, len(text))

        self.assertGreater(start, 0)
        self.assertEqual(cleaned, "Bölgesel nüfus düşüşü küresel üreme yükümlülüğü doğurmaz.")

    @unittest.skipUnless(
        Path("models/component_classifier/final").exists()
        and Path("models/relation_classifier/final").exists(),
        "Bundled ModernBERT weights are not available.",
    )
    def test_extract_components_returns_valid_non_empty_spans(self) -> None:
        text = "Sentetik veri model performansını güçlü biçimde artırır."
        pipeline = ModernBERTPipeline()

        components = pipeline.extract_components(text, default_type="claim")

        self.assertGreater(len(components), 0)
        for component in components:
            self.assertTrue(component.text.strip())
            self.assertGreaterEqual(component.start_idx, 0)
            self.assertGreater(component.end_idx, component.start_idx)
            self.assertLessEqual(component.end_idx, len(text))
            self.assertGreaterEqual(component.confidence, 0.0)
            self.assertLessEqual(component.confidence, 1.0)

    @unittest.skipUnless(
        Path("models/component_classifier/final").exists()
        and Path("models/relation_classifier/final").exists(),
        "Bundled ModernBERT weights are not available.",
    )
    def test_extract_components_returns_sentence_level_spans(self) -> None:
        text = (
            "Uzaktan çalışma verimliliği artırır. "
            "Stanford Üniversitesi'nden Nicholas Bloom'un 2015 yılında yayımlanan "
            "randomize kontrollü deneyi, evden çalışanların verimliliğinin %13 "
            "arttığını göstermiştir. Bu somut veriler, uzaktan çalışmanın "
            "verimliliği kanıtlanabilir şekilde yükselttiğini ortaya koymaktadır."
        )
        pipeline = ModernBERTPipeline()

        components = pipeline.extract_components(text, default_type="claim")

        self.assertGreater(len(components), 0)
        self.assertLessEqual(len(components), 5)
        for component in components:
            words = component.text.split()
            self.assertGreaterEqual(len(words), 4)
            self.assertGreaterEqual(len(component.text), 24)

        self.assertGreaterEqual(len(components), 3)
        self.assertEqual(components[0].text, "Uzaktan çalışma verimliliği artırır.")

    @unittest.skipUnless(
        Path("models/component_classifier/final").exists()
        and Path("models/relation_classifier/final").exists(),
        "Bundled ModernBERT weights are not available.",
    )
    def test_extract_components_splits_selected_conjunctions(self) -> None:
        text = (
            "Yapay zeka öğrencinin eksiklerini hızlı tespit eder, "
            "ancak yanlış veriyle hatalı yönlendirme yapabilir."
        )
        pipeline = ModernBERTPipeline()

        units = pipeline._text_units(text)

        self.assertEqual(len(units), 2)
        self.assertEqual(text[units[1][0] : units[1][0] + 5].lower(), "ancak")

    @unittest.skipUnless(
        Path("models/component_classifier/final").exists()
        and Path("models/relation_classifier/final").exists(),
        "Bundled ModernBERT weights are not available.",
    )
    def test_parenthood_claim_hard_cases_are_not_dropped_as_other(self) -> None:
        pipeline = ModernBERTPipeline()
        text = (
            "Agent Alpha'nın savındaki temel zayıflık, doğal olanın ahlaken doğru sayılması "
            "yanılgısına dayanmasıdır. Sonuç olarak bilinçli ebeveynlik argümanı, ahlaki "
            "bir seçimin değil bir ayrıcalığın savunusudur."
        )

        components = pipeline.extract_components(text, default_type="claim")

        self.assertGreaterEqual(len(components), 1)
        self.assertTrue(any(component.component_type == "claim" for component in components))


if __name__ == "__main__":
    unittest.main()
