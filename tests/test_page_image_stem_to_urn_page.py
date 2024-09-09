from pathlib import Path

import pytest

from samisk_ocr.utils import page_image_stem_to_urn_page


@pytest.mark.parametrize(
    "file_name, page_number",
    [
        ("URN_NBN_no-nb_digibok_2018062248001_0026.jpg", 26),
        ("no-nb_digavis_avvir_null_null_20230714_16_133_1-1_004_hovedavis.jpg", 4),
        ("URN_NBN_no-nb_digibok_2017121908058_0014.jpg", 14),
        ("no-nb_digavis_avvir_null_null_20230714_16_133_1-1_011_hovedavis.jpg", 11),
        ("URN_NBN_no-nb_digibok_2016031548149_0136.jpg", 136),
        ("URN_NBN_no-nb_digibok_2012092808086_C1.jpg", -1),
        ("URN_NBN_no-nb_pliktmonografi_000024690_0006.jpg", 6),
        ("no-nb_digavis_avvir_null_null_20160719_9_133_1-1_008_null.jpg", 8),
        ("no-nb_digavis_samiaigi_null_null_19851014_6_65_1-1_004_null.jpg", 4),
    ],
)
def test_page_image_stem_to_urn_page(file_name, page_number):
    _urn, inferred_page_number = page_image_stem_to_urn_page(Path(file_name).stem)
    assert inferred_page_number == page_number


if __name__ == "__main__":
    test_page_image_stem_to_urn_page()
