from pathlib import Path

from samisk_ocr.utils import page_image_stem_to_urn_page


def test_page_image_stem_to_urn_page():
    test_data = [
        {"file_name": "URN_NBN_no-nb_digibok_2018062248001_0026.jpg", "page_number": 26},
        {
            "file_name": "no-nb_digavis_avvir_null_null_20230714_16_133_1-1_004_hovedavis.jpg",
            "page_number": 4,
        },
        {"file_name": "URN_NBN_no-nb_digibok_2017121908058_0014.jpg", "page_number": 14},
        {
            "file_name": "no-nb_digavis_avvir_null_null_20230714_16_133_1-1_011_hovedavis.jpg",
            "page_number": 11,
        },
        {"file_name": "URN_NBN_no-nb_digibok_2016031548149_0136.jpg", "page_number": 136},
        {"file_name": "URN_NBN_no-nb_digibok_2012092808086_C1.jpg", "page_number": -1},
        {"file_name": "URN_NBN_no-nb_pliktmonografi_000024690_0006.jpg", "page_number": 6},
        {
            "file_name": "no-nb_digavis_avvir_null_null_20160719_9_133_1-1_008_null.jpg",
            "page_number": 8,
        },
        {
            "file_name": "no-nb_digavis_samiaigi_null_null_19851014_6_65_1-1_004_null.jpg",
            "page_number": 4,
        },
    ]

    for e in test_data:
        urn, page_number = page_image_stem_to_urn_page(Path(e["file_name"]).stem)
        assert page_number == e["page_number"]


if __name__ == "__main__":
    test_page_image_stem_to_urn_page()
