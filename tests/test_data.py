import pytest
from src.data.generate_data_embeddings import create_embeddings
from src.data.normalize_data_entries import process_EQEQ_entries, process_gain_energy_entries

def test_example():
    assert 1 == 1


def test_create_embeddings_single_valid():

    input_string = "2D_AXIAL ENERGY 10.0 BOUNDARY N 1.00 K 0.10 LAYER THICKNESS 200.00 N 3.53 GAIN 1200.00 BOUNDARY N 1.00 K 0.10"

    expected_embeddings = [[1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1.0],
                           [0, 0, 1, 0, 0, 1, 0, 0, 0, 0, 0.1],
                           [0, 0, 0, 1, 0, 1, 0, 0, 0, 0, 0.1],
                           [0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0.2],
                           [0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0.353],
                           [0, 0, 0, 0, 1, 0, 0, 0, 0, 0, -0.0011839618747617523],
                           [0, 0, 1, 0, 0, 1, 0, 0, 0, 0, 0.1],
                           [0, 0, 0, 1, 0, 1, 0, 0, 0, 0, 0.1]]

    result = create_embeddings(input_string)

    assert result == expected_embeddings, f"Expected embeddings {expected_embeddings}, got {str(result)}"


def test_create_embeddings_single_invalid_string():
    input_string = "ENERGY ENERGY BOUNDARY N 1.00 K 0.10 LAYER THICKNESS 200.00 N 3.53 GAIN 1200.00 BOUNDARY N 1.00 K 0.10"

    expected_embeddings = []

    result = create_embeddings(input_string)

    assert result == expected_embeddings, f"Expected embeddings {expected_embeddings}, got {str(result)}"


def test_create_embeddings_single_invalid_range():
    input_string = "ENERGY 1.00 BOUNDARY N 1.00 K 0.10 LAYER THICKNESS 2000000.00 N 3.53 GAIN 1200.00 BOUNDARY N 1.00 K 0.10"

    expected_embeddings = []

    result = create_embeddings(input_string)

    assert result == expected_embeddings, f"Expected embeddings {expected_embeddings}, got {str(result)}"


def test_create_embeddings_single_invalid_Energy():
    input_string = "ENERGA 2.0 BOUNDARY N 1.00 K 0.10 LAYER THICKNESS 200.00 N 3.53 GAIN 1200.00 BOUNDARY N 1.00 K 0.10"

    expected_embeddings = []

    result = create_embeddings(input_string)

    assert result == expected_embeddings, f"Expected embeddings {expected_embeddings}, got {str(result)}"


def test_create_embeddings_single_empty_string():
    input_string = ""

    expected_embeddings = []

    result = create_embeddings(input_string)

    assert result == expected_embeddings, f"Expected embeddings {expected_embeddings}, got {str(result)}"


def test_create_embeddings_DBR_valid():
    input_string = "2D_AXIAL | ENERGY 1.0 | BOUNDARY N 1 K 0.00 | PAIRS_COUNT 50 { LAYER THICKNESS 5.00 N 3.41 K 0.00 ; LAYER THICKNESS 206.76 N 2.92 K 0.00 } | BOUNDARY N 1.00 K 0.00"

    expected_embeddings = [[1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0.1],
                           [0, 0, 1, 0, 0, 1, 0, 0, 0, 0, 0.1],
                           [0, 0, 0, 1, 0, 1, 0, 0, 0, 0, 0.0],
                           [0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0.5],
                           [0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0.005],
                           [0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 0.341],
                           [0, 0, 0, 1, 0, 0, 1, 0, 0, 0, 0.0],
                           [0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0.20676],
                           [0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 0.292],
                           [0, 0, 0, 1, 0, 0, 1, 0, 0, 0, 0.0],
                           [0, 0, 1, 0, 0, 1, 0, 0, 0, 0, 0.1],
                           [0, 0, 0, 1, 0, 1, 0, 0, 0, 0, 0.0]]

    result = create_embeddings(input_string)

    assert result == expected_embeddings, f"Expected embeddings {expected_embeddings}, got {str(result)}"


def test_create_embeddings_ALL_valid():

    input_string = "2D_AXIAL | ENERGY 1.0 | BOUNDARY N 1 K 0.00 | PAIRS_COUNT 20 { LAYER THICKNESS 5.00 N 3.41 K 0.00 ; LAYER THICKNESS 206.76 N 2.92 K 0.00 } | LAYER THICKNESS 136.49 N 3.00 K 0.00  | DUAL_LAYER { RADIUS 400 THICKNESS 5 N 3.53 GAIN 1200.00 THICKNESS 5 N 3.53 K 0.001 } | LAYER THICKNESS 31.85 N 3.08 K 0.00 | PAIRS_COUNT 20 { LAYER THICKNESS 5.00 N 3.41 K 0.00 ; LAYER THICKNESS 206.76 N 2.92 K 0.00 } | BOUNDARY N 1.00 K 0.00"

    expected_embeddings = [[1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0.1],
                           [0, 0, 1, 0, 0, 1, 0, 0, 0, 0, 0.1],
                           [0, 0, 0, 1, 0, 1, 0, 0, 0, 0, 0.0],
                           [0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0.2],
                           [0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0.005],
                           [0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 0.341],
                           [0, 0, 0, 1, 0, 0, 1, 0, 0, 0, 0.0],
                           [0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0.20676],
                           [0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 0.292],
                           [0, 0, 0, 1, 0, 0, 1, 0, 0, 0, 0.0],
                           [0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0.13649],
                           [0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0.3],
                           [0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0.0],
                           [0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0.4],
                           [0, 1, 0, 0, 0, 0, 0, 0, 1, 0, 0.005],
                           [0, 0, 1, 0, 0, 0, 0, 0, 1, 0, 0.353],
                           [0, 0, 0, 0, 1, 0, 0, 0, 1, 0, -0.011839618747617525],
                           [0, 1, 0, 0, 0, 0, 0, 0, 1, 0, 0.005],
                           [0, 0, 1, 0, 0, 0, 0, 0, 1, 0, 0.353],
                           [0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0.001],
                           [0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0.03185],
                           [0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0.308],
                           [0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0.0],
                           [0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0.2],
                           [0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0.005],
                           [0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 0.341],
                           [0, 0, 0, 1, 0, 0, 1, 0, 0, 0, 0.0],
                           [0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0.20676],
                           [0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 0.292],
                           [0, 0, 0, 1, 0, 0, 1, 0, 0, 0, 0.0],
                           [0, 0, 1, 0, 0, 1, 0, 0, 0, 0, 0.1],
                           [0, 0, 0, 1, 0, 1, 0, 0, 0, 0, 0.0]]

    result = create_embeddings(input_string)

    assert result == expected_embeddings, f"Expected embeddings {expected_embeddings}, got {str(result)}"


def test_create_embeddings_VCSEL_valid():

    input_string = "2D_AXIAL | ENERGY 1.264693197 | BOUNDARY N 3.53 K 0.00 | PAIRS_COUNT 35 { LAYER THICKNESS 82.88 N 2.96 K 0.00 ; LAYER THICKNESS 70.38 N 3.48 K 0.00 } | LAYER THICKNESS 82.88 N 2.96 K 0.00 | LAYER THICKNESS 135.76 N 3.48 K 0.00 | LAYER THICKNESS 10.00 N 3.48 GAIN 1200.00 | LAYER THICKNESS 135.76 N 3.48 K 0.00 | PAIRS_COUNT 30 { LAYER THICKNESS 82.88 N 2.96 K 0.00 ; LAYER THICKNESS 70.38 N 3.48 K 0.00 } | BOUNDARY N 1.00 K 0.00"

    expected_embeddings = [[1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0.1264693197],
                           [0, 0, 1, 0, 0, 1, 0, 0, 0, 0, 0.353],
                           [0, 0, 0, 1, 0, 1, 0, 0, 0, 0, 0.0],
                           [0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0.35],
                           [0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0.08288],
                           [0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 0.296],
                           [0, 0, 0, 1, 0, 0, 1, 0, 0, 0, 0.0],
                           [0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0.07038],
                           [0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 0.348],
                           [0, 0, 0, 1, 0, 0, 1, 0, 0, 0, 0.0],
                           [0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0.08288],
                           [0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0.296],
                           [0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0.0],
                           [0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0.13576],
                           [0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0.348],
                           [0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0.0],
                           [0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0.01],
                           [0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0.348],
                           [0, 0, 0, 0, 1, 0, 0, 0, 0, 0, -0.009361652909735329],
                           [0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0.13576],
                           [0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0.348],
                           [0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0.0],
                           [0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0.3],
                           [0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0.08288],
                           [0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 0.296],
                           [0, 0, 0, 1, 0, 0, 1, 0, 0, 0, 0.0],
                           [0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0.07038],
                           [0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 0.348],
                           [0, 0, 0, 1, 0, 0, 1, 0, 0, 0, 0.0],
                           [0, 0, 1, 0, 0, 1, 0, 0, 0, 0, 0.1],
                           [0, 0, 0, 1, 0, 1, 0, 0, 0, 0, 0.0]]

    result = create_embeddings(input_string)

    assert result == expected_embeddings, f"Expected embeddings {expected_embeddings}, got {str(result)}"


def test_create_embeddings_DBR_valid():

    input_string = "2D_AXIAL | ENERGY 1.0 | BOUNDARY N 1 K 0.00 | PAIRS_COUNT 1 { LAYER THICKNESS 5.00 N 3.41 K 0.00 ; LAYER THICKNESS 206.76 N 2.92 K 0.00 } | BOUNDARY N 1.00 K 0.00"

    expected_embeddings = [[1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0.1],
                           [0, 0, 1, 0, 0, 1, 0, 0, 0, 0, 0.1],
                           [0, 0, 0, 1, 0, 1, 0, 0, 0, 0, 0.0],
                           [0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0.01],
                           [0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0.005],
                           [0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 0.341],
                           [0, 0, 0, 1, 0, 0, 1, 0, 0, 0, 0.0],
                           [0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0.20676],
                           [0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 0.292],
                           [0, 0, 0, 1, 0, 0, 1, 0, 0, 0, 0.0],
                           [0, 0, 1, 0, 0, 1, 0, 0, 0, 0, 0.1],
                           [0, 0, 0, 1, 0, 1, 0, 0, 0, 0, 0.0]]

    result = create_embeddings(input_string)

    assert result == expected_embeddings, f"Expected embeddings {expected_embeddings}, got {str(result)}"


def test_create_embeddings_SINGLE_valid():

    input_string = "2D_AXIAL ENERGY 1.0 BOUNDARY N 1.00 K 0.00 LAYER THICKNESS 5.00 N 1.00 GAIN 500.00 BOUNDARY N 1.00 K 0.00"

    expected_embeddings = [[1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0.1],
                           [0, 0, 1, 0, 0, 1, 0, 0, 0, 0, 0.1],
                           [0, 0, 0, 1, 0, 1, 0, 0, 0, 0, 0.0],
                           [0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0.005],
                           [0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0.1],
                           [0, 0, 0, 0, 1, 0, 0, 0, 0, 0, -0.0049331744781739685],
                           [0, 0, 1, 0, 0, 1, 0, 0, 0, 0, 0.1],
                           [0, 0, 0, 1, 0, 1, 0, 0, 0, 0, 0.0]]

    result = create_embeddings(input_string)

    assert result == expected_embeddings, f"Expected embeddings {expected_embeddings}, got {str(result)}"



from src.data.utilities import convert_gain_to_k
def test_convert_gain_to_k():
    TGM = 101320.669343
    kappa_QW = -0.9996650802287824
    ENERGY = 1

    K = convert_gain_to_k(ENERGY, TGM)

    assert K == kappa_QW


def test_process_EQEQ_entries():

    input_string = "EIGEN_ENERGY_1 1 Q1 0.034590 EIGEN_ENERGY_2 1 Q2 0.034592"

    expected_output = [0.1, 0.3782458801678635, 0.1, 0.3782479726894347]

    result = process_EQEQ_entries(input_string)

    assert result == expected_output


def test_process_EQEQ_entries_invalid_1():

    input_string = "EIGEN_ENERGY_1 0.1 Q1 0.034590 EIGEN_ENERGY_2 1 Q2 0.034592"

    expected_output = [0.1, 0.3782479726894347]

    result = process_EQEQ_entries(input_string)

    assert result == expected_output


def test_process_EQEQ_entries_invalid_2():

    input_string = "EIGEN_ENERGY_1 1 Q1 0.034590 EIGEN_ENERGY_2 0.1 Q2 0.034592"

    expected_output = [0.1, 0.3782458801678635]

    result = process_EQEQ_entries(input_string)

    assert result == expected_output


def test_process_EQEQ_entries_invalid_3():

    input_string = "EIGEN_ENERGY_1 0.1 Q1 0.034590 EIGEN_ENERGY_2 0.1 Q2 0.034592"

    expected_output = []

    result = process_EQEQ_entries(input_string)

    assert result == expected_output


def test_process_EQEQ_entries_invalid_4():

    input_string = "EIGEN_ENERGY_1 1 Q1 0.034590"

    expected_output = [0.1, 0.3782458801678635]

    result = process_EQEQ_entries(input_string)

    assert result == expected_output


def test_process_EQEQ_entries_invalid_Q():

    input_string = "EIGEN_ENERGY_1 1 Q1 10000000"

    expected_output = []

    result = process_EQEQ_entries(input_string)

    assert result == expected_output


def test_process_EQEQ_entries_invalid():

    input_string = "EIGEN_ENERGY_1 12 12 12"

    expected_output = []

    result = process_EQEQ_entries(input_string)

    assert result == expected_output


def test_process_gain_energy_entries_valid():

    input_string = "TRESHOLD_MATERIAL_GAIN 100 ENERGY 1 kappa_QW 0.999578"

    expected_output = [0.1, 0.0009866348956347937]

    result = process_gain_energy_entries(input_string)

    assert result == expected_output

def test_process_gain_energy_entries_invalid_energy():

        input_string = "TRESHOLD_MATERIAL_GAIN 100.0 ENERGY 0.05 kappa_QW 0.999578"

        expected_output = []

        result = process_gain_energy_entries(input_string)

        assert result == expected_output

def test_process_gain_energy_entries_invalid_gain():

        input_string = "TRESHOLD_MATERIAL_GAIN -101311.828623 ENERGY 0.005782 kappa_QW 0.999578"

        expected_output = []

        result = process_gain_energy_entries(input_string)

        assert result == expected_output


def test_process_gain_energy_entries_missing_tokens():
        input_string = "ENERGY 0.005782 kappa_QW 0.999578"

        expected_output = []

        result = process_gain_energy_entries(input_string)

        assert result == expected_output


def test_process_gain_energy_entries_extra_tokens():

        input_string = "TRESHOLD_MATERIAL_GAIN 100 EXTRA_TOKEN 123.456 ENERGY 1 kappa_QW 0.999578"

        expected_output = [0.1, 0.0009866348956347937]

        result = process_gain_energy_entries(input_string)

        assert result == expected_output


def test_process_gain_energy_entries_valid2():

    input_string = "TRESHOLD_MATERIAL_GAIN 98279.222633 ENERGY 2.137307 kappa_QW 0.969657"

    expected_output = [0.2137307, 0.4536817151938332]

    result = process_gain_energy_entries(input_string)

    assert result == expected_output
