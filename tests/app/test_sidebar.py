 
from ...src.app.sidebar import convert_hour_to_string


def test_convert_hour_to_string():

    input = [
        0, 0.01, 0.02, 0.03, 0.04, 0.05, 0.1, 0.2, 0.25, 0.5, 1, 3.75, 6.6, 
        10.9, 12, 20.5, 24.1, 32.25]
    output = [
        '0:00', '0:01', '0:01', '0:02', '0:02',   '0:03',  '0:06',  '0:12', 
        '0:15', '0:30', '1:00', '3:45', '6:36', '10:54', '12:00', '20:30', 
        '24:06', '32:15']

    for i, e in enumerate(input):
        result = convert_hour_to_string(e)
        assert result == output[i]

