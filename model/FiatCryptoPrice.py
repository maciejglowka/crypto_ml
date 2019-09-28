from typing import Tuple


class FiatCryptoPrice(object):
    def __init__(self, fcp: Tuple) -> None:
        self.id = fcp[0]
        self.high = fcp[1]
        self.low = fcp[2]
        self.open = fcp[3]
        self.close = fcp[4]
        self.volume = fcp[5]
        self.date_time = fcp[6]
        self.period = fcp[7]
        self.fiat_code = fcp[8]
        self.crypto_code = fcp[9]
