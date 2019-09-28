class Key(object):
    def __init__(self, period: int, fiat: str, crypto: str) -> None:
        self.period = period
        self.fiat = fiat
        self.crypto = crypto

    def __eq__(self, o: object) -> bool:
        return isinstance(o, type(self)) and (self.period, self.fiat, self.crypto) == (o.period, o.fiat, o.crypto)

    def __hash__(self) -> int:
        return hash((self.period, self.fiat, self.crypto))

    def __str__(self) -> str:
        return str(self.period) + ':' + str(self.fiat) + ':' + str(self.crypto)

    def __repr__(self) -> str:
        return str(self.period) + ':' + str(self.fiat) + ':' + str(self.crypto)


class PreviousPriceHolder(object):
    def __init__(self) -> None:
        self.holder = {}

    def get(self, period: int, fiat: str, crypto: str):
        key = Key(period, fiat, crypto)

        if key in self.holder:
            return self.holder[key]
        else:
            return None

    def insert(self, period: int, fiat: str, crypto: str, price: int):
        key = Key(period, fiat, crypto)
        self.holder[key] = price
