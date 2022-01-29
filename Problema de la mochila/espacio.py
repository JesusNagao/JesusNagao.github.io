class Espacio:
    def __init__(self):
        self.lleno = False
        self.peso = 0.0
        self.ganancia = 0.0

    def llenar(self, peso, ganancia):
        self.lleno = True
        self.peso = peso
        self.ganancia = ganancia

    def get_profit(self):
        return self.ganancia

    def get_weight(self):
        return self.peso

    